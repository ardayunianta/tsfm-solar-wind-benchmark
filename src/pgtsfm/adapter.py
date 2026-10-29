"""PG-Adapter: a small network that turns a frozen foundation-model prior into a
weather- and physics-aware probabilistic day-ahead forecast.

Per target day the inputs are, for each of the 96 quarter-hours:
  prior     (96, Qp)  quantiles from the frozen time-series foundation model
  covariate (96, F)   weather forecast, solar geometry, calendar, recent-power summaries
The output is (96, Q) non-crossing quantiles of the load factor in [0, 1].

Physics enters in three places:
  * a physics proxy for the expected output (solar: clear-sky-index x clear-sky shape;
    wind: a learned monotone power curve of the 100 m wind-speed forecast),
  * a learned gate that blends the foundation-model prior with the physics proxy per step,
  * hard constraints: output in [0, 1] of capacity and zero solar output at night.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class MonotonePowerCurve(nn.Module):
    """Load factor as a sum of increasing sigmoids of wind speed (m/s): monotone by construction."""

    def __init__(self, n=4):
        super().__init__()
        self.centres = nn.Parameter(torch.linspace(4.0, 12.0, n))
        self.log_scales = nn.Parameter(torch.zeros(n))
        self.logit_weights = nn.Parameter(torch.zeros(n))

    def forward(self, ws):
        w = torch.softmax(self.logit_weights, 0)
        s = F.softplus(self.log_scales) + 0.3
        return (w * torch.sigmoid((ws.unsqueeze(-1) - self.centres) / s)).sum(-1)


class SolarProxy(nn.Module):
    """Expected PV load factor = k * clear-sky index * normalised clear-sky irradiance * temperature derate."""

    def __init__(self):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(math.log(0.8)))
        self.beta = nn.Parameter(torch.tensor(0.004))

    def forward(self, ghi_fc, ghi_clear, t2m):
        derate = 1 - self.beta * (t2m - 25.0)
        return torch.exp(self.log_k) * (ghi_fc / 1000.0) * derate


class PGAdapter(nn.Module):
    def __init__(self, n_prior, n_cov, quantiles, tech, d_model=64, n_layers=2, n_heads=4, dropout=0.1,
                 use_prior=True, use_physics=True, prior_mid=None, clip_train=False):
        super().__init__()
        self.tech, self.use_prior, self.use_physics, self.clip_train = tech, use_prior, use_physics, clip_train
        # Which prior column is the median used as the base forecast (default: the middle one).
        self.prior_mid = n_prior // 2 if prior_mid is None else prior_mid
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))
        self.mid = int(torch.argmin((self.q - 0.5).abs()))
        n_in = n_cov + (n_prior if use_prior else 0) + (1 if use_physics else 0)
        self.inp = nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.tod = nn.Embedding(96, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, n_layers)
        self.gate = nn.Linear(d_model, 1)
        self.resid = nn.Linear(d_model, 1)
        self.spread = nn.Linear(d_model, len(quantiles) - 1)
        self.phys = MonotonePowerCurve() if tech == "wind" else SolarProxy()

    def physics_proxy(self, raw):
        if self.tech == "wind":
            return self.phys(raw["ws100"] / 3.6)  # km/h -> m/s
        return self.phys(raw["ghi"], raw["ghi_clear"], raw["t2m"])

    def forward(self, prior, cov, raw, night_mask=None):
        """prior (B,96,Qp) or None; cov (B,96,F) scaled features; raw: dict of unscaled (B,96) tensors."""
        parts = [cov]
        prior_mid = None
        if self.use_prior:
            parts.append(prior)
            prior_mid = prior[..., self.prior_mid]
        phys = None
        if self.use_physics:
            phys = self.physics_proxy(raw).clamp(0, 1.2)
            parts.append(phys.unsqueeze(-1))
        h = self.inp(torch.cat(parts, -1)) + self.tod.weight.unsqueeze(0)
        h = self.mixer(h)
        # Base forecast: blend of prior and physics proxy, chosen per step by a learned gate.
        if prior_mid is not None and phys is not None:
            g = torch.sigmoid(self.gate(h)).squeeze(-1)
            base = g * prior_mid + (1 - g) * phys
        elif prior_mid is not None:
            base = prior_mid
        elif phys is not None:
            base = phys
        else:
            base = torch.zeros(h.shape[:2], device=h.device)
        median = base + self.resid(h).squeeze(-1)
        # Non-crossing quantiles: cumulative positive gaps below and above the median.
        gaps = F.softplus(self.spread(h)) + 1e-4
        lo = gaps[..., :self.mid].flip(-1).cumsum(-1).flip(-1)
        hi = gaps[..., self.mid:].cumsum(-1)
        qs = torch.cat([median.unsqueeze(-1) - lo, median.unsqueeze(-1), median.unsqueeze(-1) + hi], -1)
        # Clipping to [0, 1] gives a zero gradient outside the range, so the training loss uses the
        # unclipped quantiles and a median pushed below 0 can recover. (clip_train=True is the first
        # version, which let some limited-data models collapse to an all-zero forecast.) The forecast
        # is always clipped, which never raises the pinball loss of a target in [0, 1].
        if self.clip_train or not self.training:
            qs = qs.clamp(0.0, 1.0)
        if night_mask is not None:
            qs = qs.masked_fill(night_mask.unsqueeze(-1), 0.0)
        return qs


def pinball_loss(qs, y, levels):
    """Mean pinball loss over valid targets. qs (B,96,Q), y (B,96) with NaN for missing."""
    mask = ~torch.isnan(y)
    y = torch.nan_to_num(y)
    d = y.unsqueeze(-1) - qs
    loss = torch.maximum(levels * d, (levels - 1) * d)
    return (loss.mean(-1) * mask).sum() / mask.sum().clamp(min=1)
