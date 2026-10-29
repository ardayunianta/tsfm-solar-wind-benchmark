"""Tables, figure and LaTeX macros for the paper, written from the saved results.

Every number in the Results section comes from here; nothing is typed by hand.

--prepare first recomputes what the tables read: the adaptively calibrated 80 % intervals
(12_conformal.py) and the imbalance cost (11_value.py) for the models below, then the test-year
comparison (08_compare.py). Without it, only the tables are rebuilt from the existing files.

Significance: Diebold-Mariano tests on the daily mean absolute error averaged over the series of a
technology, one value per day (metrics.dm_days), with Holm's correction within each table.
Foundation-model rows use the settings chosen on the validation half-year (choose_chronos2.py for
zero-shot Chronos-2, choose_settings.py for TimesFM 2.5, Moirai 2.0, TiRex and fine-tuned Chronos-2);
rows whose forecast file does not exist yet are shown as "--". TimesFM 3.0 (released during the test
year) is an extra row outside the significance tests; it is compared with Chronos-2 and TimesFM 2.5 only in
uncorrected, exploratory tests (macros "exploratory@...").

Usage: python scripts/14_make_tables.py [--prepare]
Output: paper/tables/*.tex, paper/figures/limited_data.{pdf,png}, paper/results_macros.tex,
        and the same tables as CSV in results/metrics/tables/
"""
import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pgtsfm.config import RESULTS, SOLAR_REGIONS, WIND_SERIES  # noqa: E402
from pgtsfm.metrics import dm_days, evaluate, holm  # noqa: E402
from pgtsfm.windows import load  # noqa: E402

TECHS = ("solar", "wind")
TABLES, FIGURES = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
CSV = RESULTS / "metrics" / "tables"
ALPHA = 0.05
PROVINCES = ["Antwerp", "Brussels", "East-Flanders", "Flemish-Brabant", "Hainaut", "Limburg", "Liège",
             "Luxembourg", "Namur", "Walloon-Brabant", "West-Flanders"]

# (key, label, forecast file). "c2:<backbone>" is the Chronos-2 run chosen on validation for that
# backbone, "c2short" the 7-day-history Chronos-2 + weather run used for the limited-data comparison,
# "zs:<model>" the zero-shot run and "ft:<variant>" the fine-tuned Chronos-2 run chosen on validation.
MAIN = [
    ("Reference forecasts", [("persistence", "Persistence", "persistence"),
                             ("clim", "Mean of last 7 days", "climatology7"),
                             ("elia", "Elia day-ahead", "elia_da11")]),
    ("Trained on weather forecasts", [("lgb", "LightGBM", "lightgbm_multi"),
                                      ("nhits", "N-HiTS", "nf_nhits_multi_in1d"),
                                      ("tide", "TiDE", "nf_tide_multi"),
                                      ("tft", "TFT", "nf_tft_multi"),
                                      ("lstm", "LSTM", "nf_lstm_multi")]),
    ("Foundation models, zero-shot", [("bolt", "Chronos-Bolt, power only", "zs_chronos-bolt-base"),
                                      ("c2", "Chronos-2, power only", "c2:chronos-2"),
                                      ("tfm", "TimesFM 2.5, power only", "zs:timesfm-2.5"),
                                      ("moirai", "Moirai 2.0, power only", "zs:moirai-2.0"),
                                      ("tirex", "TiRex, power only", "zs:tirex"),
                                      ("c2cov", "Chronos-2 + weather", "c2:chronos-2-cov"),
                                      ("tfmcov", "TimesFM 2.5 + weather", "zs:timesfm-2.5-cov")]),
    ("Foundation model fine-tuned on the training year", [
        ("c2ft", "Chronos-2 + weather, fine-tuned", "ft:cov")]),
    ("Frozen foundation model in a trained network", [
        ("pgaNone", "PG-Adapter, no prior", "pga_none_multi_365d_d2_ens"),
        ("pgaBolt", "PG-Adapter, Chronos-Bolt prior", "pga_chronos-bolt-base_multi_365d_d2_ens")]),
    ("Setting B: Elia's forecast as an input", [("lgbElia", "LightGBM + Elia", "lightgbm_multi_elia"),
                                                ("c2covElia", "Chronos-2 + weather + Elia", "c2:chronos-2-cov-elia"),
                                                ("c2ftElia", "Chronos-2 + weather + Elia, fine-tuned",
                                                 "ft:cov-elia"),
                                                ("pgaElia", "PG-Adapter, Elia prior", "pga_elia_multi_365d_d2_ens")]),
    ("Extra: released during the test year$^{\\dagger}$", [
        ("tfm3cov", "TimesFM 3.0 + weather", "zs:timesfm-3.0-cov")]),
]
ROWS = {k: (label, spec) for _, rows in MAIN for k, label, spec in rows}
ABLATION = [
    ("pgaNone1", "No prior", "ECMWF", "yes", "pga_none_365d_d2_ens"),
    ("pgaNone", "No prior", "3 models", "yes", "pga_none_multi_365d_d2_ens"),
    ("pgaBolt1", "Chronos-Bolt", "ECMWF", "yes", "pga_chronos-bolt-base_365d_d2_ens"),
    ("pgaBolt", "Chronos-Bolt", "3 models", "yes", "pga_chronos-bolt-base_multi_365d_d2_ens"),
    ("pgaBoltNoPhys", "Chronos-Bolt", "3 models", "no", "pga_chronos-bolt-base_nophys_multi_365d_d2_ens"),
    ("pgaElia1", "Elia (Setting B)", "ECMWF", "yes", "pga_elia_365d_d2_ens"),
    ("pgaElia", "Elia (Setting B)", "3 models", "yes", "pga_elia_multi_365d_d2_ens"),
]
FILES = {k: f for k, *_, f in ABLATION} | {"lgbElia1": "lightgbm_elia"}
CONTRASTS = [  # (key, label, model A, model B); A - B < 0 means A is better
    ("prior3", "Chronos-Bolt prior vs.\\ no prior (3 NWP models)", "pgaBolt", "pgaNone"),
    ("prior1", "Chronos-Bolt prior vs.\\ no prior (ECMWF)", "pgaBolt1", "pgaNone1"),
    ("physics", "Physics proxy vs.\\ none (Chronos-Bolt prior)", "pgaBolt", "pgaBoltNoPhys"),
    ("nwp", "3 NWP models vs.\\ ECMWF (Chronos-Bolt prior)", "pgaBolt", "pgaBolt1"),
    ("c2weather", "Chronos-2: + weather vs.\\ power only", "c2cov", "c2"),
    ("c2lgb", "Chronos-2 + weather vs.\\ LightGBM", "c2cov", "lgb"),
    ("pgaC2", "PG-Adapter (Chronos-Bolt) vs.\\ Chronos-2 + weather", "pgaBolt", "c2cov"),
    ("bElia3", "B: PG-Adapter vs.\\ LightGBM, both + Elia (3 NWP)", "pgaElia", "lgbElia"),
    ("bElia1", "B: PG-Adapter vs.\\ LightGBM, both + Elia (ECMWF)", "pgaElia1", "lgbElia1"),
    ("bC2", "B: Chronos-2 vs.\\ LightGBM, both + weather + Elia", "c2covElia", "lgbElia"),
    ("bPgaC2", "B: PG-Adapter vs.\\ Chronos-2, both + weather + Elia", "pgaElia", "c2covElia"),
    # Added with TimesFM 2.5 and fine-tuning, before any of their test-year forecasts existed
    ("tfmC2", "TimesFM 2.5 vs.\\ Chronos-2, both + weather", "tfmcov", "c2cov"),
    ("ft", "Chronos-2 + weather: fine-tuned vs.\\ zero-shot", "c2ft", "c2cov"),
    ("ftLgb", "Fine-tuned Chronos-2 + weather vs.\\ LightGBM", "c2ft", "lgb"),
    ("pgaFt", "PG-Adapter (Chronos-Bolt) vs.\\ fine-tuned Chronos-2 + weather", "pgaBolt", "c2ft"),
    ("bFt", "B: Chronos-2 + weather + Elia: fine-tuned vs.\\ zero-shot", "c2ftElia", "c2covElia"),
    ("bFtLgb", "B: fine-tuned Chronos-2 vs.\\ LightGBM, both + weather + Elia", "c2ftElia", "lgbElia"),
]
# TimesFM 3.0 was released in August 2026, during the test year, and added after the other test results
# were known. Its row is left out of the significance tests and the bold best values; it is compared with
# the zero-shot models before it only in uncorrected, exploratory tests: (key, model A, model B).
EXTRA = ["tfm3cov"]
EXPLORATORY = [("tfm3C2", "tfm3cov", "c2cov"), ("tfm3Tfm", "tfm3cov", "tfmcov")]
E2_DAYS = (7, 14, 30, 90)
E2_COLS = [("lgb", "LightGBM", "lightgbm_multi_roll{n}d"),
           ("pgaNone", "PG-Adapter, no prior", "pga_none_multi_roll{n}d_d2_ens"),
           ("pgaBolt", "PG-Adapter, Chronos-Bolt", "pga_chronos-bolt-base_multi_roll{n}d_d2_ens"),
           ("c2", "Chronos-2 + weather, zero-shot", "c2short")]
CALIBRATED = ["elia", "lgb", "pgaBolt", "c2cov", "c2ft", "lgbElia", "c2covElia", "c2ftElia", "pgaElia"]
VALUED = ["persistence", "elia", "lgb", "tft", "c2cov", "c2ft", "pgaBolt", "lgbElia", "c2covElia", "c2ftElia",
          "pgaElia"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache
def chronos2():
    try:
        return script("choose_chronos2.py").choices()
    except FileNotFoundError:
        return pd.DataFrame(columns=["tech", "backbone", "role", "model"])


@lru_cache
def chosen(kind):
    """Settings chosen on validation by choose_settings.py: kind "zs" or "ft"."""
    try:
        return getattr(script("choose_settings.py"), f"{kind}_choices")()
    except FileNotFoundError:
        return pd.DataFrame(columns=["tech", "model", "backbone", "variant"])


def resolve(spec, tech):
    """Forecast file name for a row spec, or None if it does not exist (yet)."""
    if spec in ROWS:
        spec = ROWS[spec][1]
    spec = FILES.get(spec, spec)
    if spec[:3] in ("zs:", "ft:"):
        c = chosen(spec[:2])
        c = c[(c["tech"] == tech) & (c["backbone" if spec[:2] == "zs" else "variant"] == spec[3:])]
        if c.empty:
            return None
        spec = c["model"].iloc[0]
    elif spec.startswith("c2"):
        c = chronos2()
        if spec == "c2short":
            c = c[(c["tech"] == tech) & (c["role"] == "short")]
        else:
            c = c[(c["tech"] == tech) & (c["role"] == "best") & (c["backbone"] == spec[3:])]
        if c.empty:
            return None
        spec = c["model"].iloc[0]
    return spec if (RESULTS / "forecasts" / tech / f"{spec}.parquet").exists() else None


@lru_cache(maxsize=64)
def forecast(tech, name):
    return pd.read_parquet(RESULTS / "forecasts" / tech / f"{name}.parquet")


@lru_cache
def compare(tech):
    c = pd.read_csv(RESULTS / "metrics" / f"{tech}_compare.csv")
    return c[c["series"] == "ALL"].set_index("model")


@lru_cache
def daylight(tech):
    if tech != "solar":
        return None
    df = load(tech)
    geo = df[df["series"] == df["series"].iloc[0]]  # same daylight definition as 08_compare.py
    return pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"])


def metric(tech, name, col):
    c = compare(tech)
    return c.loc[name, col] if name is not None and name in c.index else np.nan


def test(tech, a, b):
    """DM statistic and p-value of model a against model b (row keys or file names), or NaNs."""
    fa, fb = resolve(a, tech), resolve(b, tech)
    if fa is None or fb is None:
        return np.nan, np.nan
    return dm_days(forecast(tech, fa), forecast(tech, fb))


# ---------------------------------------------------------------------------------------------
# Formatting

def num(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def millions(x):
    """A parameter count in millions: 205, 1.2, 0.11."""
    return "--" if pd.isna(x) else "0" if x == 0 else f"{x:.0f}" if x >= 10 else f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def mark(stat, p_adj):
    """+ significantly better, - significantly worse than the reference (after Holm)."""
    if pd.isna(p_adj) or p_adj >= ALPHA:
        return ""
    return "$^{+}$" if stat < 0 else "$^{-}$"


# Tables narrow enough for one column of a two-column journal (main_mpce.tex); the others span both columns
NARROW = {"ablation", "transfer", "calibration", "value"}


def write_table(name, body, colspec, head, caption, note=""):
    TABLES.mkdir(parents=True, exist_ok=True)
    # adjustbox only shrinks: a table wider than the text or taller than most of a page is scaled down to fit.
    # \resulttable is "table" unless the main file defines it first (main_mpce.tex: table*, across both columns).
    env = "table" if name in NARROW else r"\resulttable"
    lines = [r"\providecommand{\resulttable}{table}", rf"\begin{{{env}}}[tbp]", r"\centering\small",
             rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
             r"\begin{adjustbox}{max width=\linewidth, max totalheight=0.7\textheight}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule",
             r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def two_tech_head(cols, first="Model"):
    n = len(cols)
    head = rf"{first} & \multicolumn{{{n}}}{{c}}{{Solar}} & \multicolumn{{{n}}}{{c}}{{Wind}} \\"
    rule = rf"\cmidrule(lr){{2-{n + 1}}}\cmidrule(lr){{{n + 2}-{2 * n + 1}}}"
    return [head, rule, " & " + " & ".join(cols * 2) + r" \\"]


MACROS = {}


def save(name, df):
    """Keep a table as CSV and as macros \\R{name}{row}{column} for the text."""
    CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV / f"{name}.csv")
    for row, r in df.iterrows():
        row = unicodedata.normalize("NFKD", str(row)).encode("ascii", "ignore").decode()  # e.g. Liège -> Liege
        for col, v in r.items():
            if not isinstance(v, (float, int, np.floating)) or pd.isna(v):
                continue
            if col.split("-")[-1] in ("p", "p_holm"):
                text = f"$p={v:.3f}$" if v >= 0.001 else "$p<0.001$"  # used in running text
            elif col == "runs":
                text = f"{v:.0f}"
            elif col in ("paramsM", "trainedM"):
                text = millions(v)
            else:
                text = f"{v:.1f}" if any(w in col for w in ("PICP", "PINAW", "ost", "inutes", "hours")) else f"{v:.2f}"
            MACROS[f"{name}@{row}@{col}"] = text


# ---------------------------------------------------------------------------------------------
# Tables

def main_table():
    rows, body = [], []
    for tech in TECHS:
        keys = [k for k in ROWS if k != "elia" and k not in EXTRA]
        tests = {k: test(tech, k, "elia") for k in keys}
        adj = dict(zip(keys, holm([tests[k][1] if not pd.isna(tests[k][1]) else 1.0 for k in keys])))
        for k in ROWS:
            f = resolve(k, tech)
            stat, p = tests.get(k, (np.nan, np.nan))
            rows.append(dict(key=k, tech=tech, nMAE=metric(tech, f, "nMAE"), CRPS=metric(tech, f, "CRPS"),
                             PICP80=metric(tech, f, "PICP80"), DM=stat, p=adj.get(k, np.nan)))
    t = pd.DataFrame(rows)
    wide = t.pivot(index="key", columns="tech")
    wide.columns = [f"{tech}-{m}" for m, tech in wide.columns]
    save("main", wide.loc[list(ROWS)])
    core = t[~t.key.isin(EXTRA)]
    best = {(tech, m): core[core.tech == tech][m].min() for tech in TECHS for m in ("nMAE", "CRPS")}
    for group, members in MAIN:
        body.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{group}}}}} \\")
        for k, label, _ in members:
            cells = []
            for tech in TECHS:
                r = t[(t.tech == tech) & (t.key == k)].iloc[0]
                for m in ("nMAE", "CRPS"):
                    s = num(r[m])
                    s = rf"\textbf{{{s}}}" if r[m] == best[(tech, m)] and k not in EXTRA else s
                    cells.append(s + (mark(r.DM, r.p) if m == "nMAE" else ""))
                cells.append(num(r.PICP80, 1))
            body.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    write_table("main", body, "l" + "r" * 6, two_tech_head(["nMAE", "CRPS", "PICP80"]),
                "Test-year accuracy (October 2025--September 2026). nMAE and CRPS in \\% of capacity (solar: "
                "daylight quarter-hours); PICP80: coverage of the 80\\% interval (\\%, target 80). Best value per "
                "column in bold.",
                "$^{+}$/$^{-}$: daily mean absolute error significantly lower/higher than Elia's (Diebold--Mariano "
                "test on daily errors averaged over regions, Holm-corrected, 5\\%). Foundation-model settings were "
                "chosen on the validation half-year. In Setting B the forecast is issued at 10:00 UTC, after Elia's "
                "forecast is published. $^{\\dagger}$TimesFM 3.0 was released in August 2026, during the test year, "
                "and added after the other results were known, so it is left out of the tests and the bold values. "
                "Its model card lists pretraining data that end before the test year; we could not check this.")


def ablation_tables():
    rows = []
    for k, prior, nwp, phys, f in ABLATION:
        r = dict(key=k)
        for tech in TECHS:
            r[f"{tech}-nMAE"], r[f"{tech}-CRPS"] = metric(tech, f, "nMAE"), metric(tech, f, "CRPS")
        rows.append(r)
    t = pd.DataFrame(rows).set_index("key")
    save("ablation", t)
    body = [f"{prior} & {nwp} & {phys} & " + " & ".join(num(t.loc[k, c]) for c in t.columns) + r" \\"
            for k, prior, nwp, phys, _ in ABLATION]
    head = [r"Prior & NWP & Physics & \multicolumn{2}{c}{Solar} & \multicolumn{2}{c}{Wind} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r" & & & nMAE & CRPS & nMAE & CRPS \\"]
    write_table("ablation", body, "lllrrrr", head,
                "PG-Adapter variants on the test year (mean of three seeds), in \\% of capacity.")

    rows = []
    for key, label, a, b in CONTRASTS:
        for tech in TECHS:
            stat, p = test(tech, a, b)
            rows.append(dict(key=key, tech=tech, label=label, dnMAE=metric(tech, resolve(a, tech), "nMAE")
                             - metric(tech, resolve(b, tech), "nMAE"), DM=stat, p=p))
    t = pd.DataFrame(rows)
    ok = t["p"].notna()
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p"])
    wide = t.pivot(index="key", columns="tech", values=["dnMAE", "DM", "p_holm"])
    wide.columns = [f"{tech}-{m.replace('_holm', '')}" for m, tech in wide.columns]
    save("contrasts", wide.loc[[c[0] for c in CONTRASTS]])
    body = []
    for key, label, *_ in CONTRASTS:
        cells = []
        for tech in TECHS:
            r = t[(t.key == key) & (t.tech == tech)].iloc[0]
            cells += [num(r.dnMAE) + mark(r.DM, r.p_holm), num(r.DM, 1),
                      "--" if pd.isna(r.p_holm) else ("$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}")]
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    # The comparison labels wrap, so the table fits the page without being scaled down
    write_table("contrasts", body, r">{\raggedright\arraybackslash}p{0.4\linewidth}" + "r" * 6,
                two_tech_head([r"$\Delta$nMAE", "DM", "$p$"], first="Comparison (A vs.\\ B)"),
                "Paired comparisons on the test year: $\\Delta$nMAE = nMAE(A) $-$ nMAE(B) in percentage points "
                "(negative: A is better), Diebold--Mariano statistic on daily errors averaged over regions, and "
                "Holm-corrected $p$-value over all rows and both technologies.",
                "$^{+}$/$^{-}$: A significantly better/worse than B at 5\\%. B: Setting B (Elia's forecast as an input).")


def exploratory_tests():
    """Uncorrected DM tests of the extra TimesFM 3.0 row; macros only, quoted in the text as exploratory."""
    rows = []
    for key, a, b in EXPLORATORY:
        for tech in TECHS:
            stat, p = test(tech, a, b)
            rows.append(dict(key=key, tech=tech, dnMAE=metric(tech, resolve(a, tech), "nMAE")
                             - metric(tech, resolve(b, tech), "nMAE"), DM=stat, p=p))
    wide = pd.DataFrame(rows).pivot(index="key", columns="tech", values=["dnMAE", "DM", "p"])
    wide.columns = [f"{tech}-{m}" for m, tech in wide.columns]
    save("exploratory", wide.loc[[e[0] for e in EXPLORATORY]])


def limited_data():
    rows = []
    for tech in TECHS:
        for n in E2_DAYS:
            ref = "lightgbm_multi_roll{n}d".format(n=n)
            for k, _, pattern in E2_COLS:
                f = resolve(pattern.format(n=n), tech)
                stat, p = (np.nan, np.nan) if k == "lgb" else test(tech, f or "missing", ref)
                rows.append(dict(tech=tech, N=n, key=k, nMAE=metric(tech, f, "nMAE"), DM=stat, p=p))
        for k, f in [("lgb", "lightgbm_multi"), ("pgaNone", "pga_none_multi_365d_d2_ens"),
                     ("pgaBolt", "pga_chronos-bolt-base_multi_365d_d2_ens"), ("c2", "c2short")]:
            rows.append(dict(tech=tech, N=365, key=k, nMAE=metric(tech, resolve(f, tech), "nMAE")))
    t = pd.DataFrame(rows)
    ok = t["p"].notna()
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p"])
    wide = t.pivot(index="N", columns=["tech", "key"], values="nMAE")
    wide.columns = [f"{tech}-{k}" for tech, k in wide.columns]
    pw = t[ok].pivot(index="N", columns=["tech", "key"], values="p_holm")
    pw.columns = [f"{tech}-{k}-p" for tech, k in pw.columns]
    save("limited", wide.join(pw))
    body = []
    for n in list(E2_DAYS) + [365]:
        cells = []
        for tech in TECHS:
            for k, *_ in E2_COLS:
                r = t[(t.tech == tech) & (t.N == n) & (t.key == k)].iloc[0]
                cells.append(num(r.nMAE) + (mark(r.DM, r.p_holm) if n != 365 else ""))
        label = f"{n} days" if n != 365 else r"Full year$^{\ast}$"
        body.append((r"\midrule " if n == 365 else "") + f"{label} & " + " & ".join(cells) + r" \\")
    cols = ["LGBM", "PGA", "PGA-CB", "C2"]
    write_table("limited", body, "l" + "r" * 8, two_tech_head(cols, first="History"),
                "Limited data: test-year nMAE (\\% of capacity) when every model is retrained at the start of each "
                "test month on the $N$ days before it. LGBM: LightGBM; PGA: PG-Adapter without prior; PGA-CB: "
                "PG-Adapter with Chronos-Bolt prior (mean of three seeds); C2: Chronos-2 + weather, zero-shot with "
                "7 days of history (needs no training data, so it is the same in every row).",
                "$^{+}$/$^{-}$: significantly better/worse than LightGBM with the same history (Diebold--Mariano, "
                "Holm-corrected over the table, 5\\%). $^{\\ast}$Full year: one model trained before the test year "
                "(not retrained monthly).")
    return t


def limited_figure(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f1f1e", "#6b6a64"
    style = {"lgb": ("LightGBM", "#2a78d6", "o"), "pgaNone": ("PG-Adapter, no prior", "#eb6834", "s"),
             "pgaBolt": ("PG-Adapter, Chronos-Bolt", "#1baf7a", "^"),
             "c2": ("Chronos-2 + weather (zero-shot)", "#eda100", "D")}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, tech in zip(axes, TECHS):
        d = t[(t.tech == tech) & (t.N != 365)]
        for k, (label, color, marker) in style.items():
            s = d[d.key == k].sort_values("N")
            if s["nMAE"].notna().any():
                ax.plot(s.N, s.nMAE, color=color, lw=1.5, marker=marker, ms=5, label=label,
                        ls="--" if k == "c2" else "-")
        elia = metric(tech, "elia_da11", "nMAE")
        ax.axhline(elia, color=muted, lw=1, ls=":")
        ax.annotate("Elia day-ahead", (90, elia), xytext=(0, 3), textcoords="offset points", ha="right",
                    va="bottom", color=muted, fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(E2_DAYS, [str(n) for n in E2_DAYS])
        ax.minorticks_off()
        ax.set_xlabel("Days of training data")
        ax.set_title(tech.capitalize(), color=ink, fontsize=9, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    axes[0].set_ylabel("nMAE (% of capacity)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"limited_data.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def transfer_table():
    tech, rows = "solar", []
    models = {"held": "pga_chronos-bolt-base_multi_365d_d2_lo-{p}_ens", "full": "pgaBolt", "c2": "c2cov",
              "lgb": "lgb", "elia": "elia"}
    for prov in PROVINCES:
        files = {k: resolve(v.format(p=prov), tech) for k, v in models.items()}
        parts = {k: forecast(tech, f)[forecast(tech, f)["series"] == prov] for k, f in files.items() if f}
        keys = set.intersection(*[set(zip(g["series"], g["ds"])) for g in parts.values()])
        r = dict(province=prov, **dict.fromkeys(models, np.nan))
        for k, g in parts.items():
            g = g[[(s, d) in keys for s, d in zip(g["series"], g["ds"])]]
            r[k] = evaluate(g, daylight(tech))["nMAE"]
            parts[k] = g
        r["DM"], r["p"] = dm_days(parts["held"], parts["elia"])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("province")
    t["p_holm"] = holm(t["p"])
    t.loc["Mean"] = t.drop(columns=["DM", "p", "p_holm"]).mean()
    save("transfer", t)
    body = []
    for prov, r in t.iterrows():
        cells = [num(r.held) + ("" if prov == "Mean" else mark(r.DM, r.p_holm))] + \
                [num(r[k]) for k in ("full", "c2", "lgb", "elia")]
        body.append((r"\midrule " if prov == "Mean" else "") + f"{prov} & " + " & ".join(cells) + r" \\")
    head = [r"Province & \multicolumn{1}{c}{Held out} & \multicolumn{1}{c}{All provinces} & "
            r"\multicolumn{1}{c}{Chronos-2} & \multicolumn{1}{c}{LightGBM} & \multicolumn{1}{c}{Elia} \\"]
    write_table("transfer", body, "lrrrrr", head,
                "Solar provinces never seen in training: nMAE (\\% of capacity, daylight) of PG-Adapter (Chronos-Bolt "
                "prior, three seeds) trained on the other ten provinces, compared with the same model trained on all "
                "provinces, Chronos-2 + weather (zero-shot), LightGBM trained on all provinces, and Elia.",
                "$^{+}$/$^{-}$: held-out model significantly better/worse than Elia in that province "
                "(Diebold--Mariano, Holm-corrected over the 11 provinces, 5\\%).")


def coverage(fc):
    f = fc.dropna(subset=["y"])
    f = f[f["q90"] > 0]  # daylight for solar (night forecasts are exactly 0), as in 12_conformal.py
    inside = (f["y"] >= f["q10"]) & (f["y"] <= f["q90"])
    season = f["ds"].dt.month.map(lambda m: SEASONS[(m % 12) // 3])
    out = {s: 100 * inside[season == s].mean() for s in SEASONS}
    out["Year"], out["Width"] = 100 * inside.mean(), 100 * (f["q90"] - f["q10"]).mean()
    return out


def calibration_table():
    rows, body = [], []
    for tech in TECHS:
        body.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{tech.capitalize()}}}}} \\")
        for k in CALIBRATED:
            f = resolve(k, tech)
            for aci in (False, True):
                name = f"{f}_aci" if (f and aci) else f
                ok = name is not None and (RESULTS / "forecasts" / tech / f"{name}.parquet").exists()
                c = coverage(forecast(tech, name)) if ok else dict.fromkeys(SEASONS + ["Year", "Width"], np.nan)
                rows.append(dict(key=f"{tech}-{k}{'-aci' if aci else ''}", **{f"PICP80-{s}": v for s, v in c.items()
                                                                              if s != "Width"}, PINAW80=c["Width"]))
                label = ROWS[k][0] + (" + conformal" if aci else "")
                body.append(rf"\quad {label} & " + " & ".join(num(c[s], 1) for s in SEASONS + ["Year", "Width"])
                            + r" \\")
    save("calibration", pd.DataFrame(rows).set_index("key"))
    head = [r"Model & DJF & MAM & JJA & SON & Year & Width \\"]
    write_table("calibration", body, "lrrrrrr", head,
                "Coverage of the 80\\% interval by season (\\%, target 80) and its mean width (\\% of capacity), "
                "without and with online adaptive conformal calibration. Solar: quarter-hours with a 90\\% quantile "
                "above zero, so the yearly coverage differs slightly from Table~\\ref{tab:main}. PG-Adapter: "
                "mean of three seeds.")


def value_table():
    v = pd.read_csv(RESULTS / "metrics" / "value.csv").set_index(["tech", "model"])
    rows, body = [], []
    for tech in TECHS:
        body.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{tech.capitalize()}}}}} \\")
        for k in VALUED:
            f = resolve(k, tech)
            r = v.loc[(tech, f)] if f and (tech, f) in v.index else pd.Series(dtype=float)
            vals = [r.get("imbalance_cost_MEUR", np.nan), r.get("cost_EUR_per_MWh", np.nan),
                    r.get("abs_spread_cost_MEUR", np.nan)]
            rows.append(dict(key=f"{tech}-{k}", cost=vals[0], costMWh=vals[1], absCost=vals[2]))
            body.append(rf"\quad {ROWS[k][0]} & " + " & ".join(num(x, 1) for x in vals) + r" \\")
    save("value", pd.DataFrame(rows).set_index("key"))
    head = [r"Model & Cost (M EUR) & EUR/MWh & Sign-blind (M EUR) \\"]
    write_table("value", body, "lrrr", head,
                "Imbalance cost of selling the point forecast day-ahead over the test year, relative to a perfect "
                "forecast, at Belgian day-ahead and imbalance prices; the sign-blind cost charges every deviation the "
                "absolute price spread. PG-Adapter: mean of three seeds.")


def grid_range(key, tech):
    """Fine-tuned rows: the shortest and longest fine-tuning among the grid runs that differ from the chosen one
    only in the learning rate, as recorded in their train_info.json."""
    name = resolve(key, tech) if key in ("c2ft", "c2ftElia") else None
    if not name:
        return {}
    same = re.sub(r"_lr[0-9.e-]+_", "_", name)
    runs = (RESULTS / "models" / "chronos2_ft" / tech).glob("*/train_info.json")
    minutes = [json.loads(p.read_text())["minutes"] for p in runs if re.sub(r"_lr[0-9.e-]+_", "_", p.parent.name) == same]
    return dict(gridMinMinutes=min(minutes), gridMaxMinutes=max(minutes))


def compute_table():
    """Wall-clock cost of every model from 17_timing.py, and the hours the fine-tuning grid took."""
    path = RESULTS / "metrics" / "timing.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    tech = t["tech"].iloc[0]
    mfile = RESULTS / "metrics" / "timing_machine.json"  # written when 17_timing.py finishes
    machine = json.loads(mfile.read_text()) if mfile.exists() else {"chip": "machine not recorded", "memory_gb": 0}
    pfile = RESULTS / "metrics" / "parameters.csv"  # written by 18_parameters.py
    params = (pd.read_csv(pfile, index_col="key")[["total", "trained"]] / 1e6 if pfile.exists()
              else pd.DataFrame(columns=["total", "trained"]))
    own_train = ["PG-Adapter training (own clock)", "fine-tuning (train_info.json)"]
    own_forecast = ["PG-Adapter test forecast (own clock)", "test forecast"]
    rows, body = [], []
    for group, members in MAIN:
        keys = [(k, label) for k, label, _ in members if k in set(t["key"])]
        if keys:
            body.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{group}}}}} \\")
        for k, label in keys:
            g = t[t["key"] == k]
            total = g.loc[g["counted"], "seconds"].sum() / 60
            train = g.loc[g["step"].isin(own_train), "seconds"].sum() / 60 if g["step"].isin(own_train).any() \
                else np.nan
            if k in ("lgb", "lgbElia", "nhits", "tide", "tft", "lstm"):
                forecast = np.nan  # trained and forecast in one call
            elif g["step"].isin(own_forecast).any():
                forecast = g.loc[g["step"].isin(own_forecast), "seconds"].sum() / 60
            else:
                forecast = total  # zero-shot: nothing to train
            p_total, p_trained = params.loc[k] if k in params.index else (np.nan, np.nan)
            rows.append(dict(key=k, paramsM=p_total, trainedM=p_trained, minutes=total, trainMinutes=train,
                             forecastMinutes=forecast, **grid_range(k, tech)))
            cells = [g["device"].iloc[0].removesuffix(" (MPS)"), millions(p_total), millions(p_trained)] + [
                "--" if pd.isna(x) else ("$<$0.1" if x < 0.1 else num(x, 1)) for x in (train, forecast, total)]
            body.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    save("compute", pd.DataFrame(rows).set_index("key"))
    # Every fine-tuning run of the grid (both settings, input sets and learning rates), as recorded when it ran
    info = [json.loads(p.read_text()) for p in (RESULTS / "models" / "chronos2_ft").glob("*/*/train_info.json")]
    grid = pd.DataFrame([dict(tech=i["args"]["tech"], minutes=i["minutes"]) for i in info])
    save("computeGrid", pd.DataFrame({"hours": grid.groupby("tech")["minutes"].sum() / 60,
                                      "runs": grid.groupby("tech").size()}))
    n = len(SOLAR_REGIONS) if tech == "solar" else len(WIND_SERIES)
    head = [r" & & \multicolumn{2}{c}{Parameters (M)} & \multicolumn{3}{c}{Minutes} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
            r"Model & Device & Total & Trained & Training & Forecast & Total \\"]
    write_table("compute", body, "llrrrrr", head,
                f"Model size and compute cost on the {n} {tech} series: parameters, and wall-clock minutes to train "
                f"and to forecast the test year from the processed data, measured one job at a time on a laptop "
                f"({machine['chip']}, {machine['memory_gb']:.0f}~GB) with scripts/17\\_timing.py and "
                f"18\\_parameters.py.",
                "Parameters: all those used to make a forecast, and those trained on our data, in millions. "
                "Zero-shot models train none; fine-tuned Chronos-2 trains a LoRA adapter; PG-Adapter with the "
                "Chronos-Bolt prior includes the frozen backbone; LightGBM, a tree ensemble, has no comparable count. "
                "GPU: the laptop's integrated GPU, through PyTorch's MPS backend. "
                "Training: the training loop (PG-Adapter, its own clock) or the LoRA fine-tuning (Chronos-2). "
                "Minutes --: not measured separately, because LightGBM and the neural models train and forecast in one call. "
                "Total also includes loading data and models and each pipeline's own data preparation, so small "
                "differences partly reflect our code, and, for PG-Adapter with the Chronos-Bolt prior, the prior on "
                "all days. One seed; PG-Adapter's reported forecasts average three. Choosing settings on "
                "the validation half-year is not included. $^{\\dagger}$Outside the main comparison "
                "(Table~\\ref{tab:main}).")


def write_macros():
    lines = [r"% Written by scripts/14_make_tables.py. Use \R{table}{row}{column}, e.g. \R{main}{pgaElia}{solar-nMAE}.",
             # A name that does not exist prints a bold ?? instead of nothing
             r"\providecommand{\R}[3]{\ifcsname r@#1@#2@#3\endcsname\csname r@#1@#2@#3\endcsname"
             r"\else\textbf{??}\fi}"]
    lines += [rf"\expandafter\def\csname r@{k}\endcsname{{{v}}}" for k, v in sorted(MACROS.items())]
    (ROOT / "paper" / "results_macros.tex").write_text("\n".join(lines) + "\n")


def prepare():
    conformal, value, comp = script("12_conformal.py"), script("11_value.py"), script("08_compare.py")
    for tech in TECHS:
        conformal.main(tech, [f for f in (resolve(k, tech) for k in CALIBRATED) if f])
    value.main({f for tech in TECHS for f in (resolve(k, tech) for k in VALUED) if f})
    for tech in TECHS:
        comp.main(tech)
    compare.cache_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="recompute calibration, cost and comparison first")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    main_table()
    ablation_tables()
    limited_figure(limited_data())
    transfer_table()
    calibration_table()
    value_table()
    compute_table()
    exploratory_tests()
    write_macros()
    missing = [f"{tech}:{k}" for tech in TECHS for k in ROWS if resolve(k, tech) is None]
    print(f"Wrote {len(list(TABLES.glob('*.tex')))} tables, the figure and {len(MACROS)} macros."
          + (f" Not available yet: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
