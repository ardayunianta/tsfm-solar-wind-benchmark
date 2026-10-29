# Research design: do time-series foundation models improve day-ahead solar and wind forecasts? A benchmark against a transmission system operator

Status: draft v2, 26 Oct 2026 (v1: 23 Oct 2026). Topic and design choices below are Claude's calls unless marked as yours.

*Changed on 2026-10-26 (your decision):* the paper is now framed as a benchmark instead of a new-method paper.
With the full results in, the foundation-model prior adds little to PG-Adapter, and PG-Adapter beats the
operator only when the operator's own forecast is one of its inputs. A method paper would have claimed more than
the evidence supports. The benchmark compares three ways of using a foundation model (zero-shot, zero-shot with
weather covariates, frozen prior inside PG-Adapter), and two baselines were added for that: Chronos-2 zero-shot
with covariates, and LightGBM with the operator's forecast as an input.

## 1. Why this topic

Evidence from the citation lists we built (sheet "DL x RE 2022-2026"):

- Most highly cited deep-learning × renewable-energy papers since 2022 are hybrids like "decomposition (VMD/EMD) + CNN-LSTM/GRU + attention + metaheuristic optimiser". Among the 1,049 highly cited candidates, 199 titles use CNN/LSTM/GRU hybrids and 82 use decomposition. Reviewers at top journals now treat another such hybrid as incremental.
- Under-explored among highly cited work: foundation / pretrained models (1 title), transfer and few-shot learning (16), physics-guided deep learning (28), and decision value (2).
- 2024–2026 journals are starting to publish foundation-model and LLM forecasting for renewables: STELLM (Applied Energy 2024, 78 Crossref citations), SolarFM (Applied Energy 2026), and several LLM wind papers in Renewable Energy (2025–2026). The topic is clearly accepted by top journals, but open questions remain:
  1. Day-ahead renewable output is driven by weather. A model that only sees past power cannot know tomorrow's clouds or wind, so zero-shot foundation models need weather forecasts to be useful. How best to inject them is open.
  2. New plants and regions have little history. Foundation models promise good forecasts with little data, but this is rarely tested against a real operator's forecast.
  3. Most papers report point accuracy only, not calibrated uncertainty or money saved.

## 2. Research questions

- **RQ1:** How do zero-shot time-series foundation models (TSFMs), with and without weather covariates, compare with (a) the grid operator's own day-ahead forecast and (b) models trained per region?
- **RQ2:** Does a TSFM prior improve a trained model? Tested with PG-Adapter with and without the prior, and in Setting B (operator's forecast as input) against LightGBM and Chronos-2 given the same inputs.
- **RQ3:** Do TSFMs help when a region has only 7–90 days of history, or none (a province left out of training)?
- **RQ4:** Are the probabilistic forecasts calibrated across seasons (with adaptive conformal calibration), and what are the errors worth at real Belgian imbalance prices?

(v1 had calibration as RQ3, value as RQ4 and little data inside RQ2.)

We report results as they come out, including negative ones (for example, if zero-shot TSFMs lose to the operator forecast).

## 3. One of the compared approaches: PG-Adapter

(In v1 this was the paper's proposed method; it is now the "frozen prior inside a trained model" way of using a TSFM.)
A frozen pretrained TSFM provides a prior forecast from recent power history. A small trainable network corrects it using tomorrow's weather forecast and simple physics:

1. **Backbone (frozen):** Chronos-Bolt or Chronos-2 / TimesFM. Input: the last 7 days of normalised power (load factor = power / monitored capacity). Output: prior quantiles for the next day (96 × 15-min).
2. **Physics features:**
   - Solar: clear-sky irradiance and solar zenith (pvlib), NWP clear-sky index, temperature derating.
   - Wind: NWP 100 m wind speed passed through a learned monotone power-curve layer; wind direction as sine/cosine.
3. **Fusion network (trainable, ~0.1–1 M parameters):** a gated residual network with attention across the 96 future steps. Inputs: prior quantiles, physics features, raw NWP and calendar features.
4. **Non-crossing quantile head:** quantiles built as a base value plus cumulative positive increments; the forecast is clipped to [0, 1] of capacity. Solar is forced to 0 when the sun is below the horizon.
5. **Loss:** pinball loss over quantiles 0.05–0.95, computed on the quantiles before clipping. (The physics-consistency penalty planned here was not implemented; the physics enters through the proxy and the constraints.) *Changed on 2026-10-26:* the first version computed the loss after clipping. A clipped quantile has no gradient, and in the rolling E2 runs about a tenth of the series-months collapsed to an all-zero forecast (low-output months only; no full-data or transfer run collapsed). The fix was chosen on the 12 months before the test year (`scripts/select_unclipped.sh`: clipped loss 421 of 1,728 series-months at zero, unclipped none, lower error in 17 of 18 paired runs), and every PG-Adapter run was retrained. Old runs are in `results/forecasts/<tech>/superseded/clipped_loss/`.
6. **Calibration:** adaptive conformal inference applied after the model, updated daily, so 80 % intervals keep about 80 % coverage as seasons change.

Why this counts as "new / improved deep learning": the paper does not stack another hybrid. It proposes a general way to adapt foundation models to weather-driven targets with physics and little data, and tests it rigorously against an operational benchmark.

## 4. Data (all public)

| Source | What | Resolution | Period used |
|---|---|---|---|
| Elia Open Data `ods032` | Solar: measured output, Elia day-ahead (11h) forecast with P10/P90, monitored capacity, for 11 Belgian provinces | 15 min | 2024-03-15 → 2026-09-30 |
| Elia Open Data `ods031` | Wind: same fields for offshore, onshore Flanders and onshore Wallonia (Elia- and DSO-connected), 5 series | 15 min | same |
| Open-Meteo Previous Runs API | Archived ECMWF IFS 0.25° forecasts issued about 2 days before the valid time (`previous_day2`): irradiance (GHI/direct/diffuse), cloud cover, temperature, 10 m / 100 m wind, wind direction | hourly → 15 min | same (archive starts Mar 2024) |
| Elia Open Data `ods134` | Imbalance prices per quarter-hour | 15 min | test year |

Leakage rule: forecasts are issued at 09:00 UTC on day D-1 for day D (00:00–23:45 UTC). Power history is cut at issue time. Weather uses the `previous_day2` forecasts, which were all available before 09:00 UTC on D-1. This is slightly pessimistic, because Elia uses fresher weather runs. A `previous_day1` run is reported only as an optimistic sensitivity check.

Splits (chronological):

- Train: 2024-04-01 → 2025-03-31
- Validation: 2025-04-01 → 2025-09-30
- Test: 2025-10-01 → 2026-09-30, a full year never used for any tuning

Final models are retrained on train + validation before testing.

## 5. Baselines

1. Persistence (last 24 h before issue time)
2. Diurnal climatology (14-day profile)
3. **Elia's operational day-ahead forecast**, including its P10/P90 (the industry benchmark)
4. LightGBM quantile regression on weather and calendar features (strong non-deep baseline)
5. Deep learning trained per region, with weather covariates: LSTM seq2seq, TFT, N-HiTS, TiDE (via neuralforecast)
6. Zero-shot TSFMs:
   - Chronos-Bolt (power history only)
   - Chronos-2: power history only, with weather covariates, and with weather covariates plus the operator's forecast (Setting B). History length (7 or 28 days) and NWP set chosen on the validation half-year (`scripts/select_chronos2.sh`). Chosen (lowest validation CRPS, `results/metrics/diagnostics/chronos2_selection.csv`): 28 days for every variant; all three NWP models, except wind with the operator's forecast (ECMWF only). A 7-day, three-model run with weather is also kept for the limited-data comparison.
   - *Added on 2026-10-27* (`scripts/16_other_tsfms.py`, `scripts/run_more_tsfms.sh`): TimesFM 2.5, Moirai 2.0 (small) and TiRex from power history only, and TimesFM 2.5 with the same weather covariates as Chronos-2 (its in-context linear regression, "xreg + timesfm"). History length (7 or 28 days) and, for TimesFM 2.5 with weather, the NWP set are chosen by the lowest validation CRPS (`scripts/choose_settings.py`, `results/metrics/diagnostics/other_tsfm_selection.csv`). These models output the quantiles 0.1–0.9; 0.05 and 0.95 are set to the 0.1 and 0.9 forecasts, as Chronos-Bolt does. Moirai 2.0 gets no covariates: its paper (arXiv 2511.11698) drops covariate support, and a check on wind validation with all three NWP models was worse than without them (`results/metrics/diagnostics/moirai_covariates_check.csv`). TiRex takes no covariates. Chosen: 28 days of history for every model and both technologies; all three NWP models for TimesFM 2.5 with weather.
   - *Extra row, added on 2026-10-27 after the other test results were known:* TimesFM 3.0 with the same weather covariates (read as extra input series by the model itself), same grid and selection rule (`16_other_tsfms.py`, model `timesfm-3.0-cov`). Released in August 2026, during the test year; its model card lists pretraining data that end before the test year (GIFT-Eval pretraining corpus, October 2024; Wikipedia page views to November 2023; Google Trends to 2022; synthetic series), which we cannot check. Therefore a separate row in the main table, outside the Holm families and the bold values; compared with Chronos-2 and TimesFM 2.5 + weather only in uncorrected, exploratory DM tests (`results/metrics/tables/exploratory.csv`). Weights under the TimesFM Non-Commercial License v1.0, accepted by the author for this research use and downloaded on 2026-10-27 (Hugging Face revision 43046b85ec22d584a13f8098c2ed39c889e129c2). Chosen: 28 days of history with all three NWP models for both technologies.
8. *Added on 2026-10-27:* Chronos-2 fine-tuned on the training year (`scripts/15_finetune_chronos2.py`), with weather covariates (Setting A) and with weather plus Elia's forecast (Setting B). LoRA (the package default: rank 8 on the attention and output layers), 1000 steps, checkpoint with the lowest loss on every 4th validation day (checked every 100 steps). Grid, chosen by validation CRPS (`results/metrics/diagnostics/finetune_selection.csv`): learning rate 1e-5, 1e-4 or 3e-4 (3e-4 added on 2026-10-28, see below) × {7 days of history with all three NWP models, 128 rows per batch; 28 days with ECMWF only, 32 rows per batch}. Not run on the 16 GB laptop: full fine-tuning (10.6 s/step against 2.5 for LoRA, so about 3 h per run) and 28 days with all three NWP models (out of memory at 64 rows per batch). Unlike the trained baselines, the fine-tuned model is not refitted on training + validation.
   - *Changed on 2026-10-28 (your decision):* with the first grid, 1e-4 won all eight validation comparisons (two settings × two technologies × two input sets), and on wind the 1e-4 runs reached their best checkpoint after 300–600 of 1,000 steps, so the best rate may lie above the grid. One more rate, 3e-4, was added for all eight configurations. This was decided before any fine-tuned test-year forecast was scored: the four test forecasts with 1e-4 existed but had not been evaluated. One extension only: if 3e-4 wins, it is reported as still at the edge of the grid and the grid is not extended again. The test year is forecast again only where the choice changes; the unused 1e-4 test forecasts are kept but not scored.
   - *Result (2026-10-29):* 3e-4 beat 1e-4 in three of the eight configurations. Chosen: with weather only (Setting A), 7 days with all three NWP models at 3e-4 for both technologies (for wind this replaced 28 days with ECMWF at 1e-4), so this choice is again at the edge of the grid, with validation CRPS 2.788 against 2.793 (solar) and 3.431 against 3.438 (wind) for the best 1e-4 run; with Elia's forecast (Setting B), 28 days with ECMWF only at 1e-4 for both technologies. As fixed above, the grid was not extended again. The two unused 1e-4 test forecasts were moved, unscored, to `results/forecasts/<tech>/superseded/finetune_grid/`.
9. Setting B (operator's forecast as an input): PG-Adapter with the Elia prior, LightGBM with Elia's P10/P50/P90 as features (`05_lightgbm.py ... elia`), and Chronos-2 (zero-shot and fine-tuned) with Elia's forecast as a covariate.
10. Paired tests added with these models, fixed before any of their test-year forecasts existed (`scripts/14_make_tables.py`, `CONTRASTS`): TimesFM 2.5 vs. Chronos-2 (both with weather); fine-tuned vs. zero-shot Chronos-2 (Settings A and B); fine-tuned Chronos-2 vs. LightGBM (Settings A and B); PG-Adapter vs. fine-tuned Chronos-2.

## 6. Experiments

- **E1 (RQ1):** all baselines on all 16 series, full test year.
- **E2 (RQ2):** PG-Adapter (with and without the TSFM prior) against LightGBM when each model gets only 7, 14, 30 or 90 days of power history (365 = E1). Rolling design: at the start of every test month a new model is trained on the N days that end two days before the month (the last days fully observed at the 09:00 UTC issue time) and forecasts that month. Early stopping uses the last 20 % of the N days. Weather, solar-geometry and calendar inputs are standardised with the 365 days before the month (weather archives exist for any site); recent-power inputs stay as capacity factors. *Changed on 2026-10-25:* the first version trained once on the last N days of the training split and was tested on the whole test year; the 6–18-month gap made the neural models extrapolate across seasons (solar nMAE 11–62 %), so it measured seasonal shift rather than little data. Old runs are kept in `results/forecasts/<tech>/superseded/`.
- **E3 (RQ2, transfer):** train the adapter on other provinces, then apply it to a held-out province with zero or 14 days of its own data.
- **E4:** ablations. Remove the TSFM prior, remove physics features, remove NWP; full fine-tuning against the adapter; swap backbones.
- **E5 (RQ3):** reliability diagrams, coverage, interval width and Winkler score, with and without adaptive conformal calibration, reported by season.
- **E6 (RQ4):** imbalance cost of each forecast with real imbalance prices.
- **E7:** compute cost (parameters, training time, inference time on an M2 Pro laptop). Fine-tuning times are in `results/models/chronos2_ft/<tech>/<run>/train_info.json`. *Specified on 2026-10-29:* most runs recorded no time, or ran next to other jobs, so every model in the main table is timed again with `scripts/17_timing.py`: its own pipeline with the chosen settings, run once on the five wind series, one job at a time on an otherwise idle laptop, from the processed data to the test-year forecasts (training, priors and model loading included; choosing settings not included; one seed). Outputs go to a temporary folder (`PGTSFM_RESULTS`), so no reported forecast changes. Times: `results/metrics/timing.csv`; table: `paper/tables/compute.tex`.
  - *Change (2026-10-29), decided before the new times were known:* LightGBM, Chronos-Bolt and TiRex were first timed in a smoke test while the laptop was in interactive use, and Chronos-Bolt's time was implausible (391 s for the 365 test days, against 445 s for all 913 days in PG-Adapter's prior step, same code). All three are timed again on the idle laptop after the main run (`17_timing.py --only lgb bolt tirex --redo`); the first rows are kept in `results/metrics/timing_superseded.csv`.
  - Parameters (2026-10-29): counted with `scripts/18_parameters.py`, total and trained on our data (`results/metrics/parameters.csv`); shown in the same table.
  - Result (2026-10-29): the re-timed Chronos-Bolt took 133 s (smoke test: 391 s); LightGBM and TiRex changed by under
    3 s. Fine-tuning Chronos-2 with Elia's forecast took 238.5 min in the timing run, against 73.9-194.6 min in the
    grid runs of the same configuration (other configurations: 39.8-57.0 min, stable). Cause not established;
    reported as measured, with the grid spread, in the paper.

## 7. Metrics and statistics

- nMAE and nRMSE in % of capacity (solar also reported for daylight hours only)
- Skill score against persistence and against Elia
- CRPS (from quantiles), pinball loss, PICP/PINAW at 80 %
- Diebold–Mariano tests on daily losses, with Holm correction across models
- Results per region, plus the average across regions

## 8. Target journals

Primary: **Applied Energy** (hybrid; APC $4,210 if open access).

Alternatives:
- IEEE Transactions on Sustainable Energy ($2,800)
- Renewable Energy ($4,150)
- Energy Conversion and Management ($4,500)

All four are SCIE and in Scopus (checked 23 Oct 2026). Publishing via the subscription route costs nothing.

- *Change (2026-10-29):* the target is now the **Journal of Modern Power Systems and Clean Energy** (MPCE; IEEE
  Xplore; SCIE, JCR Q1; no APC, costs paid by its sponsor). It was chosen from journals indexed in SCIE or ESCI and
  Scopus, JCR Q1, with an APC under US$1,800. MPCE asks for 10 pages in its IEEE two-column template.

## 9. Reproducibility and integrity

- **Code and outputs:** every result comes from a script in this folder with fixed seeds and logged configs. Raw data downloads are scripted.
- **Reported numbers:** all numbers in the paper come from saved result files, never typed in by hand.
- **AI use:** the paper must disclose AI assistance in line with the journal's policy. You (the author) must understand and be able to defend every method choice and result.

## 10. Phases

1. Environment, data download and dataset build
2. Simple baselines, the Elia benchmark and LightGBM
3. Zero-shot TSFMs
4. Trained deep-learning baselines
5. PG-Adapter, data-scarcity experiments and ablations
6. Calibration and economic value
7. Figures, tables and paper draft

## 11. Risks

- **TSFM package compatibility on Apple MPS.** Fall back to CPU, or drop that backbone.
- **Weather is a single grid point per region.** We start with each region's centroid and add a few points per province if time allows.
- **Elia "measured" solar is an upscaled estimate for small rooftop PV.** We state this; it is the same target that Elia's own forecast is judged against.
- **The adapter may not beat strong trained models when data is plentiful.** The data-scarcity and transfer results are the main claim either way.
