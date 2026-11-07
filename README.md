# Hybrid HEC-HMS–ANFIS residual learning — Kurukavak Creek Basin

Code and data for:

> Temelli, S. U. & Tombul, M. *A Hybrid Residual-Learning Approach Integrating HEC-HMS and ANFIS for
> Event-Based Rainfall–Runoff Modelling in a Data-Scarce Small Catchment.* Acta Geophysica
> (manuscript AGPH-D-26-00469, revision 1, under review).

An Adaptive Neuro-Fuzzy Inference System (ANFIS) learns the residuals of a calibrated HEC-HMS model
for five 5-min storm events in the 4.7 km² Kurukavak Creek Basin (Bilecik, Türkiye). The workflow
evaluates the hybrid model in two operating modes:

- **updating mode (one-step-ahead):** observed discharge at *t* and *t−1* is an input;
- **simulation mode:** no observed discharge after initialisation (recursive, or with no discharge feedback).

The models are benchmarked against HEC-HMS, persistence, a linear ARX model and standalone ANFIS. They are
tested on a fixed hold-out event (01 June 2005) and by leave-one-event-out cross-validation, with
leakage-free scaling and metrics in physical units.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs every step and regenerates all numbers, tables and figures of the paper. The full
run takes a few seconds. It works on Windows, macOS and Linux. Console output of each step is
written to `results/run_log.txt`.

Tested with Python 3.11–3.13, NumPy 2.x, pandas 2.x–3.x and Matplotlib 3.x. NumPy ≥ 1.26 is
required, because the code uses `np.trapezoid`.

## Repository contents

| Path | Content |
|---|---|
| `data/` | The five storm events used in the paper (see *Data* below) |
| `data_original/` | The same files with rainfall as originally supplied (see *Data*) |
| `pipeline.py` | Data loading, scaling, metrics, subtractive clustering (Chiu 1994) and first-order Sugeno ANFIS with hybrid learning (Jang 1993) |
| `experiments.py` | All models in both operating modes, three scaling schemes, fixed split and leave-one-event-out → `results/experiments.json` |
| `extra.py` | Optimiser sensitivity of the original protocol, LOEO summary, LOEO without the 1998 event |
| `tables.py` | LaTeX tables → `results/tab_*.tex` |
| `figures.py` | Figures 3–6 → `figures/` |
| `reproduce.py`, `reproduce2.py` | Reproduction of the original-submission protocol |
| `run_all.py` | Runs all of the above in order |
| `matlab/anfis_hybrid_model.m` | MATLAB script of the first submission (kept for transparency) |
| `hec-hms/` | HEC-HMS 4.2 project of the basin |
| `results/`, `figures/` | Outputs as reported in the revised paper |

### Where each result of the paper comes from

| Paper | Produced by | Output |
|---|---|---|
| Table 3 (2005 test event) | `experiments.py` → `tables.py` | `results/tab_test.tex` |
| Table 4 (LOEO NSE) | `experiments.py` → `tables.py` | `results/tab_loeo.tex` |
| Table 5 (scaling sensitivity) | `experiments.py` → `tables.py` | `results/tab_norm.tex` |
| Table A1 (complete LOEO metrics) | `experiments.py` → `tables.py` | `results/tab_full.tex` |
| Fig. 3 (events and HEC-HMS) | `figures.py` | `figures/fig3_events_hms.*` |
| Fig. 4 (training curves) | `figures.py` | `figures/fig4_training.*` |
| Fig. 5 (2005 test hydrographs) | `figures.py` | `figures/fig5_test2005.*` |
| Fig. 6 (LOEO NSE) | `figures.py` | `figures/fig6_loeo.*` |
| Sect. 2.9 (MATLAB comparison, optimiser sensitivity), Sect. 3.3 (LOEO without 1998) | `reproduce*.py`, `extra.py` | `results/run_log.txt` |

## Data

Each CSV file contains one storm event:

| Column | Meaning | Unit |
|---|---|---|
| `Time_min` | Time since the start of the event | min |
| `Rainfall` | Basin rainfall per row | mm per 5 min |
| `Qobs` | Observed discharge at the outlet gauge H1 | m³/s |
| `Qhms` | HEC-HMS simulated discharge (20-min output, linearly interpolated to the observation times) | m³/s |

| File | Event | Rows | Role (fixed split) |
|---|---|---|---|
| `event_1988_30Jun.csv` | 30 June 1988 | 61 | HEC-HMS calibration; training |
| `event_1990_02May.csv` | 02 May 1990 | 108 | Training |
| `event_1995_03Jun.csv` | 03 June 1995 | 72 | Training |
| `event_1998_23Mar.csv` | 23 March 1998 | 48 | Training |
| `event_2005_01Jun.csv` | 01 June 2005 | 174 | Test |

Notes:

- Rows are at 5-min intervals, except for 14 intervals of 45 min in the 2005 record (periods without
  recorded measurements). All models treat consecutive rows as consecutive time steps.
- The rainfall of the 1988 and 2005 events was supplied in units 1000 times larger than that of the
  other events. `data/` contains the converted values (mm per 5 min); `data_original/` contains the files
  as supplied. Model results do not depend on this conversion, because rainfall is min–max scaled
  within each event. Only Fig. 3 and the pooled-training scaling sensitivity (Table 5) change.
- Data source: General Directorate of State Hydraulic Works (DSİ), Eskişehir Research Institute.
  Please cite the paper and acknowledge the data source when using the data.

## Implementation notes

- With an influence radius of 0.8, subtractive clustering yields a single rule for the standalone ANFIS.
  A single-rule first-order Sugeno system is linear in its inputs, so the standalone ANFIS is equivalent
  to an ARX model. `reproduce.py` shows this: ordinary least squares reproduces the original MATLAB
  ANFIS results to four decimal places.
- The Python ANFIS follows the MATLAB Fuzzy Logic Toolbox defaults: `genfis2` initialisation, hybrid
  learning, 50 epochs, initial step size 0.01 with increase/decrease rates 1.1/0.9, and the parameter set
  with the lowest internal-validation error is retained. Hybrid-model results agree with the original
  MATLAB results within 0.0004 NSE for the training events. The original hybrid test value (NSE = 0.9806)
  could not be reproduced; see Sect. 2.9 of the paper.
- Primary (leakage-free) scaling: discharge in each event is divided by the peak of the HEC-HMS
  simulation of that event; rainfall is min–max scaled within each event. The pooled-training scaling and
  the per-event observed-range scaling of the original submission are evaluated as sensitivity tests.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

The code is released under the MIT License (see `LICENSE`).

## Contact

Savas Ugur Temelli — Istanbul Nişantaşı University — savasugur.temelli@nisantasi.edu.tr
