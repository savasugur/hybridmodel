# Hybrid HEC-HMS + ANFIS Residual-Learning Framework
### Event-Based Rainfall–Runoff Modeling | Kurukavak Creek Basin, Türkiye

> **Reference:** Temelli, S.U. & Tombul, M. — *A Hybrid Error-Correction Framework for Rainfall–Runoff Simulation Integrating Process-Based Modeling and ANFIS* (submitted)

---

## Overview

This repository contains the MATLAB implementation and data for the hybrid residual-learning framework described in the reference paper. The framework couples a process-based hydrological model (HEC-HMS) with an Adaptive Neuro-Fuzzy Inference System (ANFIS) to correct systematic simulation errors while preserving physical interpretability.

**Key result:** Hybrid model NSE = 0.981 on independent test event vs. HEC-HMS baseline NSE = 0.669.

---

## Repository Structure

```
├── anfis_hybrid_model.m        # Main MATLAB script (standalone ANFIS + hybrid model)
├── event_1988_30Jun.csv        # Storm event: 30 June 1988  (61 time steps)
├── event_1990_02May.csv        # Storm event: 02 May 1990  (108 time steps)
├── event_1995_03Jun.csv        # Storm event: 03 June 1995  (72 time steps)
├── event_1998_23Mar.csv        # Storm event: 23 March 1998  (48 time steps)
├── event_2005_01Jun.csv        # Storm event: 01 June 2005  (174 time steps) ← test event
└── README.md
```

---

## Data Format

Each CSV file contains four columns at 5-minute resolution:

| Column | Variable | Unit | Description |
|--------|----------|------|-------------|
| `Time_min` | t | min | Elapsed time from event start |
| `Rainfall` | P | mm / 5 min | Areal rainfall depth |
| `Qobs` | Q_obs | m³/s | Observed discharge at basin outlet |
| `Qhms` | Q_HMS | m³/s | HEC-HMS simulated discharge (interpolated to 5-min from 20-min output) |

HEC-HMS outputs were originally at 20-minute intervals and linearly interpolated to 5-minute resolution to match observed data.

---

## Study Area

**Basin:** Kurukavak Creek Basin, Pazaryeri, Bilecik, northwestern Türkiye  
**Area:** ~4.706 km²  
**Monitoring period:** 1988–2005 (5 storm events with sufficient data quality)  
**Hydrology:** Small, poorly gauged catchment; rapid runoff response to rainfall

---

## Requirements

- MATLAB R2021a or later
- **Fuzzy Logic Toolbox** (required for `genfis2`, `anfis`, `evalfis`)

---

## Usage

1. Clone the repository or download all files into a single folder.
2. Open MATLAB and set the working directory to that folder.
3. Open `anfis_hybrid_model.m` and adjust the settings at the top of the script if needed:

```matlab
TEST_EVENT_INDEX = 5;   % 5 = 2005 event (default, matches paper)
LAG              = 2;   % Lag order (paper: LAG=2)
N_EPOCHS         = 50;  % Training epochs
CLUSTER_RAD      = 0.8; % Subtractive clustering radius
SAVE_FIGURES     = false; % Set true to save figures to /results
```

4. Run the script. Outputs include:
   - Full performance table (NSE, RMSE, MAE, KGE, PBIAS) for all 5 events × 3 models
   - Six figures (training curves, hydrographs, scatter plot)

---

## Model Architecture

### Standalone ANFIS
- **Inputs:** P(t), Q(t), P(t−1), Q(t−1) — 4 inputs, lag = 2
- **Output:** Q_obs(t+1)
- FIS generated via subtractive clustering (radius = 0.8) → single Sugeno-type rule
- Training: hybrid least-squares + gradient descent, 50 epochs, internal 80/20 split

### Hybrid Model (HEC-HMS + ANFIS Residual Correction)
- HEC-HMS provides the process-based simulation Q_HMS
- **Inputs:** P(t), Q_HMS(t), Q_obs(t), P(t−1), Q_HMS(t−1), Q_obs(t−1) — 6 inputs
- **Target:** residual r(t+1) = Q_obs(t+1) − Q_HMS(t+1)
- **Final prediction:** Q_hybrid = Q_HMS + r̂ (ANFIS-predicted residual)

### Training–Test Split
Four events (1988, 1990, 1995, 1998) are used for training; the 2005 event is reserved exclusively for independent testing. This strict event-based separation prevents temporal data leakage.

---

## Performance (Independent Test Event — 01 June 2005)

| Model | NSE | RMSE | MAE | KGE | PBIAS |
|-------|-----|------|-----|-----|-------|
| HEC-HMS | 0.669 | 0.1414 | 0.1246 | 0.605 | 37.68 |
| Standalone ANFIS | **0.993** | 0.0205 | 0.0097 | 0.947 | 2.57 |
| Hybrid | 0.981 | 0.0342 | 0.0312 | 0.861 | 13.87 |

*All metrics computed on normalised discharge values [0, 1] using the lag-trimmed evaluation window.*

---

## Citation

If you use this code or data, please cite:

```
Temelli, S.U. & Tombul, M. (submitted). A Hybrid Error-Correction Framework for
Rainfall–Runoff Simulation Integrating Process-Based Modeling and ANFIS.
```

---

## Data Availability

Hydrometeorological data (observed precipitation and discharge) were obtained from the Turkish State Meteorological Service (MGM) and the General Directorate of State Hydraulic Works (DSİ) and are not publicly available. The CSV files in this repository contain the processed event data as used in the study.

---

## License

This code is released for research reproducibility. Please contact the corresponding author for any other use.

**Corresponding author:** Savas Ugur Temelli — savasugur.temelli@nisantasi.edu.tr
