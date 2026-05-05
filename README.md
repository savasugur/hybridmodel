# HybridANFIS-Rainfall-Runoff

MATLAB implementation of an ANFIS-based hybrid rainfall-runoff model that combines HEC-HMS physics-based simulations with a data-driven residual-correction layer.

---

## Background

Lumped hydrological models such as HEC-HMS capture basin-scale rainfall-runoff dynamics efficiently, but their simplified process representations introduce systematic errors. This repository provides a hybrid framework that treats those errors as a learnable signal: an Adaptive Neuro-Fuzzy Inference System (ANFIS) is trained on the residuals between HEC-HMS output and observed discharge, and its predictions are added back onto the physical baseline at inference time.

Training was performed on the 1988 storm event recorded in the study basin; the trained model was evaluated on the independent 2005 event.

---

## Repository Contents

```
├── hybrid_model.m      Main script — data loading, ANFIS training, evaluation, plotting
├── train_1988.xlsx     Training data (1988 storm event, normalized)
├── test_2005.xlsx      Test data (2005 storm event, normalized)
└── README.md
```

---

## Data Format

Both Excel files share the following columns:

| Column   | Description                                   |
|----------|-----------------------------------------------|
| `minute` | Time step index (5-minute intervals)          |
| `preNOM` | Normalized precipitation                      |
| `Qnom`   | Normalized observed discharge                 |
| `hed`    | Normalized HEC-HMS simulated discharge        |

> **Note:** The hydrometeorological data were obtained from the Turkish State Meteorological Service (MGM) and the General Directorate of State Hydraulic Works (DSİ) and are not publicly available. The Excel files provided here contain only the normalized values used directly by the model.

---

## Method

| Step | Description |
|------|-------------|
| 1 | Load normalized precipitation, observed discharge, and HEC-HMS discharge |
| 2 | Compute residuals: `r = Q_observed − Q_HEC-HMS` |
| 3 | Construct input features: `[P_norm, Q_HEC-HMS, Q_lag1]` |
| 4 | Initialize a Sugeno FIS via subtractive clustering (`genfis2`, radius = 0.8) |
| 5 | Train ANFIS on residuals for 50 epochs using the hybrid learning algorithm |
| 6 | Predict: `Q_hybrid = Q_HEC-HMS + ANFIS(features)` |
| 7 | Evaluate with NSE and RMSE on the 2005 test event |

---

## Requirements

- MATLAB R2019b or later
- Fuzzy Logic Toolbox (for `genfis2`, `anfis`, `evalfis`, `anfisOptions`)

---

## Usage

1. Place all files in the same directory.
2. Open MATLAB and set the working directory to that folder.
3. Run:

```matlab
hybrid_model
```

The script prints NSE and RMSE to the Command Window and generates a figure comparing observed, HEC-HMS, and hybrid discharge time series.

---

## Performance Metrics

- **NSE** (Nash-Sutcliffe Efficiency): values closer to 1 indicate better model performance; values above 0.75 are generally considered satisfactory in hydrology.
- **RMSE** (Root Mean Square Error): computed on normalized discharge.

---

## Computer Code Availability

- **Name of code:** HybridANFIS-Rainfall-Runoff
- **Developer:** Savas Ugur Temelli, Istanbul Nisantasi University, İstanbul, Türkiye
- **Contact:** savasugur.temelli@nisantasi.edu.tr
- **Year first available:** 2026
- **Hardware required:** Standard PC
- **Software required:** MATLAB R2019b or later; Fuzzy Logic Toolbox (MathWorks Inc., Natick, MA, USA)
- **Program language:** MATLAB
- **Program size:** < 5 KB
- **Source code:** https://github.com/savasugur/hybridmodel

> The standalone ANFIS model reported in the paper was developed using the MATLAB Neuro-Fuzzy Designer graphical interface. Because that tool stores trained models in a proprietary binary format, the resulting files are not suitable for public distribution. The hybrid model script provided here reproduces the residual-correction workflow in fully editable source code and can serve as the basis for replication.

---

## License

This code is released under the [MIT License](LICENSE).

---

## Reference

If you use this code, please cite the associated paper:

> Savas Ugur Temelli and Mustafa Tombul. (2026). A Hybrid Error-Correction Framework for Rainfall–Runoff Simulation Integrating Process-Based Modeling and ANFIS. Computers & Geosciences
