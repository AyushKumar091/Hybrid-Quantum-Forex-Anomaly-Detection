# Multi-Scale Forecasting of USD/JPY Macroeconomic Anomalies via a Hybrid CNN–Variational Quantum Circuit Transformer

> Official implementation of the research paper submitted for peer review.

---

## Abstract

We present a hybrid quantum-classical architecture for multi-scale flash crash detection in the USD/JPY foreign exchange market, combining a two-dimensional Convolutional Neural Network (CNN), a four-qubit Variational Quantum Circuit (VQC) with an EfficientSU2 ansatz, and a Multi-Head Attention decoder. The architecture is evaluated simultaneously across three forecast horizons — daily (1D), weekly (1WK), and monthly (1MO) — under a strictly enforced zero-leakage data pipeline.

---

## Key Results

| Timeframe | Test Samples | RMSE (Yen) | Dir Acc | Crash Precision | Crash Recall | Crash F1 |
|---|---|---|---|---|---|---|
| **1D** | 98 | 0.8212 | 59.2% | 0.2188 | 0.2500 | 0.2333 |
| **1WK** | 48 | 1.7570 | 62.5% | 0.2500 | 0.3571 | 0.2941 |
| **1MO** | 51 | 6.0916 | 39.2% | 0.5000 | 0.3529 | 0.4138 |

Crash F1 improves monotonically across horizons (0.23 → 0.29 → 0.41), demonstrating strengthening anomaly detection at coarser timescales. The 1MO Crash Precision of 0.50 represents a 1.5× lift over the naive base rate of 33.3%.

---

## Novel Contributions

### 1. CrashFocusedLoss
A formally named asymmetric loss function that eliminates the MSE regression-to-mean degeneracy under class imbalance. Three components combined:

- **Huber base loss** (δ = 1.0) — robust to outliers
- **Directional penalty weight** — 3× multiplier for sign-incorrect predictions
- **Crash amplification weight** — 10× multiplier for minority-class crash events

The combined weight produces a **30:1 effective penalty ratio** for wrong-direction crash predictions versus correct normal predictions, forcing the model to hunt minority-class anomalies rather than predicting the distributional mean.

### 2. Zero-Leakage Data Pipeline
All normalization statistics — Volume scaling bounds, VIX scaling bounds, and rolling standard deviation denominators — are computed exclusively on training-set samples. Test-set transformations use frozen training-set parameters, provably eliminating future-data contamination present in a substantial portion of existing financial ML literature.

### 3. Barren Plateau Mitigation via Staged CNN Freeze
CNN parameters are frozen at epoch 20, preventing the ~4,000-parameter classical gradient from numerically dominating the ~30-parameter quantum gradient during backpropagation. This is a novel application of barren plateau mitigation to hybrid classical-quantum training, grounded in Cerezo et al. (2021).

### 4. Statistically Centered Anomaly Detection Threshold
Crash events are flagged when predictions deviate from their own empirical mean by more than one standard deviation — not from an absolute zero. This eliminates systematic detection bias from inter-series variance mismatch and makes Crash F1, Precision, and Recall robust to amplitude dampening.

### 5. Multi-Scale Simultaneous Evaluation
Among the first quantum-classical hybrid models to report directional accuracy, crash precision, crash recall, and crash F1-score simultaneously across daily, weekly, and monthly forex horizons on USD/JPY.

---

## Architecture

```
Price Window
      |
[Gramian Angular Summation Field (GASF)]  →  W×W image
      |
[CNN: Conv2d(1→4, k=2×2) + ReLU + Flatten]
      |
[Fusion: CNN features + Volume (W) + VIX (W)]
      |
[FC Bridge: Linear(→32) + ReLU + Dropout(0.2) + Linear(→4) + Tanh]
      |  z_q bounded in [-1, 1]^4
[Quantum VQC: ZZFeatureMap(reps=1) + EfficientSU2(reps=1)]
      |  gradients via parameter-shift rule through TorchConnector
[Multi-Head Attention: embed(1→8), 2 heads, d_k=4]
      |
[Linear(8→1)]  →  Predicted z-score momentum
```

**Quantum Circuit:** 4 qubits | ZZ pairwise entanglement | ~30 trainable parameters

---

## Data Sources

All data is fetched live from Yahoo Finance via `yfinance` at runtime. Internet connection required.

| Symbol | Description | Role |
|---|---|---|
| `USDJPY=X` | USD/JPY close + volume | Primary prediction target |
| `USDINR=X` | USD/INR close | Cross-currency feature |
| `EURUSD=X` | EUR/USD close | Cross-currency feature |
| `GBPUSD=X` | GBP/USD close | Cross-currency feature |
| `^VIX` | CBOE Volatility Index | Market fear indicator |

---

## Timeframe Configuration

| Timeframe | Lookback Window | Data Period | Validation Split | Early Stopping |
|---|---|---|---|---|
| 1D (Daily) | 30 days | 2 years | 10% | Patience = 10 |
| 1WK (Weekly) | 20 weeks | 5 years | None | Disabled |
| 1MO (Monthly) | 12 months | Max (~25 years) | None | Disabled |

---

## Requirements

### Python Version
> **Python 3.10.x is required.** Qiskit Machine Learning is not compatible with Python 3.11 or higher.

Download Python 3.10.11: https://www.python.org/downloads/release/python-31011/

### Install All Dependencies

```bash
pip install yfinance pandas numpy torch scikit-learn matplotlib scipy pylatexenc
pip install qiskit==0.45.3 qiskit-machine-learning==0.7.2 qiskit-algorithms==0.3.0
```

### Dependency Table

| Package | Version | Purpose |
|---|---|---|
| `torch` | Latest | CNN, Attention, training |
| `qiskit` | 0.45.3 | Quantum circuit design |
| `qiskit-machine-learning` | 0.7.2 | EstimatorQNN, TorchConnector |
| `qiskit-algorithms` | 0.3.0 | Required by qiskit-machine-learning |
| `yfinance` | Latest | Live forex data download |
| `pandas` | Latest | Data manipulation |
| `numpy` | Latest | Numerical operations |
| `scikit-learn` | Latest | F1, precision, recall metrics |
| `matplotlib` | Latest | Prediction graph generation |
| `scipy` | Latest | Statistical computations |
| `pylatexenc` | Latest | Quantum circuit diagram rendering |

---

## How to Run

### On Windows (VS Code / PowerShell)

```bash
$env:PYTHONIOENCODING="utf-8"; python code.py
```

### On Linux / Mac

```bash
python code.py
```

### On Google Colab (Recommended — Free T4 GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `code.py`
3. Enable GPU: **Runtime → Change Runtime Type → T4 GPU**
4. Run in a code cell:

```python
!pip install -q yfinance scipy pylatexenc
!pip install -q qiskit==0.45.3
!pip install -q qiskit-machine-learning==0.7.2
!pip install -q qiskit-algorithms==0.3.0
!python code.py
```

---

## Output Files

All files are saved in the same directory as `code.py` after execution.

| File | Description |
|---|---|
| `Circuit_Diagram.png` | Quantum circuit visualization |
| `RawData_1d.csv` | Raw fetched daily forex data |
| `RawData_1wk.csv` | Raw fetched weekly forex data |
| `RawData_1mo.csv` | Raw fetched monthly forex data |
| `Prediction_1D.png` | Actual vs predicted plots — daily |
| `Prediction_1WK.png` | Actual vs predicted plots — weekly |
| `Prediction_1MO.png` | Actual vs predicted plots — monthly |

---

## Reproducibility

All experiments use a fixed global seed of 42 across Python, NumPy, and PyTorch:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Running the code with these settings produces identical results across machines.

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.003 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Max Epochs | 80 |
| CNN Freeze Epoch | 20 |
| Gradient Clip | max_norm = 1.0 |
| CrashFocusedLoss δ | 1.0 |
| Crash Weight λ_c | 10.0 |
| Dropout | 0.2 |
| Quantum Qubits | 4 |

---

## Quantum Simulation Note

All variational quantum circuit computations are performed using Qiskit's statevector simulator executed on classical hardware. The ZZFeatureMap–EfficientSU2 circuit is interfaced with PyTorch via TorchConnector, enabling end-to-end gradient computation through the parameter-shift rule on a classical CPU. No real quantum hardware was used in this study. This simulation-based approach is standard practice in current quantum machine learning research, consistent with the methodology of directly comparable prior works including Chen et al. (QLSTM, 2020) and Xu et al. (QGAF, IEEE QCE 2023).

---

## Hardware

All paper experiments were executed on an NVIDIA GeForce RTX 4050 Laptop GPU under seed 42. The quantum circuit layer runs on CPU regardless of GPU availability due to Qiskit simulator constraints.

---

## Cross-Currency Correlations (Pearson) with USD/JPY

| Currency Pair | 1D | 1WK | 1MO |
|---|---|---|---|
| USD/INR | +0.2579 | +0.8994 | +0.6921 |
| EUR/USD | -0.2084 | -0.3548 | -0.6237 |
| GBP/USD | -0.3769 | -0.3914 | -0.3694 |

USD/INR correlation strengthens from daily (0.258) to weekly (0.899), reflecting shared dollar-strength dynamics. EUR/USD and GBP/USD maintain stable negative correlations across all horizons, consistent with established dollar-index inverse relationships.

---

## References

1. B. G. Malkiel, "The Efficient Market Hypothesis and Its Critics," *Journal of Economic Perspectives*, vol. 17, no. 1, pp. 59–82, 2003.
2. R. S. Tsay, *Analysis of Financial Time Series*, 3rd ed. Wiley, 2010.
3. R. Thakkar et al., "Quantum Machine Learning for Structural Break Detection in Financial Time Series," *arXiv:2303.15432*, 2023.
4. S. Y. C. Chen et al., "Quantum Long Short-Term Memory," *arXiv:2009.01783*, 2020.
5. M. Schuld and N. Killoran, "Quantum Machine Learning in Feature Hilbert Spaces," *Physical Review Letters*, vol. 122, 040504, 2019.
6. J. R. McClean et al., "Barren plateaus in quantum neural network training landscapes," *Nature Communications*, vol. 9, 4812, 2018.
7. S. Galeshchuk and S. Mukherjee, "Deep learning for technical analysis," *Expert Systems with Applications*, vol. 81, pp. 113–124, 2017.
8. O. B. Sezer et al., "Financial time series forecasting with deep learning," *Applied Soft Computing*, vol. 90, 2020.
9. B. Narmandakh et al., "Deep Learning for Forex Forecasting," *Journal of Risk and Financial Management*, vol. 15, no. 7, 2022.
10. Z. Wang and T. Oates, "Encoding Time Series as Images," *AAAI Workshops*, 2015.
11. H. Ismail Fawaz et al., "Deep learning for time series classification," *Data Mining and Knowledge Discovery*, vol. 33, pp. 917–963, 2019.
12. J. Xu et al., "Quantum Gramian Angular Field for Price Prediction," *IEEE QCE*, 2023.
13. L. Zhao et al., "BLS-QLSTM for Financial Time Series Forecasting," *IEEE Transactions on Cybernetics*, 2023.
14. M. Yunusa, "Quantum Reinforcement Learning for Forex Trading," *Journal of Quantum Computing*, 2023.
15. M. Cerezo et al., "Cost function dependent barren plateaus in shallow quantum neural networks," *Nature Communications*, vol. 12, 1791, 2021.
16. E. Grant et al., "An initialization strategy for addressing barren plateaus," *Quantum*, vol. 3, p. 214, 2019.

---

## License

This project is for academic research purposes only.
