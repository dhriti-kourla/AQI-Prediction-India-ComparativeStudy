# AQI Prediction: LSTM & Machine Learning Comparative Study

Comparative analysis of **LSTM, Gradient Boosting, Random Forest, SVR, and Multiple Linear Regression** for Air Quality Index prediction across 20 Indian cities. This study evaluates deep learning vs classical ML approaches for time-series forecasting on real CPCB (Central Pollution Control Board) data.

## Key Results

| Model | R² | RMSE | MAE | Time |
|-------|-----|------|-----|------|
| **LSTM** | 0.82 | 18.5 | 12.3 | 45 min |
| **Gradient Boosting** | 0.81 | 19.1 | 12.8 | 12 min |
| **Random Forest** | 0.79 | 20.3 | 13.5 | 8 min |
| **SVR** | 0.75 | 22.8 | 15.2 | 25 min |
| **MLR** | 0.68 | 26.4 | 18.1 | 1 min |

**Finding**: LSTM improves over baseline by 20.6%, but Gradient Boosting provides the best accuracy-complexity trade-off for production use.

## Project Structure

```
├── LSTM.ipynb                      # Deep learning model
├── MLR.ipynb                       # Linear regression baseline
├── SVR.ipynb                       # Support vector regression
├── GradientBoosting.ipynb          # Ensemble method
├── RandomForest.ipynb              # Ensemble method
├── figures/                        # Performance visualizations
├── report/AQI_Prediction_Report.md # Detailed technical report
├── AQI_prediction_dataset.csv      # Raw data (20 cities)
└── requirements.txt
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a model
jupyter notebook LSTM.ipynb
# Then: Kernel → Restart & Run All
```

## Dataset

- **Source**: CPCB (Central Pollution Control Board) India
- **Coverage**: 20 major Indian cities
- **Features**: PM2.5, PM10, NO₂, SO₂, CO, O₃, NH₃ + meteorological data
- **Challenge**: ~40-60% missing values, sensor drift

## Models Overview

- **LSTM**: 2-layer recurrent network; captures temporal dependencies
- **Gradient Boosting**: Sequential ensemble; fastest training relative to accuracy
- **Random Forest**: Parallel ensemble; robust to outliers
- **SVR**: Non-linear kernel; effective for high-dimensional data
- **MLR**: Linear baseline; interpretable reference

## Results

See `figures/` for visualizations:
- `fig5_avg_r2_comparison.png` - Model accuracy comparison
- `fig6_avg_rmse_comparison.png` - Prediction error analysis
- `fig7_percity_r2_heatmap.png` - Geographic performance distribution

Full analysis in `report/AQI_Prediction_Report.md`

## License

MIT License - see LICENSE file
