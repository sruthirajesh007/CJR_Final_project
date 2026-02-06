# CJR_Final_project
Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification

Overview
This project implements an advanced multivariate time series forecasting system using deep learning with uncertainty quantification. A Temporal Convolutional Network (TCN) is employed with quantile regression to generate probabilistic forecasts, and its performance is compared against a traditional Vector AutoRegression (VAR) benchmark.

Objectives
- Forecast multivariate time series data using deep learning
- Quantify predictive uncertainty using quantile regression
- Evaluate both point forecast accuracy and interval calibration
- Compare deep learning results with a classical statistical model

Dataset
A synthetic multivariate time series dataset is programmatically generated:
- 1500 time steps
- 10 correlated features
- Sinusoidal signals with trend and correlated Gaussian noise

Model Architecture
Temporal Convolutional Network (TCN):
- Dilated causal convolutions for long-range dependency capture
- Residual connections for stable training
- Fully connected output layer producing multiple quantiles

Uncertainty Quantification
Quantile Regression:
- Quantiles used: 0.1, 0.5, 0.9
- Pinball loss function applied during training
- Prediction intervals derived from lower and upper quantiles

Training Configuration
- Input window length: 30
- Output forecasting horizon: 10
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 20

Evaluation Metrics
Point Forecast Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Uncertainty Metrics:
- Coverage Probability
- Mean Interval Width

Benchmark Model
Vector AutoRegression (VAR):
- Fitted with automatic lag selection using AIC
- Used for baseline point forecast comparison

Results
The TCN with quantile regression demonstrates competitive or superior point forecast accuracy compared to VAR, while additionally providing calibrated uncertainty estimates unavailable in traditional models.

Reproducibility
- Fixed random seeds for NumPy, PyTorch, and Python random
- Modular, well-documented, production-quality code

How to Run
1. Install dependencies:
   numpy, pandas, torch, scikit-learn, statsmodels, python-docx
2. Run the Python script:
   python main.py
3. View printed evaluation results in the console

Future Improvements
- Extend to Transformer-based architectures
- Incorporate Monte Carlo Dropout or Bayesian Neural Networks
- Evaluate on real-world datasets such as M4 or energy load data
- Improve calibration using post-hoc techniques

Author
Sruthi Rajesh
