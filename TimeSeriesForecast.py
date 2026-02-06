import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import VAR
import random

# ===============================
# Reproducibility
# ===============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Data Generation
# ===============================
def generate_multivariate_time_series(
    n_steps: int = 1500,
    n_features: int = 10
) -> np.ndarray:
    """
    Generate a synthetic correlated multivariate time series.

    Args:
        n_steps: Number of time steps
        n_features: Number of variables/features

    Returns:
        Array of shape (n_steps, n_features)
    """
    cov = 0.5 * np.ones((n_features, n_features)) + 0.5 * np.eye(n_features)
    noise = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov,
        size=n_steps
    )

    signals = []
    for i in range(n_features):
        base = np.sin(np.linspace(0, 20, n_steps) + i)
        trend = 0.01 * np.arange(n_steps)
        signals.append(base + trend)

    signals = np.stack(signals, axis=1)
    return signals + noise


# ===============================
# Dataset
# ===============================
class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for multivariate time series.
    """

    def __init__(
        self,
        data: np.ndarray,
        input_len: int,
        output_len: int
    ):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.input_len]
        y = self.data[idx + self.input_len: idx + self.input_len + self.output_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ===============================
# TCN Components
# ===============================
class TemporalBlock(nn.Module):
    """
    Single temporal block for TCN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    """
    Temporal Convolutional Network for Quantile Forecasting.
    """

    def __init__(
        self,
        n_features: int,
        output_len: int,
        quantiles: List[float],
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = n_features if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_ch, channels[i], kernel_size, dilation, dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(
            channels[-1],
            output_len * n_features * len(quantiles)
        )
        self.quantiles = quantiles
        self.output_len = output_len
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y.view(
            -1,
            len(self.quantiles),
            self.output_len,
            self.n_features
        )


# ===============================
# Loss & Metrics
# ===============================
def pinball_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: List[float]
) -> torch.Tensor:
    """
    Quantile (Pinball) loss.
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - preds[:, i]
        losses.append(torch.max(q * error, (q - 1) * error).mean())
    return torch.stack(losses).mean()


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Tuple[float, float]:
    """
    Coverage probability and mean interval width.
    """
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    return coverage, width


# ===============================
# Training
# ===============================
def train_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    quantiles: List[float],
    epochs: int = 20
) -> None:
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = pinball_loss(preds, y, quantiles)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")


# ===============================
# Benchmark: VAR
# ===============================
def var_forecast(
    train: np.ndarray,
    test_len: int
) -> np.ndarray:
    """
    VAR forecast.
    """
    model = VAR(train)
    results = model.fit(maxlags=5, ic="aic")
    return results.forecast(train[-results.k_ar:], steps=test_len)


# ===============================
# Main Execution
# ===============================
def main() -> None:
    # Data
    data = generate_multivariate_time_series()
    train_data = data[:1200]
    test_data = data[1200:]

    # Dataset
    INPUT_LEN = 30
    OUTPUT_LEN = 10
    QUANTILES = [0.1, 0.5, 0.9]

    dataset = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = TCN(
        n_features=data.shape[1],
        output_len=OUTPUT_LEN,
        quantiles=QUANTILES,
        channels=[32, 32, 32]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train_model(model, loader, optimizer, QUANTILES)

    # Deep Learning Evaluation
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(
            test_data[:INPUT_LEN],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        preds = model(x_test).cpu().numpy()

    median_pred = preds[0, 1]
    lower = preds[0, 0]
    upper = preds[0, 2]
    y_true = test_data[INPUT_LEN:INPUT_LEN + OUTPUT_LEN]

    rmse = np.sqrt(mean_squared_error(y_true.flatten(), median_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), median_pred.flatten())
    cov, width = interval_metrics(y_true, lower, upper)

    # VAR Benchmark
    var_preds = var_forecast(train_data, OUTPUT_LEN)
    var_rmse = np.sqrt(mean_squared_error(y_true.flatten(), var_preds.flatten()))
    var_mae = mean_absolute_error(y_true.flatten(), var_preds.flatten())

    # Results
    results = pd.DataFrame({
        "Model": ["TCN (Quantile)", "VAR"],
        "RMSE": [rmse, var_rmse],
        "MAE": [mae, var_mae],
        "Coverage (80%)": [cov, np.nan],
        "Mean Interval Width": [width, np.nan]
    })

    print("\nComparative Results:")
    print(results)


if __name__ == "__main__":
    main()
