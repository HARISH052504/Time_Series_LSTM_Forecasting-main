# =============================
# 1. Imports and setup
# =============================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX

os.makedirs("data", exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

# =============================
# 2. Data generation
# =============================
n_steps = 1500
time = np.arange(n_steps)

trend = time * 0.01
seasonality = np.sin(2 * np.pi * time / 50)
noise = np.random.normal(0, 0.5, n_steps)

f1 = trend + seasonality + noise
f2 = np.cos(2 * np.pi * time / 30) + np.random.normal(0, 0.3, n_steps)
f3 = np.random.normal(0, 1, n_steps)

target = 0.6 * f1 + 0.3 * f2 + 0.1 * f3

df = pd.DataFrame({
    "f1": f1,
    "f2": f2,
    "f3": f3,
    "target": target
})

df.to_csv("data/dataset.csv", index=False)

# =============================
# 3. LSTM model
# =============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =============================
# 4. Data preparation
# =============================
df = pd.read_csv("data/dataset.csv")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

SEQ_LEN = 20

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

loader = DataLoader(
    TensorDataset(X_torch, y_torch),
    batch_size=32,
    shuffle=True
)

# =============================
# 5. Train LSTM
# =============================
model = LSTMModel(input_size=3, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

print("LSTM training completed")

# =============================
# 6. SHAP explainability (SAFE)
# =============================
def shap_model(x):
    x = x.reshape(-1, SEQ_LEN, 3)
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x).numpy()

background = X[:50].reshape(50, -1)
test_samples = X[50:60].reshape(10, -1)

explainer = shap.KernelExplainer(shap_model, background)
shap_values = explainer.shap_values(test_samples)

print("SHAP analysis completed")

# =============================
# 7. SARIMA baseline
# =============================
sarima = SARIMAX(df["target"], order=(2, 1, 2))
sarima_result = sarima.fit(disp=False)
sarima_forecast = sarima_result.forecast(steps=100)

print("SARIMA baseline completed")

# =============================
# 8. Metrics
# =============================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("All steps executed successfully")

