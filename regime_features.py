import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "log_return_1",
    "volatility_10",
    "momentum_10",
    "trend_slope",
    "rsi_14"
]
def compute_trend_slope(price, window=10):
    slopes = []
    for i in range(len(price)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = price[i-window:i]
            x = np.arange(window)

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            num = np.sum((x - x_mean)*(y - y_mean))
            den = np.sum((x - x_mean)**2)

            slope = num/den if den != 0 else 0
            slopes.append(slope)

    return np.array(slopes)

def compute_rsi(price, window=14):
    delta = np.diff(price)
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta

    avg_gain = np.zeros_like(price)
    avg_loss = np.zeros_like(price)

    avg_gain[window] = np.mean(gain[:window])
    avg_loss[window] = np.mean(loss[:window])

    for i in range(window + 1, len(price)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i-1]) / window

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_data(csv_path):
    
    df = pd.read_csv(csv_path)

    # ADD THIS 👇
    df["trend_slope"] = compute_trend_slope(df["close"].values)
    df["rsi"] = compute_rsi(df["close"].values)

    df = df.dropna().reset_index(drop=True)
    return df


def extract_features(df):
    return df[FEATURE_COLUMNS].values


def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def get_regime_probabilities(detector, features_scaled):
    probs = detector.predict_probabilities(features_scaled)
    return probs