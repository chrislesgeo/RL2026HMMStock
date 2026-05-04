import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "log_return_1",
    "volatility_10",
    "momentum_10",
    "trend_slope"   # ADD THIS
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

def load_data(csv_path):
    
    df = pd.read_csv(csv_path)

    # ADD THIS 👇
    df["trend_slope"] = compute_trend_slope(df["close"].values)

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