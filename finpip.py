from regime_features import load_data, extract_features, scale_features, get_regime_probabilities
from regime_detector import RegimeDetector
from regime_plots import plot_regime_timeline, plot_price_colored, plot_regime_stats

# Load
df = load_data("itc_regime_features.csv")

# Features
features = extract_features(df)
features_scaled, scaler = scale_features(features)

# Train HMM
detector = RegimeDetector(n_states=3)
detector.fit(features_scaled)

# Outputs
states = detector.predict_states(features_scaled)
probs = get_regime_probabilities(detector, features_scaled)

import numpy as np

# ===============================
# REGIME STATISTICS
# ===============================

regime_stats = {}

for i in range(3):
    mask = states == i

    avg_return = df.loc[mask, "log_return_1"].mean()
    avg_vol = df.loc[mask, "volatility_10"].mean()
    avg_trend = df.loc[mask, "trend_slope"].mean()

    regime_stats[i] = {
        "return": avg_return,
        "vol": avg_vol,
        "trend": avg_trend
    }

print("\nRegime Stats:")
for k, v in regime_stats.items():
    print(f"Regime {k}: Return={v['return']:.5f}, Vol={v['vol']:.5f}, Trend={v['trend']:.5f}")


# ===============================
# CORRECT FINANCE-BASED MAPPING
# ===============================

remaining = set(regime_stats.keys())
assigned = {}

# 1. BULL → highest return
bull_regime = max(remaining, key=lambda i: regime_stats[i]["return"])
assigned["bull"] = bull_regime
remaining.remove(bull_regime)

# 2. BEAR → lowest return (most negative)
bear_regime = min(remaining, key=lambda i: regime_stats[i]["return"])
assigned["bear"] = bear_regime
remaining.remove(bear_regime)

# 3. SIDEWAYS → remaining
sideways_regime = remaining.pop()
assigned["sideways"] = sideways_regime

# reverse mapping
regime_map = {v: k for k, v in assigned.items()}

print("\nFinal Mapping:")
print(regime_map)

# ===============================
# APPLY LABELS
# ===============================

df["regime_label"] = [regime_map[s] for s in states]


# ===============================
# REORDER PROBABILITIES
# ===============================

bull_idx = assigned["bull"]
bear_idx = assigned["bear"]
side_idx = assigned["sideways"]

probs_mapped = np.column_stack([
    probs[:, bull_idx],
    probs[:, bear_idx],
    probs[:, side_idx]
])


# ===============================
# SAVE FINAL OUTPUT
# ===============================

df["P_bull"] = probs_mapped[:, 0]
df["P_bear"] = probs_mapped[:, 1]
df["P_sideways"] = probs_mapped[:, 2]

df.to_csv("regime_output.csv", index=False)

print("\nSaved: regime_output.csv")


# Visuals
plot_regime_timeline(states)
plot_price_colored(df, states)
plot_regime_stats(states)