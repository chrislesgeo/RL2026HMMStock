import matplotlib.pyplot as plt
import numpy as np


def plot_regime_timeline(states):
    plt.figure(figsize=(14, 3))
    plt.plot(states, drawstyle='steps-post')
    plt.title("Regime Timeline")
    plt.xlabel("Time")
    plt.ylabel("Regime")
    plt.show()


def plot_price_colored(df, states):
    price = df["close"].values

    colors = ["green", "red", "orange"]

    plt.figure(figsize=(14, 6))
    plt.plot(price, color="black", linewidth=1)

    for i in range(len(states)):
        plt.scatter(i, price[i], color=colors[states[i]], s=8)

    plt.title("Price Colored by Regime")
    plt.show()


def plot_regime_stats(states):
    unique, counts = np.unique(states, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts)
    plt.title("Regime Frequency")
    plt.xlabel("Regime")
    plt.ylabel("Count")
    plt.show()