"# Reinforcement Learning for Stock Trading with Hidden Markov Models

This project implements and evaluates two Proximal Policy Optimization (PPO) agents for stock trading, enhanced with market regime detection using a Hidden Markov Model (HMM). The goal is to create a profitable trading strategy that adapts its behavior based on whether the market is in a Bull, Bear, or Sideways regime.

Two distinct approaches are explored:
1.  **Continuous Action Space**: The agent outputs a scalar value in `[-1, 1]` representing the desired portfolio exposure (e.g., `0.5` for 50% long, `-0.2` for 20% short). This allows for nuanced position sizing.
2.  **Discrete Action Space**: The agent chooses from three distinct actions: `Hold`, `Buy`, or `Sell`. This simplifies the decision-making process to timing market entry and exit.

## Project Structure

```
.
├── Data/
│   └── itc/              # Contains historical price and feature data for ITC stock
├── Results/
│   ├── continuous/       # PNG plots for the continuous agent's performance
│   └── discrete/         # PNG plots for the discrete agent's performance
├── PPO_HMM_continuous.py # Main script for the continuous action space agent
├── PPO_HMM_discrete.py   # Main script for the discrete action space agent
├── regime_detector.py    # Class for the Hidden Markov Model
├── regime_features.py    # Functions for feature engineering (RSI, trend slope, etc.)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Features

- **PPO Agent**: Utilizes the robust PPO algorithm from `stable-baselines3` for training the trading agents.
- **HMM for Regime Detection**: A 3-state Gaussian HMM is trained on market features (returns, volatility, momentum) to identify Bull, Bear, and Sideways regimes. The regime probabilities are fed as observations to the agent, providing crucial market context.
- **Advanced Feature Engineering**: The models use a combination of technical indicators as features, including:
    - Log Returns
    - Historical Volatility (10 and 20-day)
    - Momentum (5 and 10-day)
    - Moving Average Ratios
    - **RSI (Relative Strength Index)**
    - **Trend Slope**
- **Sophisticated Reward Shaping**:
    - **Continuous**: Sortino-inspired reward that penalizes downside volatility more heavily.
    - **Discrete**: Regime-aware rewards that provide bonuses for actions aligning with the current market state (e.g., buying in a bull market).
- **Comprehensive Evaluation & Plotting**: Each script generates a suite of performance plots and metrics, including:
    - Portfolio Growth vs. Buy & Hold
    - Sharpe, Sortino, and Calmar Ratios
    - Drawdown Analysis
    - Position Sizing / Trade Actions
    - Rolling Performance Metrics

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RL2026HMMStock
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run either the continuous or the discrete agent. The scripts will handle data loading, feature engineering, model training, evaluation, and plotting automatically.

-   **To run the continuous action space agent:**
    ```bash
    python PPO_HMM_continuous.py
    ```

-   **To run the discrete action space agent:**
    ```bash
    python PPO_HMM_discrete.py
    ```

After execution, all performance charts and metrics will be saved as PNG images in the corresponding `Results/` sub-directory (`Results/continuous/` or `Results/discrete/`).

## Results

The primary output is a collection of plots saved to the `Results/` directory, allowing for a deep analysis of the agent's performance, risk profile, and behavior. Key plots include a summary dashboard, portfolio growth charts, drawdown analysis, and action/position visualizations.
" 
