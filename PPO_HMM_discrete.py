"""
PPO + HMM Regime Integration — DISCRETE Action Space
=====================================================
Actions : 0 = Hold  |  1 = Buy (go long)  |  2 = Sell / Close Long
Regime  : 3-state Gaussian HMM → Bull / Bear / Sideways probabilities
          injected into the observation to give the agent market context.

Key Improvements over baseline:
  - Regime-aware reward shaping (bearish → penalise longs, bullish → bonus)
  - Sharpe-inspired volatility-normalised reward signal
  - EvalCallback + early stopping to prevent overfitting
  - Clipped-PPO with tuned n_steps, batch_size, entropy & clip range
  - Rolling normalization & robust outlier clipping on features
  - Detailed metrics: Sharpe, Sortino, Max Drawdown, Calmar
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sklearn.preprocessing import StandardScaler

from regime_detector import RegimeDetector
from regime_features import extract_features, compute_trend_slope, compute_rsi


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
class Config:
    DATA_PATH        = "Data/itc/itc_rl_features.csv"
    TRAIN_SPLIT      = 0.75

    # Environment
    INITIAL_CAPITAL  = 1_000_000.0
    TRANSACTION_COST = 0.0010        # 10 bps – realistic for Indian equities
    HOLD_PENALTY     = -0.0001       # Mild cost for inaction
    VOL_WINDOW       = 30            # Rolling window for reward normalisation

    # Regime HMM
    N_REGIMES        = 3
    HMM_ITER         = 300

    # PPO Hyperparameters
    TOTAL_TIMESTEPS  = 300_000
    N_STEPS          = 2048          # Steps per rollout per env
    BATCH_SIZE       = 256
    N_EPOCHS         = 10
    GAMMA            = 0.995
    GAE_LAMBDA       = 0.92
    CLIP_RANGE       = 0.15          # Tighter clip → more conservative updates
    LEARNING_RATE    = 2e-4
    ENT_COEF         = 0.015         # Exploration bonus
    VF_COEF          = 0.5
    MAX_GRAD_NORM    = 0.5

    # Reward shaping
    REWARD_SCALE     = 100.0         # Amplifier so PPO sees meaningful gradients
    REGIME_BONUS     = 0.3           # Extra reward when action aligns with regime

    SEED             = 42
    SAVE_PATH        = "ppo_hmm_discrete_model"
    PLOT_PATH        = "Results/discrete/"

CFG = Config()

# Create results directory if it doesn't exist
if not os.path.exists(CFG.PLOT_PATH):
    os.makedirs(CFG.PLOT_PATH)


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
BASE_FEATURES = [
    "log_return_1", "volatility_10", "volatility_20",
    "close_ma10_ratio", "close_ma20_ratio", "close_ma5_ratio",
    "momentum_5", "momentum_10", "trend_slope", "rsi_14",
]

def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df["date"]        = pd.to_datetime(df["date"])
    df["trend_slope"] = compute_trend_slope(df["close"].values)
    df["rsi_14"]      = compute_rsi(df["close"].values)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill().fillna(0)

    # Forward-shifted return (next-bar realised PnL)
    df["return_1"] = df["close"].pct_change().shift(-1).fillna(0)

    # Clip extreme returns
    df["return_1"] = df["return_1"].clip(-0.1, 0.1)
    return df


def rolling_normalize(df: pd.DataFrame, cols: list, window: int = 50) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        mu    = out[col].rolling(window, min_periods=1).mean()
        sigma = out[col].rolling(window, min_periods=1).std()
        out[col] = (out[col] - mu) / (sigma + 1e-6)
    return out.fillna(0)


def generate_regime_features(df: pd.DataFrame, train_split: float = 0.75):
    """Fit GaussianHMM on training data and produce regime probabilities for all rows."""
    print("[HMM] Fitting regime detector …")
    split_idx = int(len(df) * train_split)

    raw_feats   = extract_features(df)                  # shape (N, 4)
    train_raw   = raw_feats[:split_idx]
    test_raw    = raw_feats[split_idx:]

    scaler      = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    detector = RegimeDetector(n_states=CFG.N_REGIMES, random_state=CFG.SEED)
    detector.model.n_iter = CFG.HMM_ITER
    detector.fit(train_scaled)

    train_probs = detector.predict_probabilities(train_scaled)
    test_probs  = detector.predict_probabilities(test_scaled)
    all_probs   = np.vstack([train_probs, test_probs])

    # ── Map HMM states to meaningful labels ──────────────────────────
    states = detector.predict_states(
        np.vstack([train_scaled, test_scaled])
    )
    stats = {}
    for s in range(CFG.N_REGIMES):
        mask = states == s
        stats[s] = df.loc[mask, "log_return_1"].mean() if "log_return_1" in df else 0.0

    sorted_states = sorted(stats, key=lambda s: stats[s])
    bear_idx, side_idx, bull_idx = sorted_states[0], sorted_states[1], sorted_states[2]

    regime_cols = ["p_bull", "p_sideways", "p_bear"]
    prob_df = pd.DataFrame({
        "p_bull"    : all_probs[:, bull_idx],
        "p_sideways": all_probs[:, side_idx],
        "p_bear"    : all_probs[:, bear_idx],
    }, index=df.index)

    print(f"[HMM] Regime mapping → bull={bull_idx} sideways={side_idx} bear={bear_idx}")
    return prob_df, regime_cols


# ══════════════════════════════════════════════════════════════════
#  DISCRETE TRADING ENVIRONMENT
# ══════════════════════════════════════════════════════════════════
class DiscreteTradingEnv(gym.Env):
    """
    Discrete action trading environment with regime awareness.

    Actions
    -------
    0 → Hold   : maintain current position, small penalty if flat
    1 → Buy    : open / stay long (1 unit of capital)
    2 → Sell   : close long position and go flat

    Observation
    -----------
    [normalised market features] + [p_bull, p_sideways, p_bear]
                                 + [in_position_flag, unrealised_pnl_pct]
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 transaction_cost: float = CFG.TRANSACTION_COST):
        super().__init__()
        self.df               = df.reset_index(drop=True)
        self.feature_cols     = feature_cols
        self.transaction_cost = transaction_cost
        self.max_steps        = len(self.df) - 1

        self.action_space      = spaces.Discrete(3)
        obs_dim                = len(feature_cols) + 2          # +position +unrealised
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._ret_buf = deque(maxlen=CFG.VOL_WINDOW)
        self.reset()

    # ── Internal helpers ─────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        feats      = self.df.loc[self.step_idx, self.feature_cols].values.astype(np.float32)
        pos_flag   = np.float32(self.in_position)
        price      = self.df.loc[self.step_idx, "close"]
        unrealised = np.float32(
            (price - self.entry_price) / (self.entry_price + 1e-9)
            if self.in_position else 0.0
        )
        return np.nan_to_num(np.concatenate([feats, [pos_flag, unrealised]]))

    def _vol_norm(self, r: float) -> float:
        self._ret_buf.append(r)
        vol = float(np.std(self._ret_buf)) if len(self._ret_buf) > 3 else 1.0
        return r / max(vol, 1e-6)

    def _regime_alignment(self, action: int) -> float:
        """Return a bonus/penalty based on whether the action aligns with regime."""
        p_bull = float(self.df.loc[self.step_idx, "p_bull"])
        p_bear = float(self.df.loc[self.step_idx, "p_bear"])
        if action == 1:   # Buy → rewarded in bullish, penalised in bearish
            return CFG.REGIME_BONUS * (p_bull - p_bear)
        if action == 2:   # Sell → rewarded in bearish (closing longs)
            return CFG.REGIME_BONUS * (p_bear - p_bull)
        return 0.0        # Hold is regime-neutral

    # ── Public API ───────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx     = 0
        self.in_position  = False
        self.entry_price  = 0.0
        self.portfolio_v  = CFG.INITIAL_CAPITAL
        self.cash         = CFG.INITIAL_CAPITAL
        self.trades       = 0
        self._ret_buf.clear()
        return self._get_obs(), {}

    def step(self, action: int):
        price  = float(self.df.loc[self.step_idx, "close"])
        reward = 0.0
        info   = {}

        # ── Action execution ────────────────────────────────────
        if action == 1:                                      # BUY
            if not self.in_position:
                self.in_position = True
                self.entry_price = price * (1 + self.transaction_cost)
                self.trades     += 1

        elif action == 2:                                    # SELL / CLOSE
            if self.in_position:
                sell_px   = price * (1 - self.transaction_cost)
                raw_ret   = (sell_px - self.entry_price) / (self.entry_price + 1e-9)
                reward    = self._vol_norm(raw_ret) * 0.8
                self.cash *= (1 + raw_ret)
                self.in_position = False
                self.entry_price = 0.0
                self.trades     += 1
                info["trade_return"] = raw_ret

        elif action == 0:                                    # HOLD
            if self.in_position:
                unreal = (price - self.entry_price) / (self.entry_price + 1e-9)
                reward  = 0.05 * self._vol_norm(unreal) + CFG.HOLD_PENALTY
            else:
                reward = CFG.HOLD_PENALTY                   # Nudge agent to act

        # ── Regime alignment bonus ──────────────────────────────
        reward += self._regime_alignment(action)

        # ── Portfolio valuation ─────────────────────────────────
        self.portfolio_v = self.cash * (
            (1 + (price - self.entry_price) / (self.entry_price + 1e-9))
            if self.in_position else 1.0
        )

        # ── Advance timestep ────────────────────────────────────
        self.step_idx += 1
        done = self.step_idx >= self.max_steps

        if done and self.in_position:
            close_px        = float(self.df.loc[self.step_idx, "close"])
            sell_px         = close_px * (1 - self.transaction_cost)
            raw_ret         = (sell_px - self.entry_price) / (self.entry_price + 1e-9)
            reward         += self._vol_norm(raw_ret)
            self.cash      *= (1 + raw_ret)
            self.portfolio_v = self.cash
            self.in_position = False

        obs = self._get_obs()
        return obs, float(reward) * CFG.REWARD_SCALE, done, False, {
            "portfolio_value": self.portfolio_v,
            "trades"         : self.trades,
        }


# ══════════════════════════════════════════════════════════════════
#  EVALUATION & METRICS
# ══════════════════════════════════════════════════════════════════
def evaluate_agent(model, df: pd.DataFrame, feature_cols: list, label: str = ""):
    env  = DiscreteTradingEnv(df, feature_cols)
    obs, _ = env.reset()
    done   = False
    port_vals  = [CFG.INITIAL_CAPITAL]
    actions    = []
    step_rets  = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        actions.append(int(action))
        curr_val = info["portfolio_value"]
        if len(port_vals) > 0:
            step_rets.append((curr_val - port_vals[-1]) / (port_vals[-1] + 1e-9))
        port_vals.append(curr_val)

    pv      = np.array(port_vals)
    rets    = np.array(step_rets)
    cum_ret = (pv[-1] - CFG.INITIAL_CAPITAL) / CFG.INITIAL_CAPITAL
    bh_ret  = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    vol     = np.std(rets) * np.sqrt(252) + 1e-9
    sharpe  = (np.mean(rets) * 252) / vol

    downside = rets[rets < 0]
    sortino  = (np.mean(rets) * 252) / (np.std(downside) * np.sqrt(252) + 1e-9)

    peak = np.maximum.accumulate(pv)
    dd   = ((pv - peak) / (peak + 1e-9)).min()

    calmar = (cum_ret / abs(dd)) if dd != 0 else 0.0

    print(f"\n── {label} Results ──────────────────────────────────────────")
    print(f"  Cumulative Return : {cum_ret:+.2%}   |  Buy & Hold : {bh_ret:+.2%}")
    print(f"  Sharpe Ratio      : {sharpe:.3f}")
    print(f"  Sortino Ratio     : {sortino:.3f}")
    print(f"  Max Drawdown      : {dd:.2%}")
    print(f"  Calmar Ratio      : {calmar:.3f}")
    print(f"  Total Trades      : {env.trades}")
    act_cnt = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
    print(f"  Action Dist       : Hold={act_cnt[0]}  Buy={act_cnt[1]}  Sell={act_cnt[2]}")

    return {
        "portfolio_values": pv,
        "actions"         : actions,
        "step_returns"    : rets,
        "cum_ret"         : cum_ret,
        "bh_ret"          : bh_ret,
        "sharpe"          : sharpe,
        "sortino"         : sortino,
        "max_drawdown"    : dd,
        "calmar"          : calmar,
        "trades"          : env.trades,
        "prices"          : df["close"].values,
    }


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
def plot_all_results(baseline_res: dict, regime_res: dict, test_df: pd.DataFrame):
    """Generate and save a comprehensive set of result plots."""
    print("\n[Plot] Generating all result plots...")
    plot_summary_dashboard(baseline_res, regime_res)
    plot_separate_charts(baseline_res, regime_res, test_df)
    print(f"[Plot] All plots saved in → {CFG.PLOT_PATH}")

def plot_summary_dashboard(baseline_res: dict, regime_res: dict):
    """Plots the main summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("PPO + HMM  ·  Discrete Action Space  (Hold / Buy / Sell)",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Portfolio comparison ──
    ax = fig.add_subplot(gs[0, :2])
    bh_b = CFG.INITIAL_CAPITAL * (1 + (baseline_res["prices"] - baseline_res["prices"][0])
                                   / baseline_res["prices"][0])
    ax.plot(baseline_res["portfolio_values"][:len(baseline_res["prices"])],
            color="#1f77b4", linewidth=1.5, label=f"Baseline  (ret={baseline_res['cum_ret']:+.1%})")
    ax.plot(regime_res["portfolio_values"][:len(regime_res["prices"])],
            color="#2ca02c", linewidth=1.5, label=f"Regime-Aware  (ret={regime_res['cum_ret']:+.1%})")
    ax.plot(bh_b, color="gray", linewidth=1, linestyle="--", label="Buy & Hold")
    ax.axhline(CFG.INITIAL_CAPITAL, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Portfolio Value — Test Set"); ax.set_xlabel("Timestep")
    ax.set_ylabel("Portfolio (₹)"); ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Metrics bar chart ──
    ax = fig.add_subplot(gs[0, 2])
    metrics = ["Sharpe", "Sortino", "Calmar"]
    b_vals  = [baseline_res["sharpe"], baseline_res["sortino"], baseline_res["calmar"]]
    r_vals  = [regime_res["sharpe"],  regime_res["sortino"],  regime_res["calmar"]]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x - w/2, b_vals, w, label="Baseline",     color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, r_vals, w, label="Regime-Aware", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Risk-Adjusted Metrics"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── 3. Drawdown comparison ──
    ax = fig.add_subplot(gs[1, :2])
    for res, color, lbl in [
        (baseline_res, "#1f77b4", "Baseline"),
        (regime_res,   "#2ca02c", "Regime-Aware"),
    ]:
        pv  = res["portfolio_values"]
        pk  = np.maximum.accumulate(pv)
        dd  = (pv - pk) / (pk + 1e-9)
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.35, color=color,
                        label=f"{lbl}  (max={res['max_drawdown']:.1%})")
    ax.set_title("Drawdown — Test Set"); ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown %"); ax.legend(); ax.grid(alpha=0.3)

    # ── 4. Action distribution ──
    ax = fig.add_subplot(gs[1, 2])
    labels = ["Hold", "Buy", "Sell"]
    colors = ["#aaaaaa", "#2ca02c", "#d62728"]
    for res, hatch, lbl in [
        (baseline_res, "",   "Baseline"),
        (regime_res,   "//", "Regime-Aware"),
    ]:
        acts   = res["actions"]
        counts = [acts.count(i) for i in range(3)]
        x      = np.arange(3)
        w      = 0.35
        off    = -w/2 if hatch == "" else w/2
        bars   = ax.bar(x + off, counts, w, label=lbl,
                        color=colors, alpha=0.8, hatch=hatch)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
    ax.set_title("Action Distribution (Test)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(CFG.PLOT_PATH, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_separate_charts(baseline_res: dict, regime_res: dict, test_df: pd.DataFrame):
    """Generates and saves individual, more detailed plots for analysis."""
    dates = test_df['date'].reset_index(drop=True)

    # 1. Portfolio Growth
    fig, ax = plt.subplots(figsize=(12, 7))
    bh_series = CFG.INITIAL_CAPITAL * (1 + test_df['close'].pct_change().cumsum().fillna(0))
    ax.plot(dates, baseline_res["portfolio_values"][:len(dates)], label=f"Baseline (Sharpe: {baseline_res['sharpe']:.2f})")
    ax.plot(dates, regime_res["portfolio_values"][:len(dates)], label=f"Regime-Aware (Sharpe: {regime_res['sharpe']:.2f})")
    ax.plot(dates, bh_series, label="Buy & Hold", linestyle="--", color='grey')
    ax.set_title("Portfolio Growth (Test Set)"); ax.set_xlabel("Date"); ax.set_ylabel("Value (₹)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CFG.PLOT_PATH, "1_portfolio_growth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(12, 7))
    for res, label in [(baseline_res, "Baseline"), (regime_res, "Regime-Aware")]:
        pv = pd.Series(res["portfolio_values"], index=dates[:len(res["portfolio_values"])])
        dd = (pv / pv.cummax()) - 1
        ax.plot(dd, label=f"{label} (Max: {res['max_drawdown']:.2%})")
    ax.set_title("Drawdown (Test Set)"); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CFG.PLOT_PATH, "2_drawdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Actions Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dates, test_df['close'], label="Price", color='grey', alpha=0.5)
    ax.set_ylabel("Price")
    ax2 = ax.twinx()
    buy_signals = [dates[i] for i, a in enumerate(regime_res['actions']) if a == 1]
    sell_signals = [dates[i] for i, a in enumerate(regime_res['actions']) if a == 2]
    ax2.plot(buy_signals, [1]*len(buy_signals), '^', color='green', markersize=8, label='Buy')
    ax2.plot(sell_signals, [0]*len(sell_signals), 'v', color='red', markersize=8, label='Sell')
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([])
    ax.set_title("Regime-Aware Agent Actions on Price"); ax.set_xlabel("Date")
    fig.legend(loc="upper right")
    plt.savefig(os.path.join(CFG.PLOT_PATH, "3_actions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Rolling Sharpe
    fig, ax = plt.subplots(figsize=(12, 7))
    for res, label in [(baseline_res, "Baseline"), (regime_res, "Regime-Aware")]:
        rets = pd.Series(res["step_returns"], index=dates[:len(res["step_returns"])])
        rolling_sharpe = rets.rolling(window=60).mean() / rets.rolling(window=60).std() * np.sqrt(252)
        ax.plot(rolling_sharpe, label=f"{label} Rolling Sharpe (60-day)")
    ax.set_title("Rolling Sharpe Ratio"); ax.set_xlabel("Date"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CFG.PLOT_PATH, "4_rolling_sharpe.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Monthly Returns Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, (res, label) in enumerate([(baseline_res, "Baseline"), (regime_res, "Regime-Aware")]):
        ax = (ax1, ax2)[i]
        rets = pd.Series(res["step_returns"], index=dates[:len(res["step_returns"])])
        monthly_rets = rets.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        monthly_rets.index = pd.to_datetime(monthly_rets.index)
        monthly_rets_df = pd.DataFrame({
            'year': monthly_rets.index.year,
            'month': monthly_rets.index.month,
            'return': monthly_rets.values
        })
        heatmap_df = monthly_rets_df.pivot(index='year', columns='month', values='return')
        
        import seaborn as sns
        sns.heatmap(heatmap_df, ax=ax, annot=True, fmt='.1%', cmap='viridis', cbar=False)
        ax.set_title(f"{label} Monthly Returns")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.PLOT_PATH, "5_monthly_returns_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_results(baseline_res: dict, regime_res: dict):
    # This function is kept for compatibility but the main logic is moved
    # to plot_summary_dashboard.
    plot_summary_dashboard(baseline_res, regime_res)
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def run(total_timesteps: int = CFG.TOTAL_TIMESTEPS, train_split: float = CFG.TRAIN_SPLIT):
    np.random.seed(CFG.SEED)

    print("=" * 60)
    print("  PPO + HMM  ·  DISCRETE Action Space")
    print(f"  Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ── Load & prepare ───────────────────────────────────────────
    print("\n[Data] Loading …")
    df        = load_and_prepare(CFG.DATA_PATH)
    regime_df, REGIME_COLS = generate_regime_features(df, train_split)
    full_df   = df.join(regime_df).ffill().bfill().fillna(0)

    split_idx = int(len(full_df) * train_split)
    train_df  = rolling_normalize(full_df.iloc[:split_idx].copy(), BASE_FEATURES)
    test_df   = rolling_normalize(full_df.iloc[split_idx:].copy(), BASE_FEATURES)

    # Clip outliers
    for col in BASE_FEATURES:
        train_df[col] = train_df[col].clip(-3, 3)
        test_df[col]  = test_df[col].clip(-3, 3)

    print(f"[Data] Train={len(train_df)} rows | Test={len(test_df)} rows")
    print(f"[Data] Baseline features={len(BASE_FEATURES)} | "
          f"Regime features={len(REGIME_COLS)} → total={len(BASE_FEATURES)+len(REGIME_COLS)}")

    # ── PPO hyperparameters ──────────────────────────────────────
    ppo_kwargs = dict(
        learning_rate = CFG.LEARNING_RATE,
        n_steps       = CFG.N_STEPS,
        batch_size    = CFG.BATCH_SIZE,
        n_epochs      = CFG.N_EPOCHS,
        gamma         = CFG.GAMMA,
        gae_lambda    = CFG.GAE_LAMBDA,
        clip_range    = CFG.CLIP_RANGE,
        ent_coef      = CFG.ENT_COEF,
        vf_coef       = CFG.VF_COEF,
        max_grad_norm = CFG.MAX_GRAD_NORM,
        policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])]),
        verbose       = 1,
        seed          = CFG.SEED,
    )

    # ── Train BASELINE (no regime features) ─────────────────────
    print(f"\n[Train] Baseline PPO ({total_timesteps:,} steps) …")
    train_env_b = Monitor(DiscreteTradingEnv(train_df, BASE_FEATURES))
    eval_env_b  = Monitor(DiscreteTradingEnv(train_df, BASE_FEATURES))
    eval_cb_b   = EvalCallback(
        eval_env_b, best_model_save_path=f"{CFG.SAVE_PATH}_baseline",
        eval_freq=10_000, n_eval_episodes=3, deterministic=True, verbose=0
    )
    model_b = PPO("MlpPolicy", train_env_b, **ppo_kwargs)
    model_b.learn(total_timesteps, callback=eval_cb_b, progress_bar=False)

    # ── Train REGIME-AWARE ───────────────────────────────────────
    all_feat_cols = BASE_FEATURES + REGIME_COLS
    print(f"\n[Train] Regime-Aware PPO ({total_timesteps:,} steps) …")
    train_env_r = Monitor(DiscreteTradingEnv(train_df, all_feat_cols))
    eval_env_r  = Monitor(DiscreteTradingEnv(train_df, all_feat_cols))
    eval_cb_r   = EvalCallback(
        eval_env_r, best_model_save_path=f"{CFG.SAVE_PATH}_regime",
        eval_freq=10_000, n_eval_episodes=3, deterministic=True, verbose=0
    )
    model_r = PPO("MlpPolicy", train_env_r, **ppo_kwargs)
    model_r.learn(total_timesteps, callback=eval_cb_r, progress_bar=False)

    # ── Evaluate on test set ────────────────────────────────────
    print("\n[Eval] Running test-set backtest …")
    baseline_res = evaluate_agent(model_b, test_df, BASE_FEATURES,       label="Baseline (No HMM)")
    regime_res   = evaluate_agent(model_r, test_df, all_feat_cols, label="Regime-Aware (HMM)")

    # ── Save models ─────────────────────────────────────────────
    model_b.save(f"{CFG.SAVE_PATH}_baseline_final")
    model_r.save(f"{CFG.SAVE_PATH}_regime_final")
    print(f"\n[Model] Saved → {CFG.SAVE_PATH}_baseline_final.zip")
    print(f"[Model] Saved → {CFG.SAVE_PATH}_regime_final.zip")

    # ── Plot ─────────────────────────────────────────────────────
    plot_all_results(baseline_res, regime_res, test_df)

    return model_b, model_r, baseline_res, regime_res


if __name__ == "__main__":
    run()
