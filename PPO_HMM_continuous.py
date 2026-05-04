"""
PPO + HMM Regime Integration — CONTINUOUS Action Space
=======================================================
Action  : Scalar in [-1, 1]
           > 0  → long exposure (fraction of capital)
           = 0  → flat / no position
           < 0  → short exposure (fraction of capital)

Regime  : 3-state Gaussian HMM → p_bull / p_sideways / p_bear
          appended to observation so the policy learns *how much*
          to size based on market regime.

Key Improvements over baseline:
  - Sortino-inspired reward: penalise downside volatility only
  - Regime position cap: max long in bearish regime is cut in half
  - Smooth transaction cost proportional to change in position size
  - EvalCallback + LR schedule for stable long-run training
  - Ruin prevention: episode terminates + heavy penalty if wealth < 20%
  - Detailed metrics: Sharpe, Sortino, Calmar, Max DD, Win Rate
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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sklearn.preprocessing import StandardScaler

from regime_detector import RegimeDetector
from regime_features import extract_features, compute_trend_slope, compute_rsi


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
class Config:
    DATA_PATH        = "Data/apple/apple_rl_features.csv"
    TRAIN_SPLIT      = 0.75

    # Environment
    TRANSACTION_COST = 0.0010        # 10 bps – proportional to position change
    RUIN_THRESHOLD   = 0.20          # Force close if wealth drops below 20%
    VOL_WINDOW       = 30

    # Regime HMM
    N_REGIMES        = 3
    HMM_ITER         = 300

    # PPO Hyperparameters
    TOTAL_TIMESTEPS  = 500_000
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
    REWARD_SCALE       = 100.0
    DOWNSIDE_PENALTY   = 5.0         # Multiplier for downside variance penalty
    PROFIT_BONUS       = 0.10        # Bonus on positive step P&L
    RUIN_PENALTY       = -200.0      # Terminal penalty for blowing up

    # Regime caps: max absolute position in each regime
    BULL_MAX_POS     = 1.0
    SIDEWAYS_MAX_POS = 0.5
    BEAR_MAX_POS     = 0.3

    SEED             = 42
    SAVE_PATH        = "ppo_hmm_continuous_model"
    PLOT_PATH        = "Results/continuous/"

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

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"]        = pd.to_datetime(df["date"])
    df["trend_slope"] = compute_trend_slope(df["close"].values)
    df["rsi_14"]      = compute_rsi(df["close"].values)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.ffill().bfill().fillna(0)

    # Next-bar realised return (label shift)
    df["return_1"] = df["close"].pct_change().shift(-1).fillna(0).clip(-0.10, 0.10)
    return df


def rolling_normalize(df: pd.DataFrame, cols: list, window: int = 50) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        mu    = out[col].rolling(window, min_periods=1).mean()
        sigma = out[col].rolling(window, min_periods=1).std()
        out[col] = (out[col] - mu) / (sigma + 1e-6)
    return out.fillna(0)


def generate_regime_features(df: pd.DataFrame, train_split: float = 0.75):
    """Fit GaussianHMM on training window; produce p_bull / p_sideways / p_bear."""
    print("[HMM] Fitting regime detector …")
    split_idx    = int(len(df) * train_split)
    raw_feats    = extract_features(df)
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(raw_feats[:split_idx])
    test_scaled  = scaler.transform(raw_feats[split_idx:])

    detector = RegimeDetector(n_states=CFG.N_REGIMES, random_state=CFG.SEED)
    detector.model.n_iter = CFG.HMM_ITER
    detector.fit(train_scaled)

    all_scaled = np.vstack([train_scaled, test_scaled])
    states     = detector.predict_states(all_scaled)
    all_probs  = detector.predict_probabilities(all_scaled)

    # Map HMM indices to bull / bear / sideways by average return
    stats = {}
    for s in range(CFG.N_REGIMES):
        mask    = states == s
        avg_ret = df.loc[mask, "log_return_1"].mean() if "log_return_1" in df.columns else 0.0
        stats[s] = avg_ret

    sorted_states = sorted(stats, key=lambda s: stats[s])
    bear_idx, side_idx, bull_idx = sorted_states[0], sorted_states[1], sorted_states[2]

    regime_cols = ["p_bull", "p_sideways", "p_bear"]
    prob_df = pd.DataFrame({
        "p_bull"    : all_probs[:, bull_idx],
        "p_sideways": all_probs[:, side_idx],
        "p_bear"    : all_probs[:, bear_idx],
    }, index=df.index)

    print(f"[HMM] Mapping → bull={bull_idx}  sideways={side_idx}  bear={bear_idx}")
    return prob_df, regime_cols


# ══════════════════════════════════════════════════════════════════
#  CONTINUOUS TRADING ENVIRONMENT
# ══════════════════════════════════════════════════════════════════
class ContinuousTradingEnv(gym.Env):
    """
    Continuous-action trading environment with regime-adaptive position caps.

    The policy outputs a target position in [-1, 1]:
      +1  → fully long
       0  → flat
      -1  → fully short

    The regime probabilities soften the effective cap so the agent is
    discouraged from holding large longs when HMM signals bearish conditions.

    Observation
    -----------
    [normalised market features] + [regime probs (3)]
                                 + [current_position, unrealised_pnl_pct]
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 transaction_cost: float = CFG.TRANSACTION_COST):
        super().__init__()
        self.df               = df.reset_index(drop=True)
        self.feature_cols     = feature_cols
        self.transaction_cost = transaction_cost
        self.max_steps        = len(self.df) - 1

        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim                = len(feature_cols) + 2         # +position +unrealised
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._ret_buf = deque(maxlen=CFG.VOL_WINDOW)
        self.reset()

    # ── Regime-adaptive position cap ─────────────────────────────
    def _regime_cap(self) -> float:
        """Dynamic max position based on current regime probability."""
        p_bull = float(self.df.loc[self.step_idx, "p_bull"]) \
                 if "p_bull" in self.df.columns else 0.33
        p_bear = float(self.df.loc[self.step_idx, "p_bear"]) \
                 if "p_bear" in self.df.columns else 0.33
        p_side = 1.0 - p_bull - p_bear

        cap = (  CFG.BULL_MAX_POS     * p_bull
               + CFG.SIDEWAYS_MAX_POS * p_side
               + CFG.BEAR_MAX_POS     * p_bear)
        return float(np.clip(cap, 0.1, 1.0))

    def _get_obs(self) -> np.ndarray:
        feats      = self.df.loc[self.step_idx, self.feature_cols].values.astype(np.float32)
        unrealised = np.float32(0.0)
        if abs(self.position) > 1e-4 and self.entry_price > 0:
            curr_px    = float(self.df.loc[self.step_idx, "close"])
            sign       = 1.0 if self.position > 0 else -1.0
            unrealised = np.float32(
                sign * (curr_px - self.entry_price) / (self.entry_price + 1e-9)
            )
        return np.nan_to_num(
            np.concatenate([feats, [np.float32(self.position), unrealised]])
        )

    def _vol_norm(self, r: float) -> float:
        self._ret_buf.append(r)
        vol = float(np.std(self._ret_buf)) if len(self._ret_buf) > 3 else 1.0
        return r / max(vol, 1e-6)

    # ── Public API ───────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx     = 0
        self.position     = 0.0
        self.portfolio_v  = 1.0         # normalised to 1.0 for simplicity
        self.entry_price  = 0.0
        self.done         = False
        self._ret_buf.clear()
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"portfolio_value": self.portfolio_v}

        raw_target = float(np.clip(action[0], -1.0, 1.0))

        # Apply regime-adaptive cap (longs limited in bearish regimes)
        cap            = self._regime_cap()
        target_pos     = np.clip(raw_target, -cap, cap)

        step_return    = float(self.df.loc[self.step_idx, "return_1"])
        curr_price     = float(self.df.loc[self.step_idx, "close"])

        # ── Transaction cost (proportional to position change) ───
        trade_size    = abs(target_pos - self.position)
        trade_penalty = trade_size * self.transaction_cost

        # ── Entry price tracking ─────────────────────────────────
        if abs(target_pos) > 0.01 and abs(self.position) < 0.01:
            self.entry_price = curr_price
        elif abs(target_pos) < 0.01:
            self.entry_price = 0.0

        # ── Raw step P&L ─────────────────────────────────────────
        raw_pnl = target_pos * step_return - trade_penalty

        # ── Sortino-inspired reward shaping ──────────────────────
        downside_penalty = CFG.DOWNSIDE_PENALTY * (min(0.0, raw_pnl) ** 2)
        profit_bonus     = CFG.PROFIT_BONUS * max(0.0, raw_pnl)

        reward = (raw_pnl + profit_bonus - downside_penalty) * CFG.REWARD_SCALE

        # ── State update ─────────────────────────────────────────
        self.portfolio_v *= (1.0 + raw_pnl)
        self.position     = target_pos
        self.step_idx    += 1

        # ── Termination conditions ───────────────────────────────
        if self.portfolio_v <= CFG.RUIN_THRESHOLD:
            reward    += CFG.RUIN_PENALTY
            self.done  = True

        elif self.step_idx >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {
            "portfolio_value": self.portfolio_v
        }


# ══════════════════════════════════════════════════════════════════
#  EVALUATION & METRICS
# ══════════════════════════════════════════════════════════════════
def evaluate_agent(model, df: pd.DataFrame, feature_cols: list, label: str = ""):
    env    = ContinuousTradingEnv(df, feature_cols)
    obs, _ = env.reset()
    done   = False

    port_vals  = [1.0]
    positions  = []
    step_rets  = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        curr_val = info["portfolio_value"]
        step_ret = (curr_val - port_vals[-1]) / (port_vals[-1] + 1e-9)
        step_rets.append(step_ret)
        port_vals.append(curr_val)
        positions.append(float(action[0]))

    pv      = np.array(port_vals)
    rets    = np.array(step_rets)
    cum_ret = pv[-1] - 1.0
    bh_ret  = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    vol    = np.std(rets) * np.sqrt(252) + 1e-9
    sharpe = (np.mean(rets) * 252) / vol

    dn_std  = np.std(rets[rets < 0]) * np.sqrt(252) + 1e-9
    sortino = (np.mean(rets) * 252) / dn_std

    peak   = np.maximum.accumulate(pv)
    dd     = ((pv - peak) / (peak + 1e-9)).min()
    calmar = cum_ret / abs(dd) if dd != 0 else 0.0

    win_rate = float(np.mean(np.array(rets) > 0))

    # Average absolute position size (aggressiveness measure)
    avg_pos = float(np.mean(np.abs(positions)))

    print(f"\n── {label} Results ──────────────────────────────────────────")
    print(f"  Cumulative Return : {cum_ret:+.2%}   |  Buy & Hold : {bh_ret:+.2%}")
    print(f"  Sharpe Ratio      : {sharpe:.3f}")
    print(f"  Sortino Ratio     : {sortino:.3f}")
    print(f"  Max Drawdown      : {dd:.2%}")
    print(f"  Calmar Ratio      : {calmar:.3f}")
    print(f"  Win Rate          : {win_rate:.1%}")
    print(f"  Avg |Position|    : {avg_pos:.3f}")

    return {
        "portfolio_values": pv,
        "positions"       : positions,
        "step_returns"    : rets,
        "cum_ret"         : cum_ret,
        "bh_ret"          : bh_ret,
        "sharpe"          : sharpe,
        "sortino"         : sortino,
        "max_drawdown"    : dd,
        "calmar"          : calmar,
        "win_rate"        : win_rate,
        "avg_pos"         : avg_pos,
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
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("PPO + HMM  ·  Continuous Action Space  (Position Sizing  −1 → +1)",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Portfolio growth ──
    ax = fig.add_subplot(gs[0, :2])
    bh_b = 1 + (baseline_res["prices"] - baseline_res["prices"][0]) / baseline_res["prices"][0]
    ax.plot(baseline_res["portfolio_values"][:len(baseline_res["prices"])],
            color="#1f77b4", linewidth=1.5, label=f"Baseline  (ret={baseline_res['cum_ret']:+.1%})")
    ax.plot(regime_res["portfolio_values"][:len(regime_res["prices"])],
            color="#2ca02c", linewidth=1.5, label=f"Regime-Aware  (ret={regime_res['cum_ret']:+.1%})")
    ax.plot(bh_b, color="gray", linewidth=1, linestyle="--", label="Buy & Hold")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Portfolio Value (normalised to 1.0) — Test Set")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Portfolio Value"); ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Risk-adjusted metrics ──
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

    # ── 3. Drawdown ──
    ax = fig.add_subplot(gs[1, :2])
    for res, color, lbl in [
        (baseline_res, "#1f77b4", "Baseline"),
        (regime_res,   "#2ca02c", "Regime-Aware"),
    ]:
        pv = res["portfolio_values"]
        pk = np.maximum.accumulate(pv)
        dd = (pv - pk) / (pk + 1e-9)
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.35, color=color,
                        label=f"{lbl}  (max={res['max_drawdown']:.1%})")
    ax.set_title("Drawdown — Test Set"); ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown"); ax.legend(); ax.grid(alpha=0.3)

    # ── 4. Position sizing over time (regime-aware) ──
    ax = fig.add_subplot(gs[1, 2])
    pos = np.array(regime_res["positions"])
    ax.plot(pos, color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(range(len(pos)), pos, 0,
                    where=(pos > 0), color="#2ca02c", alpha=0.3, label="Long")
    ax.fill_between(range(len(pos)), pos, 0,
                    where=(pos < 0), color="#d62728", alpha=0.3, label="Short")
    ax.set_title("Regime-Aware Position Sizing (Test)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Position (−1 to +1)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-1.1, 1.1)

    # ── 5. Return distribution ──
    ax = fig.add_subplot(gs[2, :2])
    rets_b = baseline_res["step_returns"]
    rets_r = regime_res["step_returns"]
    ax.hist(rets_b, bins=60, alpha=0.5, color="#1f77b4", label="Baseline", density=True)
    ax.hist(rets_r, bins=60, alpha=0.5, color="#2ca02c", label="Regime-Aware", density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Step Return Distribution — Test Set")
    ax.set_xlabel("Step Return"); ax.set_ylabel("Density"); ax.legend(); ax.grid(alpha=0.3)

    # ── 6. Summary scorecard ──
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary = [
        ["Metric",          "Baseline",                       "Regime-Aware"],
        ["Cum. Return",     f"{baseline_res['cum_ret']:+.2%}", f"{regime_res['cum_ret']:+.2%}"],
        ["Buy & Hold",      f"{baseline_res['bh_ret']:+.2%}", "—"],
        ["Sharpe",          f"{baseline_res['sharpe']:.3f}",  f"{regime_res['sharpe']:.3f}"],
        ["Sortino",         f"{baseline_res['sortino']:.3f}", f"{regime_res['sortino']:.3f}"],
        ["Max Drawdown",    f"{baseline_res['max_drawdown']:.2%}", f"{regime_res['max_drawdown']:.2%}"],
        ["Calmar",          f"{baseline_res['calmar']:.3f}",  f"{regime_res['calmar']:.3f}"],
        ["Win Rate",        f"{baseline_res['win_rate']:.1%}", f"{regime_res['win_rate']:.1%}"],
        ["Avg |Position|",  f"{baseline_res['avg_pos']:.3f}", f"{regime_res['avg_pos']:.3f}"],
    ]
    tbl = ax.table(cellText=summary[1:], colLabels=summary[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    ax.set_title("Performance Scorecard", pad=12)

    plt.savefig(os.path.join(CFG.PLOT_PATH, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_separate_charts(baseline_res: dict, regime_res: dict, test_df: pd.DataFrame):
    """Generates and saves individual, more detailed plots for analysis."""
    dates = test_df['date']

    # 1. Portfolio Growth
    fig, ax = plt.subplots(figsize=(12, 7))
    bh_series = (1 + test_df['close'].pct_change().cumsum().fillna(0))
    ax.plot(dates, baseline_res["portfolio_values"][:len(dates)], label=f"Baseline (Sharpe: {baseline_res['sharpe']:.2f})")
    ax.plot(dates, regime_res["portfolio_values"][:len(dates)], label=f"Regime-Aware (Sharpe: {regime_res['sharpe']:.2f})")
    ax.plot(dates, bh_series, label="Buy & Hold", linestyle="--", color='grey')
    ax.set_title("Portfolio Growth (Test Set)"); ax.set_xlabel("Date"); ax.set_ylabel("Value")
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

    # 3. Positions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(dates[:len(baseline_res["positions"])], baseline_res["positions"], label="Baseline Positions")
    ax1.set_title("Baseline Agent Positions"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(dates[:len(regime_res["positions"])], regime_res["positions"], label="Regime-Aware Positions", color='green')
    ax2.set_title("Regime-Aware Agent Positions"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.xlabel("Date")
    plt.savefig(os.path.join(CFG.PLOT_PATH, "3_positions.png"), dpi=150, bbox_inches="tight")
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
        
        # Create a pivot table for the heatmap
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

    # 6. Cumulative Returns vs Price
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dates, regime_res["portfolio_values"][:len(dates)], label="Regime-Aware Cumulative Return", color='green')
    ax.set_ylabel("Cumulative Return", color='green')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(dates, test_df['close'], label="Price", color='grey', alpha=0.6)
    ax2.set_ylabel("Price", color='grey')
    ax2.legend(loc='upper right')
    ax.set_title("Regime-Aware Cumulative Returns vs. Price")
    plt.savefig(os.path.join(CFG.PLOT_PATH, "6_cumulative_return_vs_price.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(baseline_res: dict, regime_res: dict):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("PPO + HMM  ·  Continuous Action Space  (Position Sizing  −1 → +1)",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Portfolio growth ──
    ax = fig.add_subplot(gs[0, :2])
    bh_b = 1 + (baseline_res["prices"] - baseline_res["prices"][0]) / baseline_res["prices"][0]
    ax.plot(baseline_res["portfolio_values"][:len(baseline_res["prices"])],
            color="#1f77b4", linewidth=1.5, label=f"Baseline  (ret={baseline_res['cum_ret']:+.1%})")
    ax.plot(regime_res["portfolio_values"][:len(regime_res["prices"])],
            color="#2ca02c", linewidth=1.5, label=f"Regime-Aware  (ret={regime_res['cum_ret']:+.1%})")
    ax.plot(bh_b, color="gray", linewidth=1, linestyle="--", label="Buy & Hold")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Portfolio Value (normalised to 1.0) — Test Set")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Portfolio Value"); ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Risk-adjusted metrics ──
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

    # ── 3. Drawdown ──
    ax = fig.add_subplot(gs[1, :2])
    for res, color, lbl in [
        (baseline_res, "#1f77b4", "Baseline"),
        (regime_res,   "#2ca02c", "Regime-Aware"),
    ]:
        pv = res["portfolio_values"]
        pk = np.maximum.accumulate(pv)
        dd = (pv - pk) / (pk + 1e-9)
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.35, color=color,
                        label=f"{lbl}  (max={res['max_drawdown']:.1%})")
    ax.set_title("Drawdown — Test Set"); ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown"); ax.legend(); ax.grid(alpha=0.3)

    # ── 4. Position sizing over time (regime-aware) ──
    ax = fig.add_subplot(gs[1, 2])
    pos = np.array(regime_res["positions"])
    ax.plot(pos, color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(range(len(pos)), pos, 0,
                    where=(pos > 0), color="#2ca02c", alpha=0.3, label="Long")
    ax.fill_between(range(len(pos)), pos, 0,
                    where=(pos < 0), color="#d62728", alpha=0.3, label="Short")
    ax.set_title("Regime-Aware Position Sizing (Test)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Position (−1 to +1)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-1.1, 1.1)

    # ── 5. Return distribution ──
    ax = fig.add_subplot(gs[2, :2])
    rets_b = baseline_res["step_returns"]
    rets_r = regime_res["step_returns"]
    ax.hist(rets_b, bins=60, alpha=0.5, color="#1f77b4", label="Baseline", density=True)
    ax.hist(rets_r, bins=60, alpha=0.5, color="#2ca02c", label="Regime-Aware", density=True)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Step Return Distribution — Test Set")
    ax.set_xlabel("Step Return"); ax.set_ylabel("Density"); ax.legend(); ax.grid(alpha=0.3)

    # ── 6. Summary scorecard ──
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary = [
        ["Metric",          "Baseline",                       "Regime-Aware"],
        ["Cum. Return",     f"{baseline_res['cum_ret']:+.2%}", f"{regime_res['cum_ret']:+.2%}"],
        ["Buy & Hold",      f"{baseline_res['bh_ret']:+.2%}", "—"],
        ["Sharpe",          f"{baseline_res['sharpe']:.3f}",  f"{regime_res['sharpe']:.3f}"],
        ["Sortino",         f"{baseline_res['sortino']:.3f}", f"{regime_res['sortino']:.3f}"],
        ["Max Drawdown",    f"{baseline_res['max_drawdown']:.2%}", f"{regime_res['max_drawdown']:.2%}"],
        ["Calmar",          f"{baseline_res['calmar']:.3f}",  f"{regime_res['calmar']:.3f}"],
        ["Win Rate",        f"{baseline_res['win_rate']:.1%}", f"{regime_res['win_rate']:.1%}"],
        ["Avg |Position|",  f"{baseline_res['avg_pos']:.3f}", f"{regime_res['avg_pos']:.3f}"],
    ]
    tbl = ax.table(cellText=summary[1:], colLabels=summary[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    ax.set_title("Performance Scorecard", pad=12)

    plt.savefig(os.path.join(CFG.PLOT_PATH, "summary_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Plot] Saved → {os.path.join(CFG.PLOT_PATH, 'summary_dashboard.png')}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def run(total_timesteps: int = CFG.TOTAL_TIMESTEPS, train_split: float = CFG.TRAIN_SPLIT):
    np.random.seed(CFG.SEED)

    print("=" * 60)
    print("  PPO + HMM  ·  CONTINUOUS Action Space")
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
        # Separate actor/critic nets → better for continuous actions
        policy_kwargs = dict(net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]),
        verbose       = 1,
        seed          = CFG.SEED,
    )

    all_feat_cols = BASE_FEATURES + REGIME_COLS

    # ── Train BASELINE ───────────────────────────────────────────
    print(f"\n[Train] Baseline PPO — {total_timesteps:,} steps …")
    train_env_b = Monitor(ContinuousTradingEnv(train_df, BASE_FEATURES))
    eval_env_b  = Monitor(ContinuousTradingEnv(train_df, BASE_FEATURES))
    eval_cb_b   = EvalCallback(
        eval_env_b, best_model_save_path=f"{CFG.SAVE_PATH}_baseline",
        eval_freq=10_000, n_eval_episodes=3, deterministic=True, verbose=0
    )
    model_b = PPO("MlpPolicy", train_env_b, **ppo_kwargs)
    model_b.learn(total_timesteps, callback=eval_cb_b, progress_bar=False)

    # ── Train REGIME-AWARE ───────────────────────────────────────
    print(f"\n[Train] Regime-Aware PPO — {total_timesteps:,} steps …")
    train_env_r = Monitor(ContinuousTradingEnv(train_df, all_feat_cols))
    eval_env_r  = Monitor(ContinuousTradingEnv(train_df, all_feat_cols))
    eval_cb_r   = EvalCallback(
        eval_env_r, best_model_save_path=f"{CFG.SAVE_PATH}_regime",
        eval_freq=10_000, n_eval_episodes=3, deterministic=True, verbose=0
    )
    model_r = PPO("MlpPolicy", train_env_r, **ppo_kwargs)
    model_r.learn(total_timesteps, callback=eval_cb_r, progress_bar=False)

    # ── Evaluate on test set ────────────────────────────────────
    print("\n[Eval] Running test-set backtest …")
    baseline_res = evaluate_agent(model_b, test_df, BASE_FEATURES,    label="Baseline (No HMM)")
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
