"""
PPO Stock Trading Agent for ITC RL Features Dataset
=====================================================
Strategy:
  - State  : 37 normalized market features per timestep
  - Actions: 0=Hold, 1=Buy, 2=Sell (discrete)
  - Reward : Risk-adjusted P&L with transaction cost penalty
  - Model  : Actor-Critic with shared FC trunk + PPO-clip + GAE
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # ─────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────
    DATA_PATH       = "/kaggle/input/datasets/chrislgeo/stockseer/itc_rl_features.csv"
    TRAIN_SPLIT     = 0.75          # ↓ less training, more realistic test

    # ─────────────────────────────────────────────
    # ENVIRONMENT
    # ─────────────────────────────────────────────
    INITIAL_CASH    = 100_000.0
    TRANSACTION_COST= 0.0015        # ↑ more realistic friction (prevents overtrading)
    HOLD_PENALTY    = -0.0002       # ↑ discourages lazy holding
    VOLATILITY_WINDOW = 30          # ↑ smoother reward normalization

    # ─────────────────────────────────────────────
    # PPO HYPERPARAMETERS (CRITICAL)
    # ─────────────────────────────────────────────
    GAMMA           = 0.995         # ↑ finance needs long-term thinking
    GAE_LAMBDA      = 0.90          # ↓ less variance → more stable
    CLIP_EPS        = 0.10          # ↓ smaller updates (VERY IMPORTANT)
    ENTROPY_COEF    = 0.02          # ↑ more exploration (prevents overfit)
    VALUE_COEF      = 0.40          # ↓ critic dominance
    MAX_GRAD_NORM   = 0.30          # ↓ tighter gradients

    # ─────────────────────────────────────────────
    # NETWORK (REDUCE CAPACITY)
    # ─────────────────────────────────────────────
    HIDDEN_DIM      = 128           # ↓ from 256 → prevents memorization
    N_LAYERS        = 2             # keep same (depth is fine)

    # ─────────────────────────────────────────────
    # TRAINING (MAJOR FIXES HERE)
    # ─────────────────────────────────────────────
    N_EPISODES      = 180           # ↓ from 500 → avoid over-training
    ROLLOUT_STEPS   = 256           # ↓ more frequent updates
    N_EPOCHS        = 5             # ↓ less overfitting per batch
    MINI_BATCH      = 64
    LR              = 1e-4          # ↓ slower learning (VERY IMPORTANT)
    LR_DECAY        = True

    # ─────────────────────────────────────────────
    # MISC
    # ─────────────────────────────────────────────
    SEED            = 42
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    PRINT_EVERY     = 10        # episodes

CFG = Config()


# ─────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(path: str):

    df = pd.read_csv(path, parse_dates=["date"])

    df = df.sort_values("date").reset_index(drop=True)



    price_col = "last_price"



    # ─────────────────────────────────────────────

    # 1. FEATURE ENGINEERING (CRITICAL)

    # ─────────────────────────────────────────────

    

    # Returns (log returns → more stable)

    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))



    # Rolling features

    for w in [5, 10, 20]:

        df[f"ma_{w}"] = df[price_col].rolling(w, min_periods=1).mean()

        df[f"vol_{w}"] = df["log_return"].rolling(w, min_periods=1).std()

        df[f"momentum_{w}"] = df[price_col] / df[price_col].shift(w) - 1



    # RSI (simple version)

    delta = df[price_col].diff()

    gain = (delta.clip(lower=0)).rolling(14, min_periods=1).mean()

    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()

    rs = gain / (loss + 1e-6)

    df["rsi"] = 100 - (100 / (1 + rs))



    # ── BUG FIX: Robust NaN Handling ──

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(axis=1, how="all")  # Drop columns that are completely NaN

    df = df.ffill().bfill().fillna(0)  # Safe fill, avoids destroying rows



    # ─────────────────────────────────────────────

    # 2. FEATURE SELECTION

    # ─────────────────────────────────────────────

    # Ensure only numeric columns are selected

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    feature_cols = [c for c in numeric_cols if c not in ["date", price_col]]



    # Remove near-constant features

    variances = df[feature_cols].var()

    feature_cols = variances[variances > 1e-6].index.tolist()

    

    if len(df) == 0:

        raise ValueError("Dataframe is empty after preprocessing. Check your dataset.")



    # ─────────────────────────────────────────────

    # 3. TRAIN / TEST SPLIT

    # ─────────────────────────────────────────────

    split = int(len(df) * CFG.TRAIN_SPLIT)

    train_df = df.iloc[:split].copy()

    test_df  = df.iloc[split:].copy()



    # ─────────────────────────────────────────────

    # 4. ROLLING NORMALIZATION (VERY IMPORTANT)

    # ─────────────────────────────────────────────

    def rolling_normalize(data, window=50):

        arr = data.copy()

        for col in feature_cols:

            # Added min_periods=1 to prevent injecting new NaNs

            rolling_mean = arr[col].rolling(window, min_periods=1).mean()

            rolling_std  = arr[col].rolling(window, min_periods=1).std()

            arr[col] = (arr[col] - rolling_mean) / (rolling_std + 1e-6)

        return arr.fillna(0)



    train_df = rolling_normalize(train_df)

    test_df  = rolling_normalize(test_df)



    # ─────────────────────────────────────────────

    # 5. ROBUST CLIPPING

    # ─────────────────────────────────────────────

    def clip_outliers(arr):

        return np.clip(arr, -3, 3)



    train_features = clip_outliers(train_df[feature_cols].values.astype(np.float32))

    test_features  = clip_outliers(test_df[feature_cols].values.astype(np.float32))



    train_prices = train_df[price_col].values.astype(np.float32)

    test_prices  = test_df[price_col].values.astype(np.float32)



    # ─────────────────────────────────────────────

    # 6. DATA AUGMENTATION (ANTI-OVERFITTING)

    # ─────────────────────────────────────────────

    noise = np.random.normal(0, 0.01, train_features.shape)

    train_features += noise



    print(f"[Data] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print(f"[Data] Feature dim after engineering: {len(feature_cols)}")



    return train_features, train_prices, test_features, test_prices, feature_cols

# ─────────────────────────────────────────────
# 3. TRADING ENVIRONMENT
# ─────────────────────────────────────────────
class StockTradingEnv:
    """
    Single-asset discrete-action trading environment.

    Actions : 0 = Hold  |  1 = Buy (go long)  |  2 = Sell / Close
    State   : [normalised features] + [position_flag, unrealised_pnl_pct]
    Reward  : Volatility-normalised P&L on close, with transaction cost & hold penalty
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray):
        self.features = features
        self.prices   = prices
        self.n_steps  = len(prices)
        self.obs_dim  = features.shape[1] + 2   # +2 for position encoding

        self._ret_buffer = deque(maxlen=CFG.VOLATILITY_WINDOW)
        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────
    def _get_obs(self):
        pos_flag    = np.float32(self.in_position)
        unrealised  = np.float32(
            (self.prices[self.t] - self.entry_price) / self.entry_price
            if self.in_position else 0.0
        )
        raw_obs = self.features[self.t]
        return np.concatenate([raw_obs, [pos_flag, unrealised]])

    def _vol_normalise(self, r):
        """Divide reward by rolling std of returns for Sharpe-shaped signal."""
        self._ret_buffer.append(r)
        vol = np.std(self._ret_buffer) if len(self._ret_buffer) > 2 else 1.0
        return r / max(vol, 1e-6)

    # ── Public API ────────────────────────────────────────────────────
    def reset(self):
        self.t           = 0
        self.in_position = False
        self.entry_price = 0.0
        self.cash        = CFG.INITIAL_CASH
        self.portfolio_v = CFG.INITIAL_CASH
        self.trades      = 0
        self._ret_buffer.clear()
        return self._get_obs()

    def step(self, action: int):
        price  = self.prices[self.t]
        reward = 0.0
        info   = {}

        # ── Action logic ──────────────────────────────────────────────
        if action == 1:                             # BUY
            if not self.in_position:
                self.in_position = True
                self.entry_price = price * (1 + CFG.TRANSACTION_COST)
                self.trades     += 1

        elif action == 2:                           # SELL
            if self.in_position:
                sell_price       = price * (1 - CFG.TRANSACTION_COST)
                raw_ret          = (sell_price - self.entry_price) / self.entry_price
                reward           = self._vol_normalise(raw_ret) * 0.8
                self.in_position = False
                self.entry_price = 0.0
                self.trades     += 1
                info["trade_return"] = raw_ret

        elif action == 0:  # HOLD
            if self.in_position:
                # reward unrealized pnl (dense signal)
                unrealized = (price - self.entry_price) / self.entry_price
                reward = 0.1 * self._vol_normalise(unrealized) + CFG.HOLD_PENALTY # small cost to hold (encourages action)

        # ── Advance time ──────────────────────────────────────────────
        self.t += 1
        done = self.t >= self.n_steps - 1

        # Force-close at episode end to realise P&L
        if done and self.in_position:
            sell_price        = self.prices[self.t] * (1 - CFG.TRANSACTION_COST)
            raw_ret           = (sell_price - self.entry_price) / self.entry_price
            reward           += self._vol_normalise(raw_ret)
            self.in_position  = False

        obs = self._get_obs()
        return obs, reward, done, info


# ─────────────────────────────────────────────
# 4. ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared trunk → Actor head (policy) + Critic head (value).
    Uses LayerNorm for stable training on financial features.
    """

    def __init__(self, obs_dim: int, n_actions: int = 3):
        super().__init__()

        # ── Shared feature extractor ──────────────────────────────────
        layers = []
        in_dim = obs_dim
        for _ in range(CFG.N_LAYERS):
            layers += [
                nn.Linear(in_dim, CFG.HIDDEN_DIM),
                nn.LayerNorm(CFG.HIDDEN_DIM),
                nn.Tanh(),
            ]
            in_dim = CFG.HIDDEN_DIM
        self.trunk = nn.Sequential(*layers)

        # ── Policy head ───────────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(CFG.HIDDEN_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

        # ── Value head ────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(CFG.HIDDEN_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        feat   = self.trunk(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(self, obs_np: np.ndarray):
        """Single-step inference → (action, log_prob, value)."""
        x   = torch.FloatTensor(obs_np).unsqueeze(0).to(CFG.DEVICE)
        with torch.no_grad():
            logits, value = self(x)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


# ─────────────────────────────────────────────
# 5. ROLLOUT BUFFER
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones  = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs);       self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value);  self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float):
        """GAE (Generalised Advantage Estimation)."""
        n         = len(self.rewards)
        advantages= np.zeros(n, dtype=np.float32)
        returns   = np.zeros(n, dtype=np.float32)
        last_gae  = 0.0

        for t in reversed(range(n)):
            next_val   = last_value if t == n - 1 else self.values[t + 1]
            next_done = float(self.dones[t])
            if t == n - 1:
                next_done = 0.0
            delta      = (self.rewards[t]
                          + CFG.GAMMA * next_val * (1 - next_done)
                          - self.values[t])
            last_gae   = delta + CFG.GAMMA * CFG.GAE_LAMBDA * (1 - next_done) * last_gae
            advantages[t] = last_gae
            returns[t]    = advantages[t] + self.values[t]

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t      = torch.FloatTensor(np.array(self.obs)).to(CFG.DEVICE)
        act_t      = torch.LongTensor(self.actions).to(CFG.DEVICE)
        lp_t       = torch.FloatTensor(self.log_probs).to(CFG.DEVICE)
        ret_t      = torch.FloatTensor(returns).to(CFG.DEVICE)
        adv_t      = torch.FloatTensor(advantages).to(CFG.DEVICE)
        return obs_t, act_t, lp_t, ret_t, adv_t


# ─────────────────────────────────────────────
# 6. PPO UPDATE
# ─────────────────────────────────────────────
def ppo_update(model, optimizer, buffer, last_value, scheduler=None):
    obs, acts, old_lps, returns, advantages = buffer.compute_returns_and_advantages(last_value)

    total_loss_log = []
    n = len(obs)

    for _ in range(CFG.N_EPOCHS):
        idx = torch.randperm(n)
        for start in range(0, n, CFG.MINI_BATCH):
            mb_idx  = idx[start: start + CFG.MINI_BATCH]
            mb_obs  = obs[mb_idx];     mb_acts = acts[mb_idx]
            mb_old  = old_lps[mb_idx]; mb_ret  = returns[mb_idx]
            mb_adv  = advantages[mb_idx]

            logits, values = model(mb_obs)
            temperature = 1.2
            dist = Categorical(logits=logits / temperature)
            new_lp   = dist.log_prob(mb_acts)
            entropy  = dist.entropy().mean()

            # ── PPO clip loss ─────────────────────────────────────────
            ratio     = (new_lp - mb_old).exp()
            clip_r    = ratio.clamp(1 - CFG.CLIP_EPS, 1 + CFG.CLIP_EPS)
            actor_loss= -torch.min(ratio * mb_adv, clip_r * mb_adv).mean()

            # ── Critic loss ───────────────────────────────────────────
            value_clipped = values + (values - mb_ret).clamp(-0.2, 0.2)
            critic_loss = torch.max(
            (values - mb_ret) ** 2,
            (value_clipped - mb_ret) ** 2).mean()
            
            # ── Total loss ────────────────────────────────────────────
            loss = actor_loss + CFG.VALUE_COEF * critic_loss - CFG.ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
            optimizer.step()
            total_loss_log.append(loss.item())

    if scheduler:
        scheduler.step()

    buffer.clear()
    return np.mean(total_loss_log)

best_reward = -np.inf
patience = 20
patience_counter = 0

# ─────────────────────────────────────────────
# 7. TRAINING LOOP (WITH EARLY STOPPING)
# ─────────────────────────────────────────────
def train(train_features, train_prices):
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    env   = StockTradingEnv(train_features, train_prices)
    model = ActorCritic(obs_dim=env.obs_dim).to(CFG.DEVICE)
    opt   = optim.Adam(model.parameters(), lr=CFG.LR, eps=1e-5)

    sched = None
    if CFG.LR_DECAY:
        sched = optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=0.1, total_iters=CFG.N_EPISODES
        )

    buffer        = RolloutBuffer()
    ep_rewards    = []
    ep_trades     = []
    losses        = []

    global_step   = 0

    # ✅ EARLY STOPPING INIT (INSIDE FUNCTION — IMPORTANT)
    best_reward = -np.inf
    patience = 20
    patience_counter = 0

    for episode in range(1, CFG.N_EPISODES + 1):
        obs       = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            action, log_prob, value = model.get_action(obs)
            next_obs, reward, done, _ = env.step(action)

            buffer.add(obs, action, log_prob, reward, value, done)

            obs        = next_obs
            ep_reward += reward
            global_step += 1

            # PPO update
            if global_step % CFG.ROLLOUT_STEPS == 0:
                _, _, last_v = model.get_action(obs)
                loss = ppo_update(model, opt, buffer, last_v, sched)
                losses.append(loss)

        # Flush remaining buffer
        if len(buffer.obs) > 0:
            _, _, last_v = model.get_action(obs)
            loss = ppo_update(model, opt, buffer, last_v)
            losses.append(loss)

        ep_rewards.append(ep_reward)
        ep_trades.append(env.trades)

        # Logging
        if episode % CFG.PRINT_EVERY == 0:
            avg_r = np.mean(ep_rewards[-CFG.PRINT_EVERY:])
            avg_t = np.mean(ep_trades[-CFG.PRINT_EVERY:])
            lr    = opt.param_groups[0]["lr"]

            print(f"[Ep {episode:4d}]  avg_reward={avg_r:+.4f}  "
                  f"avg_trades={avg_t:.1f}  lr={lr:.2e}")

        # ✅ EARLY STOPPING LOGIC
        if ep_reward > best_reward:
            best_reward = ep_reward
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print("\n✅ Training complete.")
    return model, ep_rewards, ep_trades, losses


# ─────────────────────────────────────────────
# 8. BACKTESTING (BUG FIXED)
# ─────────────────────────────────────────────
def backtest(model, features, prices, label="Test"):
    model.eval()
    env      = StockTradingEnv(features, prices)
    obs      = env.reset()
    done     = False
    total_r  = 0.0
    actions  = []
    port_vals= [CFG.INITIAL_CASH]

    cash      = CFG.INITIAL_CASH
    in_pos    = False
    entry_px  = 0.0

    t = 0
    while not done:
        x = torch.FloatTensor(obs).unsqueeze(0).to(CFG.DEVICE)
        with torch.no_grad():
            logits, _ = model(x)

        action = logits.argmax(dim=-1).item()
        actions.append(action)

        obs, reward, done, info = env.step(action)
        total_r += reward

        # ✅ FIXED POSITION LOGIC
        if "trade_return" in info:
            cash = cash * (1 + info["trade_return"])
            in_pos = False   # ← IMPORTANT FIX

        elif action == 1 and not in_pos:
            in_pos   = True
            entry_px = prices[t]

        curr_price = prices[min(t + 1, len(prices) - 1)]

        if in_pos:
            val = cash * (1 + (curr_price - entry_px) / entry_px)
        else:
            val = cash

        port_vals.append(val)
        t += 1

    port_vals = np.array(port_vals[:len(prices)])
    returns   = np.diff(port_vals) / port_vals[:-1]

    sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    dd      = (port_vals / np.maximum.accumulate(port_vals) - 1).min()
    total_r = (port_vals[-1] - CFG.INITIAL_CASH) / CFG.INITIAL_CASH

    buy_hold = (prices[-1] - prices[0]) / prices[0]

    print(f"\n── {label} Backtest Results ──────────────────")
    print(f"  Total Return   : {total_r:+.2%}")
    print(f"  Buy & Hold     : {buy_hold:+.2%}")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Max Drawdown   : {dd:.2%}")
    print(f"  Total Trades   : {env.trades}")
    print(f"  Action dist    : Hold={actions.count(0)} Buy={actions.count(1)} Sell={actions.count(2)}")

    return {
        "portfolio_values": port_vals,
        "total_return"    : total_r,
        "buy_hold"        : buy_hold,
        "sharpe"          : sharpe,
        "max_drawdown"    : dd,
        "trades"          : env.trades,
        "actions"         : actions,
        "prices"          : prices,
    }
# ─────────────────────────────────────────────
# 9. PLOTTING
# ─────────────────────────────────────────────
def plot_results(ep_rewards, losses, train_res, test_res):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PPO Stock Trading Agent — ITC Dataset", fontsize=16, fontweight="bold")

    # ── Training rewards ──
    ax = axes[0, 0]
    ax.plot(ep_rewards, alpha=0.4, color="steelblue", label="Episode reward")
    window = 20
    if len(ep_rewards) >= window:
        smooth = pd.Series(ep_rewards).rolling(window).mean()
        ax.plot(smooth, color="steelblue", linewidth=2, label=f"MA-{window}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Training Rewards"); ax.set_xlabel("Episode"); ax.legend()

    # ── PPO loss ──
    ax = axes[0, 1]
    ax.plot(losses, color="coral", linewidth=1)
    ax.set_title("PPO Loss"); ax.set_xlabel("Update step")

    # ── Train portfolio ──
    ax = axes[0, 2]
    ax.plot(train_res["portfolio_values"], color="green", linewidth=1.5, label="PPO Portfolio")
    bh_vals = CFG.INITIAL_CASH * (1 + (train_res["prices"] - train_res["prices"][0]) / train_res["prices"][0])
    ax.plot(bh_vals, color="gray", linestyle="--", linewidth=1, label="Buy & Hold")
    ax.set_title(f"Train Portfolio  (ret={train_res['total_return']:+.1%})")
    ax.set_xlabel("Step"); ax.legend()

    # ── Test portfolio ──
    ax = axes[1, 0]
    ax.plot(test_res["portfolio_values"], color="royalblue", linewidth=1.5, label="PPO Portfolio")
    bh_vals = CFG.INITIAL_CASH * (1 + (test_res["prices"] - test_res["prices"][0]) / test_res["prices"][0])
    ax.plot(bh_vals, color="gray", linestyle="--", linewidth=1, label="Buy & Hold")
    ax.set_title(f"Test Portfolio  (ret={test_res['total_return']:+.1%})")
    ax.set_xlabel("Step"); ax.legend()

    # ── Action distribution (test) ──
    ax = axes[1, 1]
    acts   = test_res["actions"]
    labels = ["Hold", "Buy", "Sell"]
    counts = [acts.count(i) for i in range(3)]
    bars   = ax.bar(labels, counts, color=["#aaa", "green", "red"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(c), ha="center")
    ax.set_title("Action Distribution (Test)")

    # ── Drawdown (test) ──
    ax = axes[1, 2]
    pv = test_res["portfolio_values"]
    dd = pv / np.maximum.accumulate(pv) - 1
    ax.fill_between(range(len(dd)), dd, 0, color="red", alpha=0.4, label="Drawdown")
    ax.set_title(f"Drawdown  (max={test_res['max_drawdown']:.1%})")
    ax.set_xlabel("Step"); ax.legend()

    plt.tight_layout()
    plt.savefig("ppo_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[Plot] Saved → ppo_results.png")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 58)
    print("   PPO Stock Trading Agent — ITC RL Features")
    print(f"   Device : {CFG.DEVICE}")
    print("=" * 58)

    # Load data
    # AFTER (Fixes Error)
    train_f, train_p, test_f, test_p, feat_cols = load_and_preprocess(CFG.DATA_PATH)
    print(f"\n[Train] Starting PPO training for {CFG.N_EPISODES} episodes …")
    model, ep_rewards, ep_trades, losses = train(train_f, train_p)

    # Save model weights
    torch.save(model.state_dict(), "ppo_stock_model.pt")
    print("[Model] Saved → ppo_stock_model.pt")

    # Backtest on both splits
    train_res = backtest(model, train_f, train_p, label="Train")
    test_res  = backtest(model, test_f,  test_p,  label="Test")

    # Plots
    plot_results(ep_rewards, losses, train_res, test_res)


if __name__ == "__main__":
    main()
