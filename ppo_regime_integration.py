import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from regime_detector import RegimeDetector
from regime_features import extract_features, scale_features, compute_trend_slope
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Normalization Utilities
# ==========================================

def rolling_normalize(df, cols, window=50):
    out = df.copy()
    for col in cols:
        mu = out[col].rolling(window, min_periods=1).mean()
        sigma = out[col].rolling(window, min_periods=1).std()
        out[col] = (out[col] - mu) / (sigma + 1e-6)
    return out.fillna(0)

# ==========================================
# Regime Generation
# ==========================================

def generate_regimes(df, train_split=0.75):
    print("[Regime] Generating regimes...")
    split_idx = int(len(df) * train_split)
    features = extract_features(df)
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    detector = RegimeDetector(n_states=3)
    detector.fit(train_scaled)
    train_probs = detector.predict_probabilities(train_scaled)
    test_probs = detector.predict_probabilities(test_scaled)
    all_probs = np.vstack([train_probs, test_probs])
    regime_cols = [f"p_regime_{i}" for i in range(all_probs.shape[1])]
    prob_df = pd.DataFrame(all_probs, columns=regime_cols, index=df.index)
    return prob_df, regime_cols

# ==========================================
# Trading Environment (PROFIT OPTIMIZED)
# ==========================================

class TradingEnv(gym.Env):
    def __init__(self, df, feature_cols, transaction_cost=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.transaction_cost = transaction_cost
        self.max_steps = len(self.df) - 1
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs: Features + Current Position + Unrealized PnL
        obs_dim = len(feature_cols) + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = 1.0
        self.entry_price = 0.0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        features = self.df.loc[self.current_step, self.feature_cols].values
        
        # Calculate unrealized pnl for the state
        current_price = self.df.loc[self.current_step, "close"]
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - current_price) / self.entry_price
            
        obs = np.append(features, [self.position, unrealized_pnl])
        return np.nan_to_num(obs.astype(np.float32))

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"portfolio_value": self.portfolio_value}
        
        target_position = np.clip(float(action[0]), -1.0, 1.0)
        current_price = self.df.loc[self.current_step, "close"]
        step_return = self.df.loc[self.current_step, "return_1"]
        
        # 1. Transaction Cost
        trade_penalty = abs(target_position - self.position) * self.transaction_cost
        
        # 2. Update Entry Price for new positions
        if abs(target_position) > 0.01 and abs(self.position) < 0.01:
            self.entry_price = current_price
        elif abs(target_position) < 0.01:
            self.entry_price = 0.0
            
        # 3. Calculate Raw P&L
        raw_pnl = (target_position * step_return) - trade_penalty
        
        # 4. Asymmetric Reward (Only penalize downside volatility)
        downside_penalty = 5.0 * (min(0, raw_pnl) ** 2) 
        
        # 5. Profit Bonus: Reward staying in a winning trade
        profit_bonus = 0.0
        if raw_pnl > 0:
            profit_bonus = raw_pnl * 0.1 # 10% bonus on positive moves to encourage greed
            
        reward = (raw_pnl + profit_bonus - downside_penalty) * 200.0 # High scale for aggressive learning
        
        # Update State
        self.portfolio_value *= (1 + raw_pnl)
        self.position = target_position
        self.current_step += 1
        
        if self.portfolio_value <= 0.2 or self.current_step >= self.max_steps:
            self.done = True
            if self.portfolio_value <= 0.2:
                reward -= 100.0 # Severe penalty for ruin
                
        return self._get_obs(), reward, self.done, False, {"portfolio_value": self.portfolio_value}

# ==========================================
# Metrics & Evaluation
# ==========================================

def calculate_metrics(returns):
    returns = np.array(returns)
    if len(returns) == 0: return {"Sharpe": 0, "MaxDD": 0, "CumRet": 0}
    volatility = np.std(returns)
    sharpe = (np.sqrt(252) * np.mean(returns) / volatility) if volatility > 0 else 0
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1e-9)
    return {"Sharpe": sharpe, "MaxDD": np.min(drawdown), "CumRet": cumulative_returns[-1] - 1}

def evaluate_agent(env, model):
    obs, _ = env.reset()
    done = False
    rets, vals = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        last_val = vals[-1] if len(vals) > 0 else 1.0
        current_val = info.get("portfolio_value", 0.0)
        step_ret = (current_val - last_val) / last_val if last_val > 0 else -1.0
        rets.append(step_ret)
        vals.append(current_val)
        if current_val <= 0.2: break
    return vals, calculate_metrics(rets)

# ==========================================
# Main
# ==========================================

def run_experiment(total_timesteps=100000, train_split=0.75):
    print("--- Loading Data ---")
    df = pd.read_csv("Data/itc/itc_rl_features.csv")
    df["trend_slope"] = compute_trend_slope(df["close"].values)
    df = df.ffill().bfill().fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df["return_1"] = df["close"].pct_change().shift(-1).fillna(0)
    
    # ENRICHED FEATURES
    BASE_FEATURES = [
        "log_return_1", "volatility_10", "volatility_20", 
        "close_ma10_ratio", "close_ma20_ratio", "close_ma5_ratio",
        "momentum_5", "momentum_10", "trend_slope"
    ]
    
    regime_df, REGIME_FEATURES = generate_regimes(df, train_split=train_split)
    full_df = df.join(regime_df).ffill().bfill().fillna(0)
    split_idx = int(len(full_df) * train_split)
    
    train_df = rolling_normalize(full_df.iloc[:split_idx], BASE_FEATURES)
    test_df = rolling_normalize(full_df.iloc[split_idx:], BASE_FEATURES)
    
    # Aggressive PPO Config
    ppo_config = {
        "learning_rate": 5e-5,   # Even slower learning for precision
        "n_steps": 8192,         # Double the experience per update
        "batch_size": 512,       # Much larger batch to smooth noise
        "n_epochs": 15,          # More intense optimization
        "gamma": 0.98,           # Focus on long-term wealth
        "ent_coef": 0.02,        # Higher entropy for better exploration of profit peaks
        "device": "cpu"
    }
    
    print(f"--- Training Baseline ({total_timesteps} steps) ---")
    env_b = TradingEnv(train_df, BASE_FEATURES)
    model_b = PPO("MlpPolicy", env_b, verbose=0, **ppo_config)
    model_b.learn(total_timesteps)
    
    print(f"--- Training Regime-Aware ({total_timesteps} steps) ---")
    env_r = TradingEnv(train_df, BASE_FEATURES + REGIME_FEATURES)
    model_r = PPO("MlpPolicy", env_r, verbose=0, **ppo_config)
    model_r.learn(total_timesteps)
    
    print("\n--- TEST SET RESULTS ---")
    test_env_b = TradingEnv(test_df, BASE_FEATURES)
    test_env_r = TradingEnv(test_df, BASE_FEATURES + REGIME_FEATURES)
    
    vals_b, met_b = evaluate_agent(test_env_b, model_b)
    vals_r, met_r = evaluate_agent(test_env_r, model_r)
    
    print(f"Baseline:    Sharpe={met_b['Sharpe']:.4f}, MaxDD={met_b['MaxDD']:.4f}, CumRet={met_b['CumRet']:.4f}")
    print(f"Regime-Aware: Sharpe={met_r['Sharpe']:.4f}, MaxDD={met_r['MaxDD']:.4f}, CumRet={met_r['CumRet']:.4f}")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(vals_b, label=f"Baseline (Ret: {met_b['CumRet']:.1%})")
    plt.plot(vals_r, label=f"Regime-Aware (Ret: {met_r['CumRet']:.1%})")
    plt.title("Portfolio Growth: Baseline vs Regime-Aware")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_profit_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_experiment()
