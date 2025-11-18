"""
Walk-Forward Validation Component
Handles time-series cross-validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, accuracy_score

from utils.ensemble_model import FinalEnsemble
from utils.signal_generator import SignalGenerator


class WalkForwardValidator:
    """Performs walk-forward validation for trading strategy"""
    
    def __init__(self, n_splits: int = 5, horizon: int = 30):
        self.n_splits = n_splits
        self.horizon = horizon
        self.results = {'fold_metrics': []}
    
    def validate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        Perform walk-forward validation
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with validation results
        """
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION ({self.n_splits} splits)")
        print(f"{'='*60}\n")
        
        # Prepare data
        required = feature_cols + ['target', 'forward_return', 'regime_trend', 
                                   'regime_direction', 'regime_volatility', 'regime_range',
                                   'atr', 'ema200', 'Close', 'adx', 'ema20_slope',
                                   'bb_width', 'ema_trend_width', 'atr_percentile',
                                   'volume_zscore']
        df_clean = df.dropna(subset=required)
        print(f"Clean data: {df_clean.shape}")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Walk-forward loop
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_clean)):
            print(f"\nFold {fold+1}/{self.n_splits}")
            print("-"*40)
            
            # Split data
            # gunakan hanya 40k–50k data terakhir (sekitar 6–10 bulan)
            window = 45000
            end = train_idx[-1] + 1           # FIX
            start = max(0, end - window)      # FIX
            train = df_clean.iloc[start:end]  # FIX


            test = df_clean.iloc[test_idx]
            print("TRAIN RANGE:", train.index.min(), "->", train.index.max())
            print("TEST RANGE :", test.index.min(), "->", test.index.max())
            
            for name in ['atr', 'adx', 'vol_score', 'Close']:
                print("TRAIN", name, ":", train[name].median(),
                    train[name].quantile([0.1,0.9]).to_dict())
                print("TEST ", name, ":", test[name].median(),
                    test[name].quantile([0.1,0.9]).to_dict())

            
            # Prepare training data
            X_train = train[feature_cols]
            y_train = train['target']
            trend_train = train['regime_trend']
            dir_train = train['regime_direction']
            vol_train = train['regime_volatility']
            range_train = train['regime_range']
            
            # Prepare test data
            X_test = test[feature_cols]
            y_test = test['target']
            trend_test = test['regime_trend']
            dir_test = test['regime_direction']
            vol_test = test['regime_volatility']
            range_test = test['regime_range']
            
            # Winsorize target
            low = y_train.quantile(0.01)
            high = y_train.quantile(0.99)
            y_train_clip = y_train.clip(lower=low, upper=high)
            
            # Scale target
            scaler_y = RobustScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_clip.values.reshape(-1,1)).ravel()
            
            # Scale features
            scaler_x = RobustScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)
            
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
            
            # Train ensemble
            ensemble = FinalEnsemble()
            
            # Extract ATR for volatility-aware model
            atr_train = train['atr'].values
            atr_test = test['atr'].values
            
            ensemble.fit(X_train_scaled, y_train_scaled, trend_train, dir_train, 
                        vol_train, range_train, atr_train)
            
            # Predict
            results = ensemble.predict(X_test_scaled, atr_test)
            pred_scaled, q25_scaled, q75_scaled, pred_trend, pred_dir, pred_dir_proba, \
            pred_vol, pred_range, pred_range_proba, pred_main, pred_vol_aware, w_main = results
            # Inverse transform predictions
            # --- build train slice correctly (include last index) ---
            end_idx = train_idx[-1] + 1
            start_idx = max(0, end_idx - window)
            train = df_clean.iloc[start_idx: end_idx]

            # ... after predictions and inverse transform:
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
            q25 = scaler_y.inverse_transform(q25_scaled.reshape(-1,1)).ravel()
            q75 = scaler_y.inverse_transform(q75_scaled.reshape(-1,1)).ravel()

            # calibrate quantiles using inverse-space pred mean
            q_shift = y_train.mean() - np.nanmean(pred)
            q25 = q25 + q_shift
            q75 = q75 + q_shift
            print("pred mean/std:", np.nanmean(pred), np.nanstd(pred))
            print("target mean/std:", np.nanmean(y_test.values), np.nanstd(y_test.values))
            print("iqr median:", np.nanmedian(q75-q25))
            
            # === VOLATILITY-AWARE MODEL DIAGNOSTICS ⭐ ===
            print(f"\nModel Ensemble Analysis:")
            print(f"  Main model weight: {np.mean(w_main):.2f} (avg)")
            print(f"  Vol-aware weight: {np.mean(1-w_main):.2f} (avg)")
            print(f"  Main pred std: {np.nanstd(pred_main):.6f}")
            print(f"  Vol-aware pred std: {np.nanstd(pred_vol_aware):.6f}")
            print(f"  Final ensemble std: {np.nanstd(pred):.6f}")
            
            # Check correlation between predictions
            corr = np.corrcoef(pred_main, pred_vol_aware)[0,1]
            print(f"  Main vs Vol-aware correlation: {corr:.3f}")
            
            # Calculate metrics
            coverage = ((y_test.values >= q25) & (y_test.values <= q75)).sum() / len(y_test)
            trend_acc = accuracy_score(trend_test, pred_trend)
            dir_acc = accuracy_score(dir_test, pred_dir)
            vol_acc = accuracy_score(vol_test, pred_vol)
            range_acc = accuracy_score(range_test, pred_range)
            
            print(f"  Coverage: {coverage:.1%}")
            print(f"  Trend: {trend_acc:.1%}, Dir: {dir_acc:.1%}, Vol: {vol_acc:.1%}, Range: {range_acc:.1%} ⭐")
            
            # Generate signals
            signal_gen = SignalGenerator()
            signals, sizes = signal_gen.generate_signals(
                pred, q25, q75, pred_trend, pred_dir, pred_dir_proba, pred_vol,
                pred_range, pred_range_proba,
                test['Close'].values,
                test['ema200'].values,
                test['atr'].values,
                test['adx'].values,
                test['ema20_slope'].values,
                test['bb_width'].values,
                test['ema_trend_width'].values,
                test['atr_percentile'].values,
                test['volume_zscore'].values,
                test.index,
                train['atr'].values,
            )
            
            # Calculate trading metrics
            returns = test['forward_return'].values
            trade_returns = signals * returns * sizes
            
            r2 = r2_score(y_test, pred)
            n_trades = np.sum(signals != 0)
            
            if n_trades > 0:
                win_rate = np.sum(trade_returns > 0) / n_trades
                sharpe = trade_returns.mean() / (trade_returns.std() + 1e-10) * np.sqrt(96*250)
                cumul = (1 + pd.Series(trade_returns)).cumprod()
                max_dd = ((cumul - cumul.expanding().max()) / cumul.expanding().max()).min()
            else:
                win_rate = sharpe = max_dd = 0
            
            # Store results
            self.results['fold_metrics'].append({
                'fold': fold+1,
                'r2': r2,
                'coverage': coverage,
                'trend_acc': trend_acc,
                'dir_acc': dir_acc,
                'vol_acc': vol_acc,
                'range_acc': range_acc,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'max_dd': max_dd
            })
            
            print(f"  R²={r2:.4f}, Trades={n_trades}, WR={win_rate:.1%}, Sharpe={sharpe:.2f}")
        
    
    def print_summary(self):
        """Print summary of all folds"""
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        m = self.results['fold_metrics']
        print(f"R²: {np.mean([x['r2'] for x in m]):.4f}")
        print(f"Coverage: {np.mean([x['coverage'] for x in m]):.1%}")
        print(f"Trend Acc: {np.mean([x['trend_acc'] for x in m]):.1%}")
        print(f"Dir Acc: {np.mean([x['dir_acc'] for x in m]):.1%}")
        print(f"Vol Acc: {np.mean([x['vol_acc'] for x in m]):.1%}")
        print(f"Range Acc: {np.mean([x['range_acc'] for x in m]):.1%} ⭐")
        print(f"Win Rate: {np.mean([x['win_rate'] for x in m]):.1%}")
        print(f"Sharpe: {np.mean([x['sharpe'] for x in m]):.2f}")
        
        print(f"\n{'='*60}")
        print("ALL FOLDS PERFORMANCE:")
        for m_fold in m:
            print(f"  Fold {m_fold['fold']}: WR={m_fold['win_rate']:.1%}, "
                  f"Sharpe={m_fold['sharpe']:.2f}, Trades={m_fold['n_trades']}")
            
        return self.results