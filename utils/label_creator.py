"""
Label Creator V2 - IMPROVED
============================
Key improvements:
1. Volatility-adjusted target WITHOUT leakage
2. Multi-horizon predictions
3. Better regime classification
4. Risk-adjusted returns
"""

import numpy as np
import pandas as pd


class LabelCreator:
    """Enhanced label creator with proper volatility adjustment"""
    
    def __init__(self, horizons: list = [15, 30, 60], atr_period: int = 14):
        self.horizons = horizons
        self.atr_period = atr_period
        self.primary_horizon = horizons[1] if len(horizons) > 1 else horizons[0]
    
    def create_all_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ALL labels with improvements:
        1. Multi-horizon targets
        2. Volatility-adjusted targets (NO LEAKAGE)
        3. Enhanced regime classification
        """
        df = df.copy()
        returns = df['Close'].pct_change()
        
        # Mid-price
        df['mid'] = (df['High'] + df['Low']) / 2
        
        # ATR calculation
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(self.atr_period).mean()
        
        # === MULTI-HORIZON TARGETS ===
        for horizon in self.horizons:
            df = self._create_target_for_horizon(df, horizon)
        
        # Primary target for main model
        df['target'] = df[f'target_{self.primary_horizon}']
        df['forward_return'] = df[f'forward_return_{self.primary_horizon}']
        
        # === VOLATILITY-ADJUSTED TARGET (NO LEAKAGE!) ===
        # Use SHIFTED ATR to ensure no future information
        atr_pct_shifted = (df['atr'].shift(1) / df['Close'].shift(1))
        df['target_vol_adj'] = df['target'] / (atr_pct_shifted + 1e-10)
        
        # Clip extreme values for stability
        df['target_vol_adj'] = df['target_vol_adj'].clip(-10, 10)
        
        # === REGIME DETECTION FEATURES ===
        df = self._calculate_technical_indicators(df)
        
        # === REGIME LABELS ===
        df = self._create_enhanced_regimes(df)
        
        self._print_regime_stats(df)
        
        return df
    
    def _create_target_for_horizon(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Create target for specific horizon"""
        df[f'future_mid_{horizon}'] = df['mid'].shift(-horizon)
        df[f'target_{horizon}'] = np.log(df[f'future_mid_{horizon}'] / df['mid'])
        df[f'forward_return_{horizon}'] = (df[f'future_mid_{horizon}'] - df['mid']) / df['mid']
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for regime detection"""
        
        # EMAs
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # EMA slopes
        df['ema20_slope'] = df['ema20'].pct_change(5)
        df['ema50_slope'] = df['ema50'].pct_change(10)
        df['ema200_slope'] = df['ema200'].pct_change(20)
        
        # ATR ratio
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # Volume
        vol_ma = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / (vol_ma + 1e-10)
        
        # ADX
        df = self._calculate_adx(df)
        
        # Volatility score
        returns_vol = df['Close'].pct_change().rolling(20).std()
        atr_norm = df['atr'] / df['Close']
        vol_score = (atr_norm / atr_norm.rolling(100).mean() + 
                     returns_vol / returns_vol.rolling(100).mean()) / 2
        df['vol_score'] = vol_score
        
        # ===  NEW: TREND STRENGTH INDICATOR ===
        df['trend_strength'] = np.abs(df['ema20_slope']) * df['adx'] / 50.0
        
        # === NEW: PRICE MOMENTUM ===
        df['price_momentum'] = df['Close'].pct_change(10)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX indicator"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr_calc = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = pd.Series(np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        ), index=df.index)
        
        minus_dm = pd.Series(np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        ), index=df.index)
        
        atr_14 = tr_calc.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        return df
    
    def _create_enhanced_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced regime classification"""
        
        # === A. ENHANCED TREND CLASSIFIER ===
        strong_adx = df['adx'] > 25
        strong_slope = np.abs(df['ema20_slope']) > 0.0005
        aligned_emas = ((df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])) | \
                       ((df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200']))
        
        df['regime_trend'] = 0  # NOT-TREND (default)
        df.loc[strong_adx & strong_slope, 'regime_trend'] = 1  # TREND
        df.loc[strong_adx & strong_slope & aligned_emas, 'regime_trend'] = 2  # STRONG-TREND
        
        # Simplify to binary for compatibility
        df['regime_trend'] = (df['regime_trend'] > 0).astype(int)
        
        # === B. DIRECTION CLASSIFIER ===
        df['regime_direction'] = 0  # DOWN (default)
        df.loc[df['ema20_slope'] > 0, 'regime_direction'] = 1  # UP
        
        # === C. ENHANCED VOLATILITY REGIME ===
        vol_20 = df['vol_score'].quantile(0.25)
        vol_60 = df['vol_score'].quantile(0.60)
        
        df['regime_volatility'] = 1  # MEDIUM (default)
        df.loc[df['vol_score'] < vol_20, 'regime_volatility'] = 0  # LOW
        df.loc[df['vol_score'] > vol_60, 'regime_volatility'] = 2  # HIGH
        
        # === D. ENHANCED RANGE CLASSIFIER ===
        low_adx = df['adx'] < 18
        flat_slope = np.abs(df['ema20_slope']) < 0.0002
        low_trend_strength = df['trend_strength'] < 0.05
        
        df['regime_range'] = 0  # NOT-RANGE (default)
        df.loc[low_adx & flat_slope & low_trend_strength, 'regime_range'] = 1  # RANGE
        
        return df
    
    def _print_regime_stats(self, df: pd.DataFrame):
        """Print statistics for all regime labels"""
        print(f"\n{'='*60}")
        print(f"LABELS CREATED V2 (primary horizon={self.primary_horizon} bars)")
        print(f"{'='*60}")
        
        for horizon in self.horizons:
            target_col = f'target_{horizon}'
            print(f"\nHorizon {horizon} ({horizon*15/60:.1f}h):")
            print(f"  Mean: {df[target_col].mean():.6f}, Std: {df[target_col].std():.6f}")
        
        print(f"\nVolatility-Adjusted Target:")
        print(f"  Mean: {df['target_vol_adj'].mean():.6f}")
        print(f"  Std: {df['target_vol_adj'].std():.6f}")
        print(f"  Min: {df['target_vol_adj'].min():.6f}")
        print(f"  Max: {df['target_vol_adj'].max():.6f}")
        
        print(f"\nRegime A (TREND):")
        print(f"  TREND: {(df['regime_trend']==1).sum()} ({(df['regime_trend']==1).sum()/len(df):.1%})")
        print(f"  NOT-TREND: {(df['regime_trend']==0).sum()} ({(df['regime_trend']==0).sum()/len(df):.1%})")
        
        print(f"\nRegime B (DIRECTION):")
        print(f"  UP: {(df['regime_direction']==1).sum()} ({(df['regime_direction']==1).sum()/len(df):.1%})")
        print(f"  DOWN: {(df['regime_direction']==0).sum()} ({(df['regime_direction']==0).sum()/len(df):.1%})")
        
        print(f"\nRegime C (VOLATILITY):")
        print(f"  LOW: {(df['regime_volatility']==0).sum()} ({(df['regime_volatility']==0).sum()/len(df):.1%})")
        print(f"  MEDIUM: {(df['regime_volatility']==1).sum()} ({(df['regime_volatility']==1).sum()/len(df):.1%})")
        print(f"  HIGH: {(df['regime_volatility']==2).sum()} ({(df['regime_volatility']==2).sum()/len(df):.1%})")
        
        print(f"\nRegime D (RANGE): ⭐ TOXIC FILTER!")
        print(f"  NOT-RANGE: {(df['regime_range']==0).sum()} ({(df['regime_range']==0).sum()/len(df):.1%})")
        print(f"  RANGE (TOXIC!): {(df['regime_range']==1).sum()} ({(df['regime_range']==1).sum()/len(df):.1%})")
        
        optimal = ((df['regime_trend']==1) & (df['regime_volatility']==1) & (df['regime_range']==0)).sum()
        toxic = ((df['regime_range']==1)).sum()
        print(f"\n✅ Optimal (TREND + MED VOL + NOT-RANGE): {optimal} ({optimal/len(df):.1%})")
        print(f"❌ Toxic (RANGE): {toxic} ({toxic/len(df):.1%}) - AVOID!")