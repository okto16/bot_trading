"""
Feature Engineer V2 - ENHANCED
================================
Improvements:
1. +25 new high-quality features
2. Price action patterns
3. Multi-timeframe features
4. Interaction features
5. Market microstructure proxies
6. Better feature selection
"""

import numpy as np
import pandas as pd
from scipy import stats


class FeatureEngineer:
    """Enhanced feature engineering with 60+ features"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 60+ predictive features"""
        df = df.copy()
        returns = df['Close'].pct_change()
        
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING V2")
        print(f"{'='*60}")
        
        # === CORE FEATURES (39 from v1) ===
        df = self._create_trend_features(df)
        df = self._create_momentum_features(df, returns)
        df = self._create_volatility_features(df, returns)
        df = self._create_zscore_features(df)
        df = self._create_trend_strength_features(df)
        df = self._create_time_features(df)
        df = self._create_regime_features(df)
        
        # === NEW: PRICE ACTION PATTERNS (8 features) ===
        df = self._create_price_action_patterns(df)
        
        # === NEW: MULTI-TIMEFRAME FEATURES (6 features) ===
        df = self._create_multitimeframe_features(df)
        
        # === NEW: INTERACTION FEATURES (5 features) ===
        df = self._create_interaction_features(df)
        
        # === NEW: MARKET MICROSTRUCTURE (4 features) ===
        df = self._create_microstructure_features(df)
        
        # === NEW: VOLATILITY REGIME INDICATORS (3 features) ===
        df = self._create_volatility_regime_features(df)
        
        print(f"\n✅ Total features created: {self._count_shifted_features(df)}")
        
        return df
    
    def _count_shifted_features(self, df: pd.DataFrame) -> int:
        """Count features that end with _f or are shifted"""
        count = 0
        for col in df.columns:
            if col.endswith('_f') or col.endswith('_shifted'):
                count += 1
        return count
    
    # === ORIGINAL FEATURES (from v1) ===
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-based features"""
        df['ema20_f'] = df['Close'].ewm(span=20, adjust=False).mean().shift(1)
        df['ema50_f'] = df['Close'].ewm(span=50, adjust=False).mean().shift(1)
        df['ema100_f'] = df['Close'].ewm(span=100, adjust=False).mean().shift(1)
        df['ema200_dist'] = ((df['Close'] - df['ema200']) / df['ema200']).shift(1)
        df['ema20_50_cross'] = ((df['ema20_f'] - df['ema50_f']) / df['ema50_f']).shift(1)
        df['ema50_200_cross'] = ((df['ema50_f'] - df['ema200']) / df['ema200']).shift(1)
        df['price_vs_ema20'] = ((df['Close'] - df['ema20_f']) / df['ema20_f']).shift(1)
        df['price_vs_ema50'] = ((df['Close'] - df['ema50_f']) / df['ema50_f']).shift(1)
        df['ema_trend_width'] = (np.abs(df['ema20_f'] - df['ema50_f']) / df['Close']).shift(1)
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Momentum-based features"""
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ((ema12 - ema26) / df['Close']).shift(1)
        df['roc5'] = df['Close'].pct_change(5).shift(1)
        df['roc10'] = df['Close'].pct_change(10).shift(1)
        df['roc20'] = df['Close'].pct_change(20).shift(1)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
        df['rsi_zscore'] = ((df['rsi'] - df['rsi'].rolling(50).mean()) / 
                            (df['rsi'].rolling(50).std() + 1e-10)).shift(1)
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Volatility-based features"""
        df['atr_pct'] = (df['atr'] / df['Close']).shift(1)
        df['atr_percentile'] = (df['atr'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)).shift(1)
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()
        df['vol_std_20'] = vol_20.shift(1)
        df['vol_std_50'] = vol_50.shift(1)
        df['vol_ratio'] = (vol_20 / (vol_50 + 1e-10)).shift(1)
        df['vol_percentile'] = (vol_20.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)).shift(1)
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['bb_position'] = ((df['Close'] - (sma20 - std20*2)) / (std20*4)).shift(1)
        df['bb_width'] = ((std20 * 4) / (sma20 + 1e-10)).shift(1)
        df['tr_percentile'] = (df['tr'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)).shift(1)
        return df
    
    def _create_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization features"""
        rolling_mean = df['Close'].rolling(50).mean()
        rolling_std = df['Close'].rolling(50).std()
        df['price_zscore_50'] = ((df['Close'] - rolling_mean) / (rolling_std + 1e-10)).shift(1)
        rolling_mean_100 = df['Close'].rolling(100).mean()
        rolling_std_100 = df['Close'].rolling(100).std()
        df['price_zscore_100'] = ((df['Close'] - rolling_mean_100) / (rolling_std_100 + 1e-10)).shift(1)
        vol_mean = df['Volume'].rolling(50).mean()
        vol_std = df['Volume'].rolling(50).std()
        df['volume_zscore'] = ((df['Volume'] - vol_mean) / (vol_std + 1e-10)).shift(1)
        atr_mean = df['atr'].rolling(50).mean()
        atr_std = df['atr'].rolling(50).std()
        df['atr_zscore'] = ((df['atr'] - atr_mean) / (atr_std + 1e-10)).shift(1)
        return df
    
    def _create_trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend strength features"""
        price_change = np.abs(df['Close'] - df['Close'].shift(14))
        high_low_range = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        df['trend_strength'] = (price_change / (high_low_range + 1e-10)).shift(1)
        df['directional_strength'] = (np.abs(df['ema20_slope']) / 
                                    (df['atr_ratio'] + 1e-10)).shift(1)
        df['momentum_strength'] = (np.abs(df['roc10']) / 
                                (df['vol_std_20'] + 1e-10)).shift(1)
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime features for classification"""
        df['adx_f'] = df['adx'].shift(1)
        df['ema20_slope_f'] = df['ema20_slope'].shift(1)
        df['atr_ratio_f'] = df['atr_ratio'].shift(1)
        df['vol_ratio_f'] = df['vol_ratio'].shift(1)
        df['vol_score_f'] = df['vol_score'].shift(1)
        return df
    
    # === NEW FEATURES ===
    
    def _create_price_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Price action pattern recognition (8 features)
        Captures candle patterns and price structures
        """
        # Body and wick ratios
        df['body_size'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['upper_wick'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / (df['High'] - df['Low'] + 1e-10)
        df['lower_wick'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Pin bar detection (long wick, small body)
        df['pin_bar_bull'] = ((df['lower_wick'] > 0.6) & (df['body_size'] < 0.3)).astype(int)
        df['pin_bar_bear'] = ((df['upper_wick'] > 0.6) & (df['body_size'] < 0.3)).astype(int)
        
        # Engulfing patterns
        bullish_engulf = (df['Close'] > df['Open']) & \
                        (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['Close'] > df['Open'].shift(1)) & \
                        (df['Open'] < df['Close'].shift(1))
        df['engulfing_bull'] = bullish_engulf.astype(int)
        
        bearish_engulf = (df['Close'] < df['Open']) & \
                        (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['Close'] < df['Open'].shift(1)) & \
                        (df['Open'] > df['Close'].shift(1))
        df['engulfing_bear'] = bearish_engulf.astype(int)
        
        # Price range expansion/contraction
        df['range_expansion'] = (df['High'] - df['Low']) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-10)
        
        # Shift all price action features
        for col in ['body_size', 'upper_wick', 'lower_wick', 'pin_bar_bull', 
                    'pin_bar_bear', 'engulfing_bull', 'engulfing_bear', 'range_expansion']:
            df[f'{col}_f'] = df[col].shift(1)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _create_multitimeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Multi-timeframe aggregations (6 features)
        Higher timeframe trend and momentum
        """
        # M30 (2x M15) aggregations
        df['close_m30'] = df['Close'].rolling(2).mean()
        df['atr_m30'] = df['atr'].rolling(2).mean()
        df['vol_m30'] = df['vol_std_20'].rolling(2).mean()
        
        # H1 (4x M15) aggregations
        df['close_h1'] = df['Close'].rolling(4).mean()
        df['trend_h1'] = (df['close_h1'] - df['close_h1'].shift(4)) / df['close_h1'].shift(4)
        df['vol_h1'] = df['vol_std_20'].rolling(4).mean()
        
        # Shift all
        for col in ['close_m30', 'atr_m30', 'vol_m30', 'close_h1', 'trend_h1', 'vol_h1']:
            df[f'{col}_f'] = df[col].shift(1)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Interaction features (5 features)
        Combinations of key indicators
        """
        # Trend × Volatility
        df['trend_vol_interaction'] = df['ema20_slope'] * df['atr_ratio']
        
        # Momentum × ADX
        df['momentum_strength_interaction'] = df['roc10'] * df['adx'] / 50.0
        
        # RSI × Volatility
        df['rsi_vol_interaction'] = (df['rsi'] - 50) * df['vol_std_20']
        
        # Price position × Trend strength
        df['position_trend_interaction'] = df['bb_position'] * np.abs(df['ema20_slope'])
        
        # Volume × Price momentum
        df['volume_momentum_interaction'] = df['vol_ratio'] * df['roc10']
        
        # Shift all
        for col in ['trend_vol_interaction', 'momentum_strength_interaction', 
                    'rsi_vol_interaction', 'position_trend_interaction', 
                    'volume_momentum_interaction']:
            df[f'{col}_f'] = df[col].shift(1)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Market microstructure proxies (4 features)
        Order flow and liquidity indicators
        """
        # Buy/sell pressure proxy
        df['buy_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['sell_pressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
        
        # Cumulative delta (proxy for order flow)
        df['order_flow_proxy'] = (df['buy_pressure'] - df['sell_pressure']).rolling(10).sum()
        
        # Volume-weighted price position
        df['vwap_distance'] = (df['Close'] - df['mid']) / (df['atr'] + 1e-10)
        
        # Shift all
        for col in ['buy_pressure', 'sell_pressure', 'order_flow_proxy', 'vwap_distance']:
            df[f'{col}_f'] = df[col].shift(1)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _create_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Volatility regime strength indicators (3 features)
        """
        # Volatility expansion/contraction
        df['vol_expansion'] = df['atr'] / df['atr'].rolling(20).mean()
        
        # Volatility trend
        df['vol_trend'] = (df['atr'] - df['atr'].shift(10)) / df['atr'].shift(10)
        
        # Historical volatility percentile
        df['historical_vol_pct'] = df['vol_std_20'].rolling(200).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Shift all
        for col in ['vol_expansion', 'vol_trend', 'historical_vol_pct']:
            df[f'{col}_f'] = df[col].shift(1)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def select_top_features(self, df: pd.DataFrame, feature_importance: pd.DataFrame, 
                           top_n: int = 40) -> list:
        """
        Feature selection based on importance
        Keep only top N features
        """
        if feature_importance is None or len(feature_importance) == 0:
            # Return all feature columns if no importance data
            exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'mid', 'tr', 'atr',
                      'future_mid', 'forward_return', 'target', 'target_vol_adj',
                      'hour', 'ema200', 'regime_trend', 'regime_direction', 
                      'regime_volatility', 'regime_range', 'ema20_slope', 'ema50_slope',
                      'ema200_slope', 'atr_ratio', 'vol_ratio', 'adx', 'vol_score',
                      'ema20', 'ema50', 'trend_strength', 'price_momentum']
            # Add multi-horizon targets
            for col in df.columns:
                if col.startswith('target_') or col.startswith('future_mid_') or col.startswith('forward_return_'):
                    exclude.append(col)
            
            return [c for c in df.columns if c not in exclude]
        
        # Get top N features
        top_features = feature_importance.nlargest(top_n, 'importance')['feature'].tolist()
        
        print(f"\n✅ Selected top {len(top_features)} features:")
        for i, feat in enumerate(top_features[:10], 1):
            imp = feature_importance[feature_importance['feature']==feat]['importance'].values[0]
            print(f"  {i}. {feat}: {imp:.4f}")
        
        return top_features