"""
Production Model Class
======================
Shared class for model training and deployment
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from utils.ensemble_model import FinalEnsemble


class ProductionModel:
    """
    Wrapper for production deployment.
    Contains trained ensemble + all necessary preprocessing.
    """
    
    def __init__(self):
        self.ensemble = FinalEnsemble()
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()
        
        # Metadata
        self.feature_cols = None
        self.train_atr_median = None
        self.train_target_mean = None
        self.train_target_std = None
        self.horizon = None
        self.version = "7-MODEL-ENSEMBLE-v1.0"
        self.creation_date = None
        self.training_data_range = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            regime_trend: pd.Series, regime_direction: pd.Series,
            regime_volatility: pd.Series, regime_range: pd.Series,
            atr_train: np.ndarray, horizon: int):
        """Train the model"""
        
        self.feature_cols = list(X_train.columns)
        self.horizon = horizon
        
        # Winsorize target
        low = y_train.quantile(0.01)
        high = y_train.quantile(0.99)
        y_train_clipped = y_train.clip(lower=low, upper=high)
        
        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(
            y_train_clipped.values.reshape(-1, 1)
        ).ravel()
        
        # Store stats
        self.train_target_mean = y_train.mean()
        self.train_target_std = y_train.std()
        
        # Scale features
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=self.feature_cols, index=X_train.index
        )
        
        # Train ensemble
        print("\n⏳ Training 7-model ensemble...")
        self.ensemble.fit(
            X_train_scaled, y_train_scaled,
            regime_trend, regime_direction,
            regime_volatility, regime_range,
            atr_train
        )
        
        print("✅ Training complete!")
        return self
    
    def predict(self, X_test: pd.DataFrame, atr_test: np.ndarray):
        """Make predictions"""
        
        # Scale features
        X_test_scaled = self.scaler_x.transform(X_test)
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=self.feature_cols, index=X_test.index
        )
        
        # Predict
        results = self.ensemble.predict(X_test_scaled, atr_test)
        (pred_scaled, q25_scaled, q75_scaled, pred_trend, pred_direction,
         pred_direction_proba, pred_volatility, pred_range, pred_range_proba,
         pred_main, pred_vol_aware, w_main) = results
        
        # Inverse transform
        pred_point = self.scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()
        
        q25 = self.scaler_y.inverse_transform(
            q25_scaled.reshape(-1, 1)
        ).ravel()
        
        q75 = self.scaler_y.inverse_transform(
            q75_scaled.reshape(-1, 1)
        ).ravel()
        
        # Calibrate quantiles
        q_shift = self.train_target_mean - np.nanmean(pred_point)
        q25 = q25 + q_shift
        q75 = q75 + q_shift
        
        # Q50 approximation
        q50 = (pred_point + (q25 + q75) / 2) / 2
        
        # Convert regime for compatibility
        regime = np.full(len(pred_point), 2, dtype=int)
        regime[pred_trend == 1] = 0
        regime[pred_range == 1] = 1
        
        return pred_point, q25, q50, q75, regime
    
    def get_metadata(self):
        """Get model metadata"""
        return {
            'version': self.version,
            'creation_date': self.creation_date,
            'horizon': self.horizon,
            'features': len(self.feature_cols) if self.feature_cols else 0,
            'train_data_range': self.training_data_range,
            'train_atr_median': self.train_atr_median,
            'train_target_mean': self.train_target_mean,
            'train_target_std': self.train_target_std
        }