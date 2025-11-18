"""
AI PREDICT WORKER ADAPTER - 7-MODEL ENSEMBLE
=============================================

This adapter allows ai_predict_worker_ultra.py to use the 7-model ensemble
while maintaining compatibility with the existing server infrastructure.

Usage:
1. Train and export model: python export_7model_ensemble.py
2. Replace the model in ai_predict_worker_ultra.py with this adapter
3. The server will automatically use the 7-model ensemble

Author: Integration adapter for production deployment
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict


class Ensemble7ModelAdapter:
    """
    Adapter class that wraps the 7-model ensemble to work with
    ai_predict_worker_ultra.py interface.
    
    This makes the 7-model ensemble compatible with the existing
    3-model EnsembleSystem interface expected by the worker.
    """
    
    def __init__(self, model_path: str = "models/ensemble_7model.pkl"):
        """
        Initialize adapter by loading the 7-model ensemble.
        
        Args:
            model_path: Path to the saved .pkl file
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the 7-model ensemble from .pkl file"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please run: python export_7model_ensemble.py"
            )
        
        print(f"Loading 7-model ensemble from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Version: {self.model.version}")
        print(f"  Created: {self.model.creation_date}")
        print(f"  Features: {len(self.model.feature_cols)}")
    
    def fit(self, X_train, y_train_reg, y_train_regime=None):
        """
        Training interface compatible with ai_predict_worker.
        
        NOTE: For production, the model is pre-trained.
        This method is here for interface compatibility only.
        
        To retrain, use: python export_7model_ensemble.py
        """
        print("\n⚠️  WARNING: 7-model ensemble is pre-trained!")
        print("To retrain the model, use: python export_7model_ensemble.py")
        print("This adapter uses the pre-trained model from .pkl file.\n")
        
        # Don't actually train - model is pre-trained
        pass
    
    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Predict interface compatible with ai_predict_worker.
        
        Returns:
            Tuple of (pred_point, pred_q25, pred_q50, pred_q75, pred_regime)
            - pred_point: Point prediction (ensembled from main + vol-aware)
            - pred_q25: 25th percentile (uncertainty lower bound)
            - pred_q50: Median prediction (approximation)
            - pred_q75: 75th percentile (uncertainty upper bound)
            - pred_regime: Regime prediction (None for compatibility)
            
        Note: This adapter translates 7-model outputs to 3-model format
        """
        
        # Need ATR for the 7-model ensemble
        # If ATR not in features, we'll need to compute it from OHLC
        # For now, assume it's available or use a fallback
        
        # Try to get ATR from the test data context
        # This requires passing additional data - see enhanced version below
        atr_test = self._get_atr_from_context(X_test)
        
        # Call the 7-model ensemble
        pred_point, q25, q75, pred_trend, pred_direction, \
        pred_volatility, pred_range, diagnostics = self.model.predict(
            X_test, atr_test
        )
        
        # Approximate Q50 (median) as average of point prediction and midpoint
        pred_q50 = (pred_point + (q25 + q75) / 2) / 2
        
        # For regime, we have multiple classifiers - let's combine them
        # TREND classifier output: 0=NOT-TREND, 1=TREND
        # We'll map to the old format: 0=TREND, 1=RANGE, 2=NEUTRAL
        
        # If in RANGE regime (toxic), mark as RANGE (1)
        # If in TREND regime, mark as TREND (0)
        # Otherwise NEUTRAL (2)
        pred_regime = np.full(len(pred_point), 2, dtype=int)  # Default: NEUTRAL
        pred_regime[pred_trend == 1] = 0  # TREND
        pred_regime[pred_range == 1] = 1  # RANGE (toxic)
        
        # Print diagnostics
        n_trend = np.sum(pred_trend == 1)
        n_range = np.sum(pred_range == 1)
        n_up = np.sum(pred_direction == 1)
        
        print(f"\n7-Model Ensemble Prediction:")
        print(f"  Samples: {len(pred_point)}")
        print(f"  TREND: {n_trend} ({n_trend/len(pred_trend):.1%})")
        print(f"  RANGE (toxic): {n_range} ({n_range/len(pred_range):.1%})")
        print(f"  Direction UP: {n_up} ({n_up/len(pred_direction):.1%})")
        print(f"  Vol-aware weight: {np.mean(1-diagnostics['weights_main']):.2%}")
        
        return pred_point, q25, pred_q50, q75, pred_regime
    
    def _get_atr_from_context(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Extract or compute ATR values.
        
        This is a fallback method. Ideally, ATR should be passed explicitly.
        """
        # Check if ATR-related features exist
        atr_features = [col for col in X_test.columns if 'atr' in col.lower()]
        
        if atr_features:
            # Use the first ATR feature as proxy
            # Note: This might be normalized - need to denormalize
            # For now, use as-is with a scaling factor
            atr_proxy = X_test[atr_features[0]].values
            
            # Typical ATR for gold is 1-5 USD
            # If values are normalized (0-1 range), scale up
            if atr_proxy.max() < 1.0:
                atr_test = atr_proxy * 3.0  # Approximate scaling
            else:
                atr_test = atr_proxy
        else:
            # Fallback: Use a constant median ATR
            # This is not ideal but prevents crashes
            print("⚠️  WARNING: ATR not found in features, using fallback value")
            atr_test = np.full(len(X_test), 2.5)  # Typical gold ATR
        
        return atr_test


# ============================================================================
# ENHANCED ADAPTER WITH EXPLICIT ATR HANDLING
# ============================================================================

class Ensemble7ModelAdapterEnhanced:
    """
    Enhanced adapter that requires explicit ATR values.
    
    This is the RECOMMENDED adapter as it properly handles the
    volatility-aware model component.
    
    Usage requires modifying ai_predict_worker to pass ATR data.
    """
    
    def __init__(self, model_path: str = "models/ensemble_7model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
        
        # Store ATR for prediction
        self.current_atr = None
        
    def _load_model(self):
        """Load the 7-model ensemble"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ 7-Model Ensemble Loaded")
        print(f"  Version: {self.model.version}")
        print(f"  Features: {len(self.model.feature_cols)}")
    
    def set_atr(self, atr_values: np.ndarray):
        """
        Set ATR values for the next prediction.
        
        Call this before predict() with the ATR values.
        """
        self.current_atr = atr_values
    
    def fit(self, X_train, y_train_reg, y_train_regime=None):
        """Interface compatibility - model is pre-trained"""
        print("⚠️  Model is pre-trained. Use export_7model_ensemble.py to retrain.")
    
    def predict(self, X_test: pd.DataFrame, 
                atr_test: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        Enhanced prediction with explicit ATR handling.
        
        Args:
            X_test: Test features
            atr_test: ATR values (optional if set_atr() was called)
            
        Returns:
            Tuple compatible with ai_predict_worker interface
        """
        
        # Use provided ATR or stored ATR
        if atr_test is None:
            if self.current_atr is None:
                raise ValueError(
                    "ATR not provided! Either pass atr_test or call set_atr() first"
                )
            atr_test = self.current_atr
        
        # Validate ATR
        if len(atr_test) != len(X_test):
            raise ValueError(
                f"ATR length ({len(atr_test)}) doesn't match X_test ({len(X_test)})"
            )
        
        # Call 7-model ensemble
        pred_point, q25, q75, pred_trend, pred_direction, \
        pred_volatility, pred_range, diagnostics = self.model.predict(
            X_test, atr_test
        )
        
        # Approximate Q50
        pred_q50 = (pred_point + (q25 + q75) / 2) / 2
        
        # Convert regime classifications
        pred_regime = np.full(len(pred_point), 2, dtype=int)
        pred_regime[pred_trend == 1] = 0  # TREND
        pred_regime[pred_range == 1] = 1  # RANGE
        
        # Store extended info for later use
        self.last_prediction_info = {
            'trend': pred_trend,
            'direction': pred_direction,
            'volatility': pred_volatility,
            'range': pred_range,
            'direction_proba': diagnostics['direction_proba'],
            'range_proba': diagnostics['range_proba'],
            'vol_aware_weight': 1 - diagnostics['weights_main']
        }
        
        return pred_point, q25, pred_q50, q75, pred_regime
    
    def get_extended_info(self) -> Dict:
        """
        Get extended prediction information from 7-model ensemble.
        
        Returns detailed regime classifications and probabilities.
        """
        if not hasattr(self, 'last_prediction_info'):
            return {}
        return self.last_prediction_info


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Example: Using the basic adapter (automatic ATR handling)"""
    
    # Initialize adapter
    adapter = Ensemble7ModelAdapter(
        model_path="models/ensemble_7model.pkl"
    )
    
    # Create sample test data (your actual features)
    X_test = pd.DataFrame({
        'ema20_f': np.random.randn(100),
        'rsi': np.random.randn(100),
        'atr_pct': np.random.rand(100) * 0.01,  # ATR percentage
        # ... other features
    })
    
    # Predict (adapter handles ATR internally)
    pred_point, q25, q50, q75, regime = adapter.predict(X_test)
    
    print(f"Predictions: {len(pred_point)}")
    print(f"Mean: {np.mean(pred_point):.6f}")
    print(f"Regime: {np.bincount(regime)}")


def example_enhanced_usage():
    """Example: Using the enhanced adapter (explicit ATR)"""
    
    # Initialize adapter
    adapter = Ensemble7ModelAdapterEnhanced(
        model_path="models/ensemble_7model.pkl"
    )
    
    # Your test data
    X_test = pd.DataFrame({
        'ema20_f': np.random.randn(100),
        'rsi': np.random.randn(100),
        # ... features
    })
    
    # ATR values (from your data pipeline)
    atr_test = np.random.rand(100) * 3.0 + 1.5  # 1.5-4.5 range
    
    # Predict with explicit ATR
    pred_point, q25, q50, q75, regime = adapter.predict(X_test, atr_test)
    
    # Get extended info
    info = adapter.get_extended_info()
    print(f"Direction UP: {np.sum(info['direction']==1)} bars")
    print(f"Range (toxic): {np.sum(info['range']==1)} bars")
    print(f"Avg vol-aware weight: {np.mean(info['vol_aware_weight']):.2f}")


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("7-MODEL ENSEMBLE ADAPTER")
    print("="*60)
    print("\nThis adapter allows ai_predict_worker_ultra.py to use")
    print("the 7-model ensemble while maintaining compatibility.")
    print("\nRECOMMENDED: Use Ensemble7ModelAdapterEnhanced")
    print("for proper ATR handling and extended features.")
    print("\n" + "="*60)
    
    # Show basic usage
    print("\nBasic usage example:")
    print("-" * 40)
    print("""
from ensemble_7model_adapter import Ensemble7ModelAdapter

# In ai_predict_worker_ultra.py, replace EnsembleSystem with:
ensemble = Ensemble7ModelAdapter("models/ensemble_7model.pkl")

# Then use normally:
pred_point, q25, q50, q75, regime = ensemble.predict(X_test)
    """)
    
    # Show enhanced usage
    print("\nEnhanced usage example:")
    print("-" * 40)
    print("""
from ensemble_7model_adapter import Ensemble7ModelAdapterEnhanced

# Initialize
ensemble = Ensemble7ModelAdapterEnhanced("models/ensemble_7model.pkl")

# Predict with ATR
pred_point, q25, q50, q75, regime = ensemble.predict(X_test, atr_test)

# Get extended regime info
info = ensemble.get_extended_info()
print(f"Toxic range bars: {np.sum(info['range']==1)}")
    """)
