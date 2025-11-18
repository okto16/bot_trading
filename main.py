"""
XAU M15 TRADING MODEL - COMPLETE PIPELINE WITH PKL EXPORT
==========================================================

Main entry point that:
1. Trains the 7-model ensemble
2. Validates performance
3. Exports to .pkl file for production use
4. Tests the exported model

This is the ONLY script you need to run!

Author: Trading Strategy Team
Version: 3.0 (With PKL Export)
"""

import warnings
warnings.filterwarnings('ignore')
from utils.production_model import ProductionModel
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from utils.data_loader import MetaTraderDataLoader
from utils.label_creator import LabelCreator
from utils.feature_engineer import FeatureEngineer
from utils.ensemble_model import FinalEnsemble
from utils.walk_forward_validator import WalkForwardValidator
from sklearn.preprocessing import RobustScaler


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
        
        # Convert regime for compatibility with old interface
        regime = np.full(len(pred_point), 2, dtype=int)  # Default: NEUTRAL
        regime[pred_trend == 1] = 0  # TREND
        regime[pred_range == 1] = 1  # RANGE
        
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


def get_feature_columns(df):
    """Extract feature column names"""
    exclude_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'mid', 'tr', 'atr', 'future_mid', 'forward_return', 'target',
        'target_normalized', 'hour', 'ema200', 
        'regime_trend', 'regime_direction', 'regime_volatility', 'regime_range',
        'ema20_slope', 'ema50_slope', 'ema200_slope', 
        'atr_ratio', 'vol_ratio', 'adx', 'vol_score'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def train_and_export(df: pd.DataFrame, feature_cols: list, 
                     horizon: int, output_path: str = "models/ensemble_7model.pkl",
                     train_window: int = 45000):
    """
    Train model on recent data and export to PKL
    
    Args:
        df: Full DataFrame with features and labels
        feature_cols: List of feature column names
        horizon: Prediction horizon
        output_path: Where to save the .pkl file
        train_window: How many recent bars to use for training
    """
    
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL FOR PRODUCTION")
    print(f"{'='*60}")
    
    # Prepare data
    required_cols = feature_cols + [
        'target', 'regime_trend', 'regime_direction',
        'regime_volatility', 'regime_range', 'atr'
    ]
    
    df_clean = df.dropna(subset=required_cols)
    print(f"\nClean data: {df_clean.shape}")
    
    # Use recent data for training
    if len(df_clean) > train_window:
        df_train = df_clean.iloc[-train_window:]
        print(f"Using last {train_window} bars for final training")
    else:
        df_train = df_clean
        print(f"Using all {len(df_clean)} bars for final training")
    
    print(f"Training range: {df_train.index.min()} to {df_train.index.max()}")
    
    # Extract training data
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    regime_trend = df_train['regime_trend']
    regime_direction = df_train['regime_direction']
    regime_volatility = df_train['regime_volatility']
    regime_range = df_train['regime_range']
    atr_train = df_train['atr'].values
    
    print(f"\nTraining data:")
    print(f"  Features: {X_train.shape}")
    print(f"  Target: mean={y_train.mean():.6f}, std={y_train.std():.6f}")
    print(f"  ATR: median={np.median(atr_train):.2f}")
    
    # Initialize production model
    model = ProductionModel()
    
    # Add metadata
    model.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model.training_data_range = f"{df_train.index.min()} to {df_train.index.max()}"
    
    # Train
    model.fit(
        X_train, y_train,
        regime_trend, regime_direction,
        regime_volatility, regime_range,
        atr_train, horizon
    )
    
    # Test on recent data
    print(f"\n{'='*60}")
    print("TESTING PRODUCTION MODEL")
    print(f"{'='*60}")
    
    test_size = min(1000, len(df_clean) // 10)
    df_test = df_clean.iloc[-test_size:]
    
    X_test = df_test[feature_cols]
    atr_test = df_test['atr'].values
    
    pred_point, q25, q50, q75, regime = model.predict(X_test, atr_test)
    
    print(f"\nPrediction test:")
    print(f"  Samples: {len(pred_point)}")
    print(f"  Mean: {np.mean(pred_point):.6f}")
    print(f"  Std: {np.std(pred_point):.6f}")
    print(f"  IQR median: {np.median(q75 - q25):.6f}")
    print(f"  TREND bars: {np.sum(regime==0)} ({np.sum(regime==0)/len(regime):.1%})")
    print(f"  RANGE bars: {np.sum(regime==1)} ({np.sum(regime==1)/len(regime):.1%})")
    
    # Export to PKL
    print(f"\n{'='*60}")
    print("EXPORTING TO PKL")
    print(f"{'='*60}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Model exported successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    
    # Print metadata
    metadata = model.get_metadata()
    print(f"\nModel Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Feature importance
    print(f"\n{'='*60}")
    print("TOP 20 FEATURES")
    print(f"{'='*60}")
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.ensemble.main_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(20).to_string(index=False))
    
    # Save feature importance
    importance_path = output_path.parent / f"{output_path.stem}_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"\n✅ Feature importance saved: {importance_path}")
    
    return model, output_path


def test_pkl_loading(pkl_path: str):
    """Test loading the saved PKL file"""
    
    print(f"\n{'='*60}")
    print("TESTING PKL FILE LOADING")
    print(f"{'='*60}")
    
    try:
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        
        metadata = model.get_metadata()
        
        print(f"\n✅ Model loaded successfully!")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(csv_path: str, horizon: int = 30, n_splits: int = 5,
         export_model: bool = True, output_path: str = "models/ensemble_7model.pkl"):
    """
    Main pipeline: Train, Validate, and Export
    
    Args:
        csv_path: Path to MetaTrader CSV
        horizon: Prediction horizon in bars
        n_splits: Number of validation splits
        export_model: Whether to export final model to PKL
        output_path: Where to save the PKL file
    """
    
    print(f"\n{'#'*60}")
    print("XAU M15 - 7-MODEL ENSEMBLE WITH PKL EXPORT")
    print(f"{'#'*60}")
    print("\nPIPELINE:")
    print("  1. Data Loading")
    print("  2. Label Creation (4 regimes + target)")
    print("  3. Feature Engineering (39 features)")
    print("  4. Walk-Forward Validation")
    print("  5. Final Model Training")
    print("  6. PKL Export ⭐")
    print("  7. Integration Test")
    
    # ========================================
    # STEP 1: LOAD DATA
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 1: DATA LOADING")
    print(f"{'='*60}")
    
    loader = MetaTraderDataLoader(csv_path)
    df = loader.load()
    
    # ========================================
    # STEP 2: CREATE LABELS
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 2: LABEL CREATION")
    print(f"{'='*60}")
    
    label_creator = LabelCreator(horizon=horizon)
    df = label_creator.create_all_labels(df)
    
    # ========================================
    # STEP 3: ENGINEER FEATURES
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 3: FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n✅ Total Features: {len(feature_cols)}")
    
    # ========================================
    # STEP 4: WALK-FORWARD VALIDATION
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 4: WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    print("(This validates the model before final training)")
    
    validator = WalkForwardValidator(n_splits=n_splits, horizon=horizon)
    results = validator.validate(df, feature_cols)
    
    # Print validation summary
    validator.print_summary()
    
    # ========================================
    # STEP 5: TRAIN FINAL MODEL & EXPORT
    # ========================================
    if export_model:
        print(f"\n{'='*60}")
        print("STEP 5: FINAL MODEL TRAINING & EXPORT")
        print(f"{'='*60}")
        
        model, pkl_path = train_and_export(
            df, feature_cols, horizon, output_path
        )
        
        # ========================================
        # STEP 6: TEST PKL LOADING
        # ========================================
        print(f"\n{'='*60}")
        print("STEP 6: PKL LOADING TEST")
        print(f"{'='*60}")
        
        if test_pkl_loading(pkl_path):
            print("\n✅ PKL file is valid and ready for deployment!")
        else:
            print("\n❌ PKL file has issues - check errors above")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    
    print("\n✅ What was accomplished:")
    print("  1. ✓ Data loaded and processed")
    print("  2. ✓ Labels created (4 regimes + target)")
    print("  3. ✓ Features engineered (39 features)")
    print("  4. ✓ Walk-forward validation completed")
    
    if export_model:
        print(f"  5. ✓ Final model trained")
        print(f"  6. ✓ Model exported to PKL")
        print(f"  7. ✓ PKL loading tested")
        
        print(f"\n📦 Production Model Ready!")
        print(f"  Location: {output_path}")
        print(f"  Use with: ensemble_7model_adapter.py")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Copy {output_path} to your production server")
        print(f"  2. Update ai_predict_worker_ultra.py:")
        print(f"     from ensemble_7model_adapter import Ensemble7ModelAdapter")
        print(f"     ensemble = Ensemble7ModelAdapter('{output_path}')")
        print(f"  3. Restart trading server")
        print(f"  4. Monitor logs for performance")
    else:
        print("  5. ⊘ Model export skipped (set export_model=True)")
    
    print(f"\n{'='*60}")
    
    return df, results


if __name__ == "__main__":
    """
    USAGE:
    
    # Standard run (with validation + export):
    python main.py
    
    # Quick test (validation only, no export):
    python main.py --no-export
    
    # Custom parameters:
    python main.py --horizon 40 --splits 7 --output models/my_model.pkl
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and export 7-model ensemble')
    parser.add_argument('--csv', default='./data/XAUUSD_M15_202301030100_202511062345.csv',
                        help='Path to CSV data')
    parser.add_argument('--horizon', type=int, default=30,
                        help='Prediction horizon in bars (default: 30)')
    parser.add_argument('--splits', type=int, default=5,
                        help='Number of validation splits (default: 5)')
    parser.add_argument('--output', default='models/ensemble_7model.pkl',
                        help='Output PKL path (default: models/ensemble_7model.pkl)')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip model export (validation only)')
    
    args = parser.parse_args()
    
    # Configuration
    CSV_PATH = args.csv
    HORIZON = args.horizon
    N_SPLITS = args.splits
    OUTPUT_PATH = args.output
    EXPORT = not args.no_export
    
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Horizon: {HORIZON} bars ({HORIZON*15/60:.1f} hours)")
    print(f"Validation Splits: {N_SPLITS}")
    print(f"Export Model: {EXPORT}")
    if EXPORT:
        print(f"Output Path: {OUTPUT_PATH}")
    print(f"{'='*60}")
    
    # Run pipeline
    df, results = main(
        csv_path=CSV_PATH,
        horizon=HORIZON,
        n_splits=N_SPLITS,
        export_model=EXPORT,
        output_path=OUTPUT_PATH
    )
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total data points: {len(df)}")
    print(f"Prediction horizon: {HORIZON} bars ({HORIZON*15/60:.1f} hours)")
    print(f"Validation folds: {N_SPLITS}")
    
    if EXPORT:
        print(f"\n✅ Model ready for production!")
        print(f"📦 PKL file: {OUTPUT_PATH}")
        print(f"\n🔗 Integration instructions:")
        print(f"   See: INTEGRATION_GUIDE.md")
    
    print("="*60)
    print("\n🎉 All done! Happy trading! 📈")