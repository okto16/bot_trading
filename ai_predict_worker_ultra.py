"""
AI PREDICT WORKER - COMPONENT-BASED ARCHITECTURE
================================================

Clean, modular worker that uses pre-trained 7-model ensemble.

This is a SIMPLIFIED version that:
1. Loads pre-trained model from PKL
2. Processes incoming data
3. Returns predictions

NO training code here - model is pre-trained!

Author: Refactored Component-Based Version
Version: 3.0
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Tuple, Dict

# Import our components
from utils.data_loader import MetaTraderDataLoader
from utils.label_creator import LabelCreator
from utils.feature_engineer import FeatureEngineer


class ProductionPredictor:
    """
    Production prediction system using pre-trained 7-model ensemble.
    
    This class handles:
    1. Loading pre-trained model
    2. Processing incoming data
    3. Making predictions
    4. Returning signals
    
    NO training - model is pre-trained and loaded from PKL!
    """
    
    def __init__(self, model_path: str = "models/ensemble_7model.pkl",
                 horizon: int = 30):
        """
        Initialize predictor with pre-trained model
        
        Args:
            model_path: Path to pre-trained .pkl file
            horizon: Prediction horizon (must match training)
        """
        self.model_path = Path(model_path)
        self.horizon = horizon
        self.model = None
        
        # Components
        self.label_creator = LabelCreator(horizon=horizon)
        self.feature_engineer = FeatureEngineer()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model from PKL"""
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please train the model first: python main.py"
            )
        
        print(f"\n{'='*60}")
        print("LOADING PRE-TRAINED MODEL")
        print(f"{'='*60}")
        print(f"Loading from: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        metadata = self.model.get_metadata()
        
        print(f"\n✅ Model loaded successfully!")
        print(f"\nModel Info:")
        print(f"  Version: {metadata['version']}")
        print(f"  Created: {metadata['creation_date']}")
        print(f"  Horizon: {metadata['horizon']} bars")
        print(f"  Features: {metadata['features']}")
        print(f"  Training Range: {metadata['train_data_range']}")
        
        # Verify horizon matches
        if metadata['horizon'] != self.horizon:
            print(f"\n⚠️  WARNING: Model horizon ({metadata['horizon']}) "
                  f"!= requested horizon ({self.horizon})")
            print(f"   Using model's horizon: {metadata['horizon']}")
            self.horizon = metadata['horizon']
        
        print(f"{'='*60}\n")
    
    def process_and_predict(self, df: pd.DataFrame) -> Dict:
        """
        Process raw OHLCV data and make predictions
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with predictions and metadata
        """
        
        print(f"\n{'='*60}")
        print("PROCESSING DATA & MAKING PREDICTIONS")
        print(f"{'='*60}")
        
        # Step 1: Create labels (for feature engineering, not for training!)
        print("\nStep 1: Creating labels...")
        df_labeled = self.label_creator.create_all_labels(df)
        
        # Step 2: Engineer features
        print("\nStep 2: Engineering features...")
        df_featured = self.feature_engineer.create_features(df_labeled)
        
        # Step 3: Prepare data for prediction
        print("\nStep 3: Preparing data...")
        
        # Get feature columns from model
        feature_cols = self.model.feature_cols
        
        # Required columns
        required_cols = feature_cols + ['atr']
        
        # Clean data
        df_clean = df_featured.dropna(subset=required_cols)
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after cleaning!")
        
        print(f"  Clean data: {df_clean.shape}")
        
        # Extract features and ATR
        X = df_clean[feature_cols]
        atr = df_clean['atr'].values
        
        # Step 4: Make predictions
        print("\nStep 4: Making predictions...")
        pred_point, q25, q50, q75, regime = self.model.predict(X, atr)
        
        print(f"  Predictions: {len(pred_point)}")
        print(f"  Mean prediction: {np.mean(pred_point):.6f}")
        print(f"  Median IQR: {np.median(q75 - q25):.6f}")
        
        # Step 5: Package results
        results = {
            'predictions': pred_point,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'regime': regime,
            'timestamps': df_clean.index,
            'close': df_clean['Close'].values,
            'atr': atr,
            'n_predictions': len(pred_point)
        }
        
        # Add regime counts
        regime_counts = {
            'trend': np.sum(regime == 0),
            'range': np.sum(regime == 1),
            'neutral': np.sum(regime == 2)
        }
        results['regime_counts'] = regime_counts
        
        print(f"\nRegime Distribution:")
        print(f"  TREND: {regime_counts['trend']} ({regime_counts['trend']/len(regime):.1%})")
        print(f"  RANGE: {regime_counts['range']} ({regime_counts['range']/len(regime):.1%})")
        print(f"  NEUTRAL: {regime_counts['neutral']} ({regime_counts['neutral']/len(regime):.1%})")
        
        print(f"\n✅ Predictions complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def predict_from_csv(self, csv_path: str) -> Dict:
        """
        Convenience method: Load CSV and predict
        
        Args:
            csv_path: Path to MetaTrader CSV file
            
        Returns:
            Dictionary with predictions
        """
        
        # Load data
        loader = MetaTraderDataLoader(csv_path)
        df = loader.load()
        
        # Process and predict
        return self.process_and_predict(df)


# ============================================================================
# SIGNAL GENERATION (Simplified version)
# ============================================================================

def generate_signals_simple(predictions: np.ndarray,
                            q25: np.ndarray,
                            q75: np.ndarray,
                            regime: np.ndarray,
                            close: np.ndarray,
                            atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple signal generation using predictions and regimes
    
    Args:
        predictions: Point predictions
        q25, q75: Uncertainty bounds
        regime: Regime classification (0=TREND, 1=RANGE, 2=NEUTRAL)
        close: Close prices
        atr: ATR values
        
    Returns:
        signals: -1/0/1 (SELL/HOLD/BUY)
        sizes: Position sizes
    """
    
    # Confidence based on uncertainty
    iqr = q75 - q25
    median_iqr = np.median(iqr[~np.isnan(iqr)])
    confidence = np.clip(1 - (iqr / (median_iqr * 2 + 1e-10)), 0, 1)
    
    # Threshold based on predictions
    threshold = np.percentile(np.abs(predictions), 75) * 0.3
    
    # Initialize
    signals = np.zeros(len(predictions), dtype=int)
    sizes = np.zeros(len(predictions))
    
    min_size = 0.02
    max_size = 0.08
    
    for i in range(len(predictions)):
        
        # Skip if in RANGE regime (toxic!)
        if regime[i] == 1:
            continue
        
        # Skip if low confidence
        if confidence[i] < 0.5:
            continue
        
        # Skip if prediction too small
        if np.abs(predictions[i]) < threshold:
            continue
        
        # Generate signals
        if predictions[i] > threshold and regime[i] == 0:  # BUY in TREND
            signals[i] = 1
            strength = np.clip(predictions[i] / threshold - 1, 0, 1)
            sizes[i] = np.clip(min_size + strength * confidence[i] * (max_size - min_size),
                              min_size, max_size)
        
        elif predictions[i] < -threshold and regime[i] == 0:  # SELL in TREND
            signals[i] = -1
            strength = np.clip(-predictions[i] / threshold - 1, 0, 1)
            sizes[i] = np.clip(min_size + strength * confidence[i] * (max_size - min_size),
                              min_size, max_size)
    
    n_signals = np.sum(signals != 0)
    n_range_filtered = np.sum(regime == 1)
    
    print(f"\nSignal Generation:")
    print(f"  Total signals: {n_signals} ({n_signals/len(signals):.1%})")
    print(f"  BUY: {np.sum(signals == 1)}")
    print(f"  SELL: {np.sum(signals == -1)}")
    print(f"  Range filtered: {n_range_filtered} ({n_range_filtered/len(signals):.1%})")
    
    return signals, sizes


# ============================================================================
# MAIN INTERFACE (For use by server)
# ============================================================================

class AIWorker:
    """
    Main worker class for production use.
    
    This is the interface that server_ultra_enhanced.js will call.
    """
    
    def __init__(self, model_path: str = "models/ensemble_7model.pkl"):
        """Initialize worker with pre-trained model"""
        self.predictor = ProductionPredictor(model_path=model_path)
        print("\n✅ AI Worker initialized and ready!")
    
    def predict(self, csv_path: str = None, df: pd.DataFrame = None) -> Dict:
        """
        Make predictions from CSV file or DataFrame
        
        Args:
            csv_path: Path to CSV file (if loading from file)
            df: DataFrame (if data already in memory)
            
        Returns:
            Dictionary with predictions and signals
        """
        
        if csv_path is not None:
            results = self.predictor.predict_from_csv(csv_path)
        elif df is not None:
            results = self.predictor.process_and_predict(df)
        else:
            raise ValueError("Must provide either csv_path or df")
        
        # Generate signals
        signals, sizes = generate_signals_simple(
            results['predictions'],
            results['q25'],
            results['q75'],
            results['regime'],
            results['close'],
            results['atr']
        )
        
        results['signals'] = signals
        results['sizes'] = sizes
        
        return results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for testing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Predict Worker')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--model', default='models/ensemble_7model.pkl',
                       help='Path to model PKL file')
    parser.add_argument('--output', help='Output CSV path (optional)')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("AI PREDICT WORKER - PRODUCTION MODE")
    print(f"{'#'*60}")
    
    # Initialize worker
    worker = AIWorker(model_path=args.model)
    
    # Make predictions
    results = worker.predict(csv_path=args.csv)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {results['n_predictions']}")
    print(f"Signals generated: {np.sum(results['signals'] != 0)}")
    print(f"  BUY: {np.sum(results['signals'] == 1)}")
    print(f"  SELL: {np.sum(results['signals'] == -1)}")
    
    # Save to CSV if requested
    if args.output:
        output_df = pd.DataFrame({
            'timestamp': results['timestamps'],
            'close': results['close'],
            'prediction': results['predictions'],
            'q25': results['q25'],
            'q75': results['q75'],
            'regime': results['regime'],
            'signal': results['signals'],
            'size': results['sizes']
        })
        output_df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    """
    USAGE:
    
    # Test with CSV file:
    python ai_predict_worker.py --csv data/XAUUSD_M15_latest.csv
    
    # Save results to CSV:
    python ai_predict_worker.py --csv data/latest.csv --output predictions.csv
    
    # Use custom model:
    python ai_predict_worker.py --csv data/latest.csv --model models/my_model.pkl
    """
    
    main()