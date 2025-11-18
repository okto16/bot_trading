"""
Component Testing Script
Test each component individually to ensure they work correctly
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import MetaTraderDataLoader
from label_creator import LabelCreator
from feature_engineer import FeatureEngineer


def test_data_loader():
    """Test the data loader component"""
    print("\n" + "="*60)
    print("TEST 1: DATA LOADER")
    print("="*60)
    
    csv_path = "../data/XAUUSD_M15_202301030100_202511141930.csv"
    
    try:
        loader = MetaTraderDataLoader(csv_path)
        df = loader.load()
        
        print(f"✓ Data loaded successfully")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        print(f"✓ Missing values: {df.isnull().sum().sum()}")
        
        return df, True
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, False


def test_label_creator(df):
    """Test the label creator component"""
    print("\n" + "="*60)
    print("TEST 2: LABEL CREATOR")
    print("="*60)
    
    try:
        label_creator = LabelCreator(horizon=30)
        df_with_labels = label_creator.create_all_labels(df)
        
        print(f"\n✓ Labels created successfully")
        print(f"✓ Target column exists: {'target' in df_with_labels.columns}")
        print(f"✓ Regime columns exist:")
        print(f"  - regime_trend: {'regime_trend' in df_with_labels.columns}")
        print(f"  - regime_direction: {'regime_direction' in df_with_labels.columns}")
        print(f"  - regime_volatility: {'regime_volatility' in df_with_labels.columns}")
        print(f"  - regime_range: {'regime_range' in df_with_labels.columns}")
        
        # Check unique values
        print(f"\n✓ Regime value counts:")
        print(f"  Trend: {df_with_labels['regime_trend'].value_counts().to_dict()}")
        print(f"  Direction: {df_with_labels['regime_direction'].value_counts().to_dict()}")
        print(f"  Volatility: {df_with_labels['regime_volatility'].value_counts().to_dict()}")
        print(f"  Range: {df_with_labels['regime_range'].value_counts().to_dict()}")
        
        return df_with_labels, True
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, False


def test_feature_engineer(df):
    """Test the feature engineer component"""
    print("\n" + "="*60)
    print("TEST 3: FEATURE ENGINEER")
    print("="*60)
    
    try:
        engineer = FeatureEngineer()
        df_with_features = engineer.create_features(df)
        
        print(f"\n✓ Features created successfully")
        
        # Count features by category
        trend_features = [c for c in df_with_features.columns if c.startswith('ema') or c.startswith('price_vs')]
        momentum_features = ['macd', 'roc5', 'roc10', 'roc20', 'rsi', 'rsi_zscore']
        volatility_features = [c for c in df_with_features.columns if 'atr' in c or 'vol' in c or 'bb' in c or 'tr' in c]
        
        print(f"\n✓ Feature categories:")
        print(f"  Trend features: {len([c for c in trend_features if c in df_with_features.columns])}")
        print(f"  Momentum features: {len([c for c in momentum_features if c in df_with_features.columns])}")
        print(f"  Volatility features: {len([c for c in volatility_features if c in df_with_features.columns])}")
        
        # Check for NaN
        print(f"\n✓ Data quality:")
        print(f"  Total features: {len(df_with_features.columns)}")
        print(f"  Missing values: {df_with_features.isnull().sum().sum()}")
        
        # Sample features
        feature_cols = [c for c in df_with_features.columns if c.endswith('_f') or 
                       c in ['macd', 'rsi', 'atr_pct', 'bb_width']]
        print(f"\n✓ Sample features (first 5):")
        for col in feature_cols[:5]:
            print(f"  {col}: mean={df_with_features[col].mean():.6f}, std={df_with_features[col].std():.6f}")
        
        return df_with_features, True
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, False


def test_ensemble_model(df):
    """Test that the ensemble model can be instantiated"""
    print("\n" + "="*60)
    print("TEST 4: ENSEMBLE MODEL")
    print("="*60)
    
    try:
        from ensemble_model import FinalEnsemble
        
        ensemble = FinalEnsemble()
        
        print(f"✓ Ensemble instantiated successfully")
        print(f"✓ Models available:")
        print(f"  - Main Regressor: {ensemble.main_model is not None}")
        print(f"  - Q25 Model: {ensemble.q25_model is not None}")
        print(f"  - Q75 Model: {ensemble.q75_model is not None}")
        print(f"  - Trend Classifier: {ensemble.trend_model is not None}")
        print(f"  - Direction Classifier: {ensemble.direction_model is not None}")
        print(f"  - Volatility Classifier: {ensemble.volatility_model is not None}")
        print(f"  - Range Classifier: {ensemble.range_model is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_signal_generator():
    """Test that the signal generator can be instantiated"""
    print("\n" + "="*60)
    print("TEST 5: SIGNAL GENERATOR")
    print("="*60)
    
    try:
        from signal_generator import SignalGenerator
        
        signal_gen = SignalGenerator(min_size=0.02, max_size=0.08, cooldown_period=3)
        
        print(f"✓ Signal generator instantiated successfully")
        print(f"✓ Configuration:")
        print(f"  Min size: {signal_gen.min_size}")
        print(f"  Max size: {signal_gen.max_size}")
        print(f"  Cooldown: {signal_gen.cooldown_period}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all component tests"""
    print("\n" + "#"*60)
    print("COMPONENT TESTING SUITE")
    print("#"*60)
    print("\nTesting individual components...")
    
    results = {}
    
    # Test 1: Data Loader
    df, success = test_data_loader()
    results['data_loader'] = success
    
    if not success:
        print("\n✗ Data loader failed. Cannot continue tests.")
        return
    
    # Test 2: Label Creator
    df, success = test_label_creator(df)
    results['label_creator'] = success
    
    if not success:
        print("\n✗ Label creator failed. Cannot continue tests.")
        return
    
    # Test 3: Feature Engineer
    df, success = test_feature_engineer(df)
    results['feature_engineer'] = success
    
    # Test 4: Ensemble Model
    success = test_ensemble_model(df)
    results['ensemble_model'] = success
    
    # Test 5: Signal Generator
    success = test_signal_generator()
    results['signal_generator'] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:20s}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ All components are working correctly")
        print("🚀 Ready to run the full pipeline")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("Please fix the errors before running the full pipeline")


if __name__ == "__main__":
    main()
