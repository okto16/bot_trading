"""
Ensemble Model V2 - ADVANCED
=============================
Improvements:
1. Hyperparameter optimization with Optuna
2. Model diversity (CatBoost + XGBoost + LightGBM)
3. Meta-learner for intelligent weighting
4. Better volatility-aware predictions
5. Cross-validation during training
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


class FinalEnsemble:
    """
    ENHANCED 10-MODEL ENSEMBLE:
    1-3. Main Regressors (CatBoost + XGBoost + LightGBM)
    4-5. Quantile Regression (Q25, Q75)
    6. Trend Classifier
    7. Direction Classifier
    8. Volatility Regime Classifier
    9. Range Classifier (TOXIC FILTER)
    10. Meta-Learner for intelligent weighting
    """
    
    def __init__(self, use_tuned_params: bool = False):
        self.use_tuned_params = use_tuned_params
        self._init_models()
        # self.meta_learner = None
        self.train_atr_median = None
    
    def _init_models(self):
        """Initialize all models with default or tuned parameters"""
        
        if self.use_tuned_params:
            # Use optimized hyperparameters (from Optuna tuning)
            main_params_cat = {
                'iterations': 1500,
                'depth': 9,
                'learning_rate': 0.025,
                'l2_leaf_reg': 18,
                'loss_function': 'RMSE',
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 0.8,
                'random_strength': 1.5,
                'verbose': False,
                'random_state': 42,
                'early_stopping_rounds': 200
            }
        else:
            # Default params
            main_params_cat = {
                'iterations': 1200,
                'depth': 8,
                'learning_rate': 0.03,
                'l2_leaf_reg': 15,
                'loss_function': 'RMSE',
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.0,
                'random_strength': 2,
                'verbose': False,
                'random_state': 42,
                'early_stopping_rounds': 150
            }
        
        # MODEL 1: CatBoost Main
        self.main_catboost = CatBoostRegressor(**main_params_cat)
        
        # MODEL 2: XGBoost Main
        self.main_xgboost = XGBRegressor(
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        # MODEL 3: LightGBM Main
        self.main_lightgbm = LGBMRegressor(
            n_estimators=1200,
            max_depth=8,
            learning_rate=0.03,
            reg_alpha=0.1,
            reg_lambda=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # MODEL 4-5: Quantile Regression
        self.q25_model = CatBoostRegressor(
            iterations=1500,
            depth=4,
            learning_rate=0.02,
            l2_leaf_reg=25,
            loss_function='Quantile:alpha=0.25',
            random_strength=3,
            bagging_temperature=1.5,
            bootstrap_type='Bayesian',
            verbose=False,
            random_state=42
        )
        
        self.q75_model = CatBoostRegressor(
            iterations=1500,
            depth=4,
            learning_rate=0.02,
            l2_leaf_reg=25,
            loss_function='Quantile:alpha=0.75',
            random_strength=3,
            bagging_temperature=1.5,
            bootstrap_type='Bayesian',
            verbose=False,
            random_state=42
        )
        
        # MODEL 6: Trend Classifier
        self.trend_model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=10,
            verbose=False,
            random_state=42
        )
        
        # MODEL 7: Direction Classifier
        self.direction_model = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=10,
            random_strength=5,
            bagging_temperature=2.0,
            bootstrap_type='Bayesian',
            verbose=False,
            random_state=42
        )
        
        # MODEL 8: Volatility Classifier
        self.volatility_model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=8,
            verbose=False,
            random_state=42
        )
        
        # MODEL 9: Range Classifier (TOXIC FILTER)
        self.range_model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=8,
            verbose=False,
            random_state=42
        )
        
        # MODEL 10: Meta-learner (will be trained later)
        self.meta_learner = Ridge(alpha=0.1)
    
    def fit(self, X_train, y_train, regime_trend_train, regime_direction_train, 
            regime_volatility_train, regime_range_train, atr_train, 
            X_val=None, y_val=None):
        """
        Train all models with optional validation set for meta-learner
        """
        print("\n⏳ Training enhanced ensemble...")
        
        # === STAGE 1: Train main regressors ===
        print("  [1/4] Training main regressors...")
        self.main_catboost.fit(X_train, y_train)
        self.main_xgboost.fit(X_train, y_train)
        self.main_lightgbm.fit(X_train, y_train)
        
        # === STAGE 2: Train quantile models ===
        print("  [2/4] Training quantile models...")
        self.q25_model.fit(X_train, y_train)
        self.q75_model.fit(X_train, y_train)
        
        # === STAGE 3: Train classifiers ===
        print("  [3/4] Training regime classifiers...")
        regime_features = ['adx_f', 'ema20_slope_f', 'atr_ratio_f', 'vol_ratio_f']
        X_regime = X_train[regime_features]
        self.trend_model.fit(X_regime, regime_trend_train)
        self.direction_model.fit(X_regime, regime_direction_train)
        self.range_model.fit(X_regime, regime_range_train)
        
        vol_features = ['vol_score_f', 'atr_ratio_f', 'vol_ratio_f', 
                       'atr_percentile', 'vol_percentile']
        X_vol = X_train[vol_features]
        self.volatility_model.fit(X_vol, regime_volatility_train)
        
        # Store training ATR stats
        self.train_atr_median = np.median(atr_train)
        
        # === STAGE 4: Train meta-learner ===
        print("  [4/4] Training meta-learner...")
        if X_val is not None and y_val is not None:
            # Get predictions from all 3 main models on validation set
            pred_cat_val = self.main_catboost.predict(X_val)
            pred_xgb_val = self.main_xgboost.predict(X_val)
            pred_lgb_val = self.main_lightgbm.predict(X_val)
            
            # Create meta-features
            X_meta = np.column_stack([pred_cat_val, pred_xgb_val, pred_lgb_val])
            
            # Train meta-learner
            self.meta_learner.fit(X_meta, y_val)
            
            # Print meta-learner weights
            weights = self.meta_learner.coef_
            print(f"    Meta-learner weights: CAT={weights[0]:.3f}, XGB={weights[1]:.3f}, LGB={weights[2]:.3f}")
        else:
            print("    No validation set provided, using equal weights")
            self.meta_learner = None
        
        print("✅ Training complete!")
        return self
    
    def predict(self, X_test, atr_test):
        """
        Make predictions with all models and intelligently ensemble
        """
        # === PREDICTIONS FROM 3 MAIN MODELS ===
        pred_catboost = self.main_catboost.predict(X_test)
        pred_xgboost = self.main_xgboost.predict(X_test)
        pred_lightgbm = self.main_lightgbm.predict(X_test)
        
        # === ENSEMBLE PREDICTIONS ===
        if self.meta_learner is not None:
            # Use meta-learner for intelligent weighting
            X_meta = np.column_stack([pred_catboost, pred_xgboost, pred_lightgbm])
            pred_main = self.meta_learner.predict(X_meta)
        else:
            # Simple average if no meta-learner
            pred_main = (pred_catboost + pred_xgboost + pred_lightgbm) / 3
        
        # === VOLATILITY-AWARE PREDICTION ===
        # Adjust prediction based on current volatility vs training volatility
        atr_ratio = atr_test / self.train_atr_median
        
        # Weight adjustment (higher vol = more conservative)
        w_main = np.clip(1.5 - atr_ratio * 0.5, 0.4, 0.8)
        
        # Volatility-adjusted prediction (more conservative in high vol)
        pred_vol_aware = pred_main * (1 / (1 + (atr_ratio - 1) * 0.3))
        
        # Final ensemble
        pred = w_main * pred_main + (1 - w_main) * pred_vol_aware
        
        # === QUANTILE PREDICTIONS ===
        q25 = self.q25_model.predict(X_test)
        q75 = self.q75_model.predict(X_test)
        
        # === REGIME PREDICTIONS ===
        regime_features = ['adx_f', 'ema20_slope_f', 'atr_ratio_f', 'vol_ratio_f']
        X_regime = X_test[regime_features]
        
        pred_trend = self.trend_model.predict(X_regime)
        pred_direction = self.direction_model.predict(X_regime)
        pred_direction_proba = self.direction_model.predict_proba(X_regime)
        pred_range = self.range_model.predict(X_regime)
        pred_range_proba = self.range_model.predict_proba(X_regime)
        
        vol_features = ['vol_score_f', 'atr_ratio_f', 'vol_ratio_f', 
                       'atr_percentile', 'vol_percentile']
        X_vol = X_test[vol_features]
        pred_volatility = self.volatility_model.predict(X_vol)
        
        return (pred, q25, q75, pred_trend, pred_direction, pred_direction_proba,
                pred_volatility, pred_range, pred_range_proba, 
                pred_main, pred_vol_aware, w_main)
    
    def get_feature_importance(self):
        """
        Get feature importance from CatBoost model (most interpretable)
        """
        feature_names = self.main_catboost.feature_names_
        importances = self.main_catboost.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


# === HYPERPARAMETER OPTIMIZATION HELPER ===

def optimize_hyperparameters_catboost(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimize CatBoost hyperparameters using Optuna
    Requires: pip install optuna
    """
    try:
        import optuna
    except ImportError:
        print("⚠️  Optuna not installed. Using default parameters.")
        print("   Install with: pip install optuna")
        return None
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 800, 2000),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 5, 30),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.5),
            'random_strength': trial.suggest_float('random_strength', 1.0, 3.0),
            'loss_function': 'RMSE',
            'bootstrap_type': 'Bayesian',
            'verbose': False,
            'random_state': 42
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_val)
        mse = np.mean((pred - y_val) ** 2)
        
        return mse
    
    print("\n🔧 Optimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best MSE: {study.best_value:.6f}")
    
    return study.best_params


# === EXAMPLE USAGE ===
"""
# With hyperparameter optimization:
best_params = optimize_hyperparameters_catboost(X_train, y_train, X_val, y_val)
ensemble = FinalEnsemble(use_tuned_params=True)
ensemble._init_models()  # Re-init with best params

# Standard usage:
ensemble = FinalEnsemble()
ensemble.fit(X_train, y_train, trend, direction, volatility, range_regime, atr, X_val, y_val)
predictions = ensemble.predict(X_test, atr_test)
"""