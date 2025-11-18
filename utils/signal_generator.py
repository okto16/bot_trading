"""
Signal Generator Component
Generates trading signals with multiple safety filters
"""

from typing import Tuple
import numpy as np


class SignalGenerator:
    """Generates trading signals with comprehensive filters"""

    def __init__(
        self, min_size: float = 0.02, max_size: float = 0.08, cooldown_period: int = 3
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.cooldown_period = cooldown_period

    def generate_signals(
        self,
        pred: np.ndarray,
        q25: np.ndarray,
        q75: np.ndarray,
        regime_trend: np.ndarray,
        regime_direction: np.ndarray,
        direction_proba: np.ndarray,
        regime_volatility: np.ndarray,
        regime_range: np.ndarray,
        range_proba: np.ndarray,
        close: np.ndarray,
        ema200: np.ndarray,
        atr: np.ndarray,
        adx: np.ndarray,
        ema20_slope: np.ndarray,
        bb_width: np.ndarray,
        ema_trend_width: np.ndarray,
        atr_percentile: np.ndarray,
        volume_zscore: np.ndarray,
        timestamps: np.ndarray,
        train_atr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals with RANGE FILTER - Prevents toxic trades!
        """

        pred = np.asarray(pred)
        q25 = np.asarray(q25)
        q75 = np.asarray(q75)
        regime_trend = np.asarray(regime_trend)
        regime_direction = np.asarray(regime_direction)
        direction_proba = np.asarray(direction_proba)
        regime_volatility = np.asarray(regime_volatility).reshape(-1)
        regime_range = np.asarray(regime_range).reshape(-1)
        range_proba = np.asarray(range_proba)
        close = np.asarray(close)
        ema200 = np.asarray(ema200)
        atr = np.asarray(atr)

        # Confidence
        iqr = q75 - q25
        median_iqr = np.median(iqr[~np.isnan(iqr)])
        confidence = np.clip(1 - (iqr / (median_iqr * 2 + 1e-10)), 0, 1)
        reliable = iqr < (median_iqr * 1.5)

        # Threshold
        # local_span = 500
        # start = max(0, i - local_span)
        # local_q75_abs = np.percentile(np.abs(pred[start:i+1]), 75)
        # threshold = local_q75_abs * 0.35
        trigger_threshold = median_iqr * 0.30

        # ATR filter
        atr_90 = np.percentile(atr[~np.isnan(atr)], 90)
        high_atr = atr > atr_90

        # Volatility sizing
        vol_multiplier = np.ones(len(regime_volatility))
        vol_multiplier[regime_volatility == 0] = 0.7  # LOW
        vol_multiplier[regime_volatility == 1] = 1.0  # MEDIUM
        vol_multiplier[regime_volatility == 2] = 0.6  # HIGH

        # Initialize
        signals = np.zeros(len(pred), dtype=int)
        sizes = np.zeros(len(pred), dtype=float)

        min_size = 0.02
        max_size = 0.08

        # Counters
        n_unreliable = 0
        n_high_atr = 0
        n_not_trend = 0
        n_low_dir_conf = 0
        n_weak_trigger = 0
        n_in_range = 0  # NEW: Most important filter!
        n_wrong_ema = 0
        n_whipsaw = 0

        # Cooldown
        cooldown = 0
        cooldown_period = 3
        past_trade_returns = []

        # News filter
        news_hours = [12, 13, 14, 19, 20, 21]
        news_days = []
        
        lookback_k = 3
        min_dir_conf = 0.65
        min_adx = 20
        for i in range(len(pred)):
            # === Dynamic threshold HARUS DI AWAL ===
            local_span = 500
            start = max(0, i - local_span)
            local_pred_q75 = np.percentile(np.abs(pred[max(0, i-400):i+1]), 75)
            threshold = local_pred_q75 * 0.40
            
            # === Volatility expansion (ATR percentile) ===
            vol_adj = atr_percentile[i] * 0.5   # 0–1 range → bump threshold 0–50%
            threshold *= (1 + vol_adj)

            # Cooldown
            if cooldown > 0:
                cooldown -= 1
                continue
            # ===============================
            #  DYNAMIC SIZE REDUCER (NEW)
            # ===============================
           
            local_max = max_size
            if len(past_trade_returns) >= 3:
                recent = past_trade_returns[-3:]
                loss_ratio = np.sum(np.array(recent)<0) / len(recent)
                if loss_ratio > 0.6:
                    local_max *= 0.5

            
            # NEWS FILTER
            hour = timestamps[i].hour
            day = timestamps[i].day
            if hour in news_hours:
                continue
            if day in news_days:
                continue
            
            if abs(ema20_slope[i]) < 0.0006:
                continue

            # new filter: avoid high-vol fake trend
            if (adx[i] > 25) and (atr[i] > atr_90):
                continue
            
            if not reliable[i]:
                n_unreliable += 1
                continue

            if high_atr[i]:
                n_high_atr += 1
                continue
            
            # =============================
            # ANTI WHIPSAW FILTER (NEW)
            # =============================
            if i >= 12:
                flips = np.sum(
                    np.sign(ema20_slope[i - 10 : i])
                    != np.sign(ema20_slope[i - 11 : i - 1])
                )
                whipsaw = adx[i] < 16 and flips >= 3 and bb_width[i] < 0.006
            else:
                whipsaw = False
                
            confirm_len = 3
            if not np.all(np.sign(pred[i-confirm_len:i]) == np.sign(pred[i])):
                continue
            
            # ============================================
            # 2-LEVEL CONFIRMATION (NEW)
            # ============================================

            # 1. EMA20 slope harus searah prediksi
            pred_dir_sign = np.sign(pred[i])
            ema_slope_sign = np.sign(ema20_slope[i])

            if pred_dir_sign != ema_slope_sign:
                continue

            # 2. Regime direction classifier harus searah
            reg_dir_sign = 1 if regime_direction[i] == 1 else -1
            if reg_dir_sign != pred_dir_sign:
                continue

            # 3. Confidence direction harus kuat
            dir_conf = direction_proba[i, regime_direction[i]]
            if dir_conf < 0.70:
                continue

            if whipsaw:
                n_whipsaw += 1
                continue

            if regime_trend[i] != 1:
                n_not_trend += 1
                continue

            if np.abs(pred[i]) < trigger_threshold:
                n_weak_trigger += 1
                continue

            if ema_trend_width[i] < 0.00040:
                continue

            if 0.40 < atr_percentile[i] < 0.60:
                continue
            
            if atr[i] > np.percentile(train_atr, 95):
                continue

            # ============================================
            # FILTER 3: Volume-based anti fake trend (NEW)
            # ============================================
            if volume_zscore[i] > 2.0:
                # low volume = weak trend = most likely fake
                continue

            # ⭐⭐⭐ RANGE FILTER - Most important!
            # NEVER trade when in RANGE (toxic condition)
            if regime_range[i] == 1:
                n_in_range += 1
                continue

            # Also avoid if high probability of being in range
            range_prob = range_proba[i, 1]  # Probability of RANGE
            if range_prob > 0.40:  # If >40% chance of being in range
                n_in_range += 1
                continue
            if regime_volatility[i] == 2:  # HIGH VOL
                # Require strong directional confidence
                if direction_proba[i, regime_direction[i]] < 0.65:
                    continue

                # Require stronger prediction magnitude
                if np.abs(pred[i]) < threshold * 1.3:
                    continue

                # JANGAN BUY ATAU SELL DI HIGH VOL JIKA CONFIDENCE TIDAK KUAT
                if direction_proba[i, regime_direction[i]] < 0.68:
                    continue

                if confidence[i] < 0.55:
                    continue
            above_ema = close[i] > ema200[i]
            below_ema = close[i] < ema200[i]

            pred_dir_class = regime_direction[i]
            dir_confidence = direction_proba[i, pred_dir_class]

            if dir_confidence < 0.55:
                n_low_dir_conf += 1
                continue
            # if dir_confidence < 0.65: continue
            if adx[i] < 20: 
                continue
            if i < lookback_k:
                continue

            prev_signs = np.sign(pred[i-lookback_k:i])

            # Jika tidak konfirmasi arah
            if not (np.all(prev_signs > 0) or np.all(prev_signs < 0)):
                n_weak_trigger += 1
                continue

            # 2) dynamic threshold
            local_span = 500
            start = max(0, i-local_span)
            local_q75_abs = np.percentile(np.abs(pred[start:i+1]), 75)
            threshold = local_q75_abs * 0.35

            # 3) adx check (need adx array passed)
            if adx[i] < min_adx:
                n_not_trend += 1
                continue

            # 4) direction confidence stricter
            pred_dir_class = regime_direction[i]
            dir_confidence = direction_proba[i, pred_dir_class]
            if dir_confidence < min_dir_conf:
                n_low_dir_conf += 1
                continue
            if not (np.all(prev_signs > 0) or np.all(prev_signs < 0)):
                n_weak_trigger += 1
                continue

            # ADX check
            if adx[i] < min_adx:
                n_not_trend += 1
                continue

            # BUY
            if (pred[i] > threshold) and (regime_direction[i] == 1) and above_ema:
                signals[i] = 1
                strength = np.clip((pred[i] - threshold) / threshold, 0, 1)
                computed_size = (
                    strength
                    * confidence[i]
                    * dir_confidence
                    * vol_multiplier[i]
                    * max_size
                )
                sizes[i] = np.clip(computed_size, min_size, max_size)
                cooldown = cooldown_period

            # SELL
            elif (pred[i] < -threshold) and (regime_direction[i] == 0) and below_ema:
                signals[i] = -1
                strength = np.clip((-threshold - pred[i]) / threshold, 0, 1)
                computed_size = (
                    strength
                    * confidence[i]
                    * dir_confidence
                    * vol_multiplier[i]
                    * max_size
                )
                sizes[i] = np.clip(computed_size, min_size, max_size)
                cooldown = cooldown_period
                past_trade_returns.append(0)
            else:
                if above_ema and (regime_direction[i] == 0):
                    n_wrong_ema += 1
                elif below_ema and (regime_direction[i] == 1):
                    n_wrong_ema += 1

        n_sig = np.sum(signals != 0)
        print(f"  Signals: {n_sig} ({n_sig/len(signals):.1%})")
        print(f"    BUY: {np.sum(signals==1)}, SELL: {np.sum(signals==-1)}")
        print(f"  Filters removed:")
        print(f"    Unreliable: {n_unreliable} ({n_unreliable/len(signals):.1%})")
        print(f"    High ATR: {n_high_atr} ({n_high_atr/len(signals):.1%})")
        print(f"    Not TREND: {n_not_trend} ({n_not_trend/len(signals):.1%})")
        print(f"    Weak trigger: {n_weak_trigger} ({n_weak_trigger/len(signals):.1%})")
        print(f"    ⭐ IN RANGE (TOXIC): {n_in_range} ({n_in_range/len(signals):.1%})")
        print(f"    Low dir conf: {n_low_dir_conf} ({n_low_dir_conf/len(signals):.1%})")

        return signals, sizes

    # def _print_stats(self, signals, sizes, filter_stats):
    #     """Print signal generation statistics"""
    #     n_sig = np.sum(signals != 0)
    #     n_total = len(signals)

    #     print(f"  Signals: {n_sig} ({n_sig/n_total:.1%})")
    #     print(f"    BUY: {np.sum(signals==1)}, SELL: {np.sum(signals==-1)}")
    #     print(f"  Filters removed:")
    #     print(f"    Unreliable: {filter_stats['unreliable']} ({filter_stats['unreliable']/n_total:.1%})")
    #     print(f"    High ATR: {filter_stats['high_atr']} ({filter_stats['high_atr']/n_total:.1%})")
    #     print(f"    Not TREND: {filter_stats['not_trend']} ({filter_stats['not_trend']/n_total:.1%})")
    #     print(f"    Weak trigger: {filter_stats['weak_trigger']} ({filter_stats['weak_trigger']/n_total:.1%})")
    #     print(f"    ⭐ IN RANGE (TOXIC): {filter_stats['in_range']} ({filter_stats['in_range']/n_total:.1%})")
    #     print(f"    Low dir conf: {filter_stats['low_dir_conf']} ({filter_stats['low_dir_conf']/n_total:.1%})")
