"""
model_training.py — Dual-Horizon Ensemble & Meta-Model Training Logic
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config import config
from redis_cache import get_enhanced_data
from globals import logger, trading_state
from collections import deque

class DualHorizonEnsembleModel:
    def __init__(self):
        self.short_term_models = {}
        self.medium_term_models = {}
        self.meta_model = None
        self.feature_importance = {}
        self.scaler_short = StandardScaler()
        self.scaler_medium = StandardScaler()
        self.explainer = None
        self.trade_id_counter = None
        self.evaluated_tickers = None
        self.last_evaluation = None
        self.recent_trades = None
        self.last_trade_time = None
        self.last_sentiment = None
        self.last_prediction = None
        self.sentiment_scores = None
        self.cooldown_timers = None
        self.prediction_memory = None
        self.sector_positions = {}
        self.hold_threshold = None
        self.watchlist = None
        self.volume_cache = None
        self.last_model_update = None
        self.last_trade_result = None
        self.trade_failures = None
        self.enabled = None
        self.last_heartbeat = None
        self.instance_id = None

    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }

    def retrain_meta_model(self):
        pass
        """Retrain meta model using logged trade data"""
        try:
            pass
            sheet = sheet_client.worksheet("MetaModelLog")
            data = pd.DataFrame(sheet.get_all_records())

            if len(data) < 50:
                logger.warning("⚠️ Not enough trade logs to train meta model.")
                return

            features = [
                "confidence", "volatility", "vwap_distance", "volume_spike",
                "kelly_fraction", "entry_hour", "cooldown_status"
            ]
            X = data[features]
            y = data["outcome"]

            X.fillna(0, inplace=True)

            meta_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
            meta_model.fit(X, y)

            self.meta_model = meta_model

            logger.deduped_log("info", f"✅ Meta model retrained on {len(data)} samples.")
        except Exception as e:
            pass
            logger.error(f"❌ Meta model retraining failed: {e}")

    def train_dual_horizon_ensemble(self, qualified_tickers: List[str], user_id) -> bool:
        """Train dual-horizon ensemble models with qualified tickers"""
        try:
            pass
            logger.deduped_log("info", f"🔄 Training dual-horizon ensemble with {len(qualified_tickers)} qualified tickers...")

            if len(qualified_tickers) < config.MIN_TICKERS_FOR_TRAINING:
                logger.error(f"❌ Insufficient qualified tickers for training: {len(qualified_tickers)} < {config.MIN_TICKERS_FOR_TRAINING}")
                return False

            # Collect training data
            short_features_list = []
            medium_features_list = []
            short_labels_list = []
            medium_labels_list = []

            for ticker in qualified_tickers[:config.MIN_TICKERS_FOR_TRAINING]:
                try:
                    pass
                    # Get training data
                    short_data = get_enhanced_data(ticker, limit=config.TRAINING_DATA_DAYS * 78)  # 78 5-min bars per day
                    medium_data = get_enhanced_data(ticker, limit=config.TRAINING_DATA_DAYS, timeframe=TimeFrame.Day)

                    if short_data is None or medium_data is None:
                        continue

                    # Extract features
                    short_features = self.extract_features(short_data)
                    medium_features = self.extract_features(medium_data)

                    if short_features is None or medium_features is None:
                        continue

                    # Generate labels (simplified - future returns > 0)
                    short_labels = (short_data['close'].shift(-1) > short_data['close']).astype(int)
                    medium_labels = (medium_data['close'].shift(-1) > medium_data['close']).astype(int)

                    # Align features and labels
                    short_features = short_features.iloc[:-1]  # Remove last row
                    medium_features = medium_features.iloc[:-1]
                    short_labels = short_labels.iloc[:-1]
                    medium_labels = medium_labels.iloc[:-1]

                    # Drop NaN values
                    short_valid = ~(short_features.isnull().any(axis=1) | short_labels.isnull())
                    medium_valid = ~(medium_features.isnull().any(axis=1) | medium_labels.isnull())

                    short_features = short_features[short_valid]
                    short_labels = short_labels[short_valid]
                    medium_features = medium_features[medium_valid]
                    medium_labels = medium_labels[medium_valid]

                    if len(short_features) > 50 and len(medium_features) > 20:
                        short_features_list.append(short_features)
                        medium_features_list.append(medium_features)
                        short_labels_list.append(short_labels)
                        medium_labels_list.append(medium_labels)

                        logger.deduped_log("info", f"✅ Training data collected for {ticker}: Short={len(short_features)}, Medium={len(medium_features)}")

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    pass
                    logger.error(f"❌ Failed to collect training data for {ticker}: {e}")
                    continue

            if not short_features_list or not medium_features_list:
                logger.error("❌ No valid training data collected")
                return False

            # Combine all training data
            combined_short_features = pd.concat(short_features_list, ignore_index=True)
            combined_medium_features = pd.concat(medium_features_list, ignore_index=True)
            combined_short_labels = pd.concat(short_labels_list, ignore_index=True)
            combined_medium_labels = pd.concat(medium_labels_list, ignore_index=True)

            logger.deduped_log("info", f"📊 Combined training data: Short={len(combined_short_features)}, Medium={len(combined_medium_features)}")

            # Scale features
            short_features_scaled = self.scaler_short.fit_transform(combined_short_features)
            medium_features_scaled = self.scaler_medium.fit_transform(combined_medium_features)

            short_features_scaled = pd.DataFrame(short_features_scaled, columns=combined_short_features.columns)
            medium_features_scaled = pd.DataFrame(medium_features_scaled, columns=combined_medium_features.columns)

            # Create base models
            base_models = self.create_base_models()

            # Train short-term models
            logger.deduped_log("info", "🔄 Training short-term models...")
            for name, model in base_models.items():
                try:
                    pass
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, short_features_scaled, combined_short_labels, cv=tscv, scoring='accuracy')

                    # Train on full dataset
                    model.fit(short_features_scaled, combined_short_labels)
                    self.short_term_models[name] = model

                    logger.deduped_log("info", f"✅ Short-term {name} trained - CV Score: {cv_scores.mean():.3f}")

                except Exception as e:
                    pass
                    logger.error(f"❌ Short-term {name} training failed: {e}")

            # Train medium-term models
            logger.deduped_log("info", "🔄 Training medium-term models...")
            for name, model in base_models.items():
                try:
                    pass
                    # Create new instance for medium-term
                    medium_model = self.create_base_models()[name]

                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(medium_model, medium_features_scaled, combined_medium_labels, cv=tscv, scoring='accuracy')

                    # Train on full dataset
                    medium_model.fit(medium_features_scaled, combined_medium_labels)
                    self.medium_term_models[name] = medium_model

                    logger.deduped_log("info", f"✅ Medium-term {name} trained - CV Score: {cv_scores.mean():.3f}")

                except Exception as e:
                    pass
                    logger.error(f"❌ Medium-term {name} training failed: {e}")

            # Train meta-model for ensemble combination
            self.train_meta_model(short_features_scaled, medium_features_scaled, combined_short_labels, combined_medium_labels)

            # Save models
            self.save_models()

            # Update training status
            _trained = True
            trading_state.last_training_time = datetime.now()

            logger.deduped_log("info", "✅ Dual-horizon ensemble trained successfully")
            return True

        except Exception as e:
            pass
            logger.error(f"❌ Dual-horizon ensemble training failed: {e}")
            return False

    def train_meta_model(self, short_features: pd.DataFrame, medium_features: pd.DataFrame,
                        short_labels: pd.Series, medium_labels: pd.Series):
        """Train meta-model for ensemble combination"""
        try:
            pass
            # Get base model predictions
            short_predictions = np.zeros((len(short_features), len(self.short_term_models)))
            medium_predictions = np.zeros((len(medium_features), len(self.medium_term_models)))

            for i, (name, model) in enumerate(self.short_term_models.items()):
                short_predictions[:, i] = model.predict_proba(short_features)[:, 1]

            for i, (name, model) in enumerate(self.medium_term_models.items()):
                medium_predictions[:, i] = model.predict_proba(medium_features)[:, 1]

            # Combine predictions (assuming same length for simplicity)
            min_length = min(len(short_predictions), len(medium_predictions))
            combined_predictions = np.hstack([
                short_predictions[:min_length],
                medium_predictions[:min_length]
            ])

            # Train meta-model
            self.meta_model = LogisticRegression(random_state=42)
            self.meta_model.fit(combined_predictions, short_labels[:min_length])

            logger.deduped_log("info", "✅ Meta-model trained successfully")

        except Exception as e:
            pass
            logger.error(f"❌ Meta-model training failed: {e}")

    def extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features for ensemble model"""
        try:
            pass
            feature_columns = [
                'rsi_14', 'macd', 'macd_histogram', 'stoch_k', 'adx',
                'mfi', 'bb_position', 'volume_ratio', 'price_momentum',
                'volatility_20', 'buying_pressure_ratio', 'smart_money_index',
                'vwap_deviation', 'price_vs_vwap'
            ]

            available_features = [col for col in feature_columns if col in df.columns]

            if not available_features:
                return None

            features = df[available_features].copy()
            features = features.dropna()

            return features

        except Exception as e:
            pass
            logger.error(f"❌ Feature extraction failed: {e}")
            return None

    def predict_dual_horizon(self, short_data: pd.DataFrame, medium_data: pd.DataFrame, user_id) -> Tuple[float, float, float]:
        """Make dual-horizon predictions"""
        try:
            pass
            if not self.short_term_models or not self.medium_term_models:
                logger.warning("⚠️ Models not trained")
                return 0.5, 0.5, 0.5

            # Extract and scale features
            short_features = self.extract_features(short_data)
            medium_features = self.extract_features(medium_data)

            if short_features is None or medium_features is None:
                return 0.5, 0.5, 0.5

            short_features_scaled = self.scaler_short.transform(short_features.tail(1))
            medium_features_scaled = self.scaler_medium.transform(medium_features.tail(1))

            # Get short-term predictions
            short_predictions = []
            for name, model in self.short_term_models.items():
                pred = model.predict_proba(short_features_scaled)[0, 1]
                short_predictions.append(pred)

            short_ensemble_pred = np.mean(short_predictions)

            # Get medium-term predictions
            medium_predictions = []
            for name, model in self.medium_term_models.items():
                pred = model.predict_proba(medium_features_scaled)[0, 1]
                medium_predictions.append(pred)

            medium_ensemble_pred = np.mean(medium_predictions)

            # Combine using meta-model if available
            if self.meta_model:
                combined_features = np.hstack([short_predictions, medium_predictions]).reshape(1, -1)
                meta_pred = self.meta_model.predict_proba(combined_features)[0, 1]
            else:
                # Weighted combination
                meta_pred = (short_ensemble_pred * config.SHORT_TERM_WEIGHT +
                        medium_ensemble_pred * config.MEDIUM_TERM_WEIGHT)

            return short_ensemble_pred, medium_ensemble_pred, meta_pred

        except Exception as e:
            pass
            logger.error(f"❌ Dual-horizon prediction failed: {e}")
            return 0.5, 0.5, 0.5

    def save_models(self):
        pass
        """Save trained models"""
        try:
            pass
            # Save ensemble models
            ensemble_data = {
                'short_term_models': self.short_term_models,
                'medium_term_models': self.medium_term_models,
                'meta_model': self.meta_model,
                'scaler_short': self.scaler_short,
                'scaler_medium': self.scaler_medium,
                'feature_importance': self.feature_importance
            }

            os.makedirs('models/ensemble', exist_ok=True)
            with open('models/ensemble/dual_horizon_ensemble.pkl', 'wb') as f:
                pass
                joblib.dump(ensemble_data, f)

            logger.deduped_log("info", "💾 Ensemble models saved successfully")

        except Exception as e:
            pass
            logger.error(f"❌ Model saving failed: {e}")

    def save_scaler(scaler, user_id: str, ticker: str, horizon: str = "short"):
        """Save scaler to disk with proper user/ticker/horizon key."""
        scaler_path = f"models/SCALER:{user_id}:{ticker}:{horizon}.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
    
    def load_models(self) -> bool:
        """Load trained models"""
        try:
            pass
            model_path = 'models/ensemble/dual_horizon_ensemble.pkl'
            if not os.path.exists(model_path):
                logger.warning("⚠️ Ensemble model file not found.")
                return False

            with open(model_path, 'rb') as f:
                pass
                ensemble_data = joblib.load(f)

            self.short_term_models = ensemble_data.get('short_term_models', {})
            self.medium_term_models = ensemble_data.get('medium_term_models', {})
            self.meta_model = ensemble_data.get('meta_model')
            self.scaler_short = ensemble_data.get('scaler_short', StandardScaler())
            self.scaler_medium = ensemble_data.get('scaler_medium', StandardScaler())
            self.feature_importance = ensemble_data.get('feature_importance', {})

            logger.deduped_log("info", "✅ Ensemble models loaded successfully")
            return True

        except Exception as e:
            pass
            logger.error(f"❌ Model loading failed: {e}")
            return False

import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from utils import get_training_data  # hypothetical function

def train_short_model(ticker: str, user_id: str) -> bool:
    try:
        X, y = get_training_data(ticker, horizon="short")
        if len(X) < 10:
            print(f"⚠️ Not enough data to train short model for {ticker}")
            return False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_scaled, y)

        save_model(model, user_id, ticker, "short")
        save_scaler(scaler, user_id, ticker, "short")

        print(f"✅ Trained and saved short model for {ticker} (user {user_id})")
        return True
    except Exception as e:
        print(f"❌ Short model training failed for {ticker}: {e}")
        return False

def train_medium_model(ticker: str, user_id: str) -> bool:
    try:
        X, y = get_training_data(ticker, horizon="medium")
        if len(X) < 10:
            print(f"⚠️ Not enough data to train medium model for {ticker}")
            return False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_scaled, y)

        save_model(model, user_id, ticker, "medium")
        save_scaler(scaler, user_id, ticker, "medium")

        print(f"✅ Trained and saved medium model for {ticker} (user {user_id})")
        return True
    except Exception as e:
        print(f"❌ Medium model training failed for {ticker}: {e}")
        return False

def train_all_models(self, tickers: List[str], user_id: str) -> bool:
    config.reload_user_settings()

    if not tickers:
        print("⚠️ No tickers in watchlist. Skipping model training.")
        return False

    logger.deduped_log("info", "🔁 Starting full model training (short + medium)...")

    success_short = all([train_short_model(ticker, user_id) for ticker in tickers])
    success_medium = all([train_medium_model(ticker, user_id) for ticker in tickers])

    return success_short and success_medium

    def get_top_features(self, user_id, n: int = 10) -> List[str]:
        """Get top N most important features"""
        return list(self.feature_importance.keys())[:n]

    # [All methods were previously verified — replaced here with placeholder to avoid size limits]
    # Add your methods here:
    # - create_base_models()
    # - train_dual_horizon_ensemble()
    # - train_meta_model()
    # - retrain_meta_model()
    # - extract_features()
    # - predict_dual_horizon()
    # - calculate_feature_importance()
    # - save_models()
    # - load_models()

def get_model(user_id: str, ticker: str, horizon: str = "short"):
    """Load model from disk based on user ID, ticker, and horizon."""
    import os
    import joblib

    model_path = f"models/MODEL:{user_id}:{ticker}:{horizon}.pkl"
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Model file not found: {model_path}")
        return None
    return joblib.load(model_path)

def get_scaler(user_id: str, ticker: str, horizon: str = "short"):
    """Load scaler from disk for a specific model."""
    import os
    import joblib

    scaler_path = f"models/SCALER:{user_id}:{ticker}:{horizon}.pkl"
    if not os.path.exists(scaler_path):
        logger.warning(f"⚠️ Scaler file not found: {scaler_path}")
        return None
    return joblib.load(scaler_path)

def predict_weighted_proba(models: dict, features_scaled, weights: dict = None) -> float:
    """
    Combine predictions from multiple models using weighted average.
    - `models`: dict of {name: model}
    - `features_scaled`: preprocessed features (e.g. from StandardScaler)
    - `weights`: dict of {name: weight}; defaults to equal weight
    """
    if not models:
        return 0.5  # Neutral default

    predictions = []
    total_weight = 0

    for name, model in models.items():
        try:
            proba = model.predict_proba(features_scaled)[0, 1]
            weight = weights.get(name, 1.0) if weights else 1.0
            predictions.append(proba * weight)
            total_weight += weight
        except Exception as e:
            print(f"⚠️ Model prediction failed for {name}: {e}")

    if total_weight == 0:
        return 0.5

    return sum(predictions) / total_weight

class MetaModelApprovalSystem:
    def __init__(self):
        self.min_accuracy = config.META_MODEL_MIN_ACCURACY
        self.min_trades = config.META_MODEL_MIN_TRADES
        self.approval_history = deque(maxlen=100)

        self.trade_id_counter = 0
        self.evaluated_tickers = {}
        self.last_evaluation = {}
        self.recent_trades = []
        self.last_trade_time = None
        self.last_sentiment = {}
        self.last_prediction = {}
        self.sentiment_scores = {}
        self.cooldown_timers = {}
        self.prediction_memory = {}
        self.sector_positions = {}
        self.hold_threshold = 0.5
        self.watchlist = []
        self.volume_cache = {}
        self.last_model_update = None
        self.last_trade_result = {}
        self.trade_failures = 0
        self.enabled = True
        self.last_heartbeat = None
        self.instance_id = "meta-model"

    def evaluate_model_performance(self) -> bool:
        """Evaluate if meta-model meets approval criteria"""
        try:
            if len(trading_state.trade_outcomes) < self.min_trades:
                logger.deduped_log("info", f"⏳ Insufficient trades for meta-model approval: {len(trading_state.trade_outcomes)}/{self.min_trades}")
                return False

            # Calculate recent accuracy
            recent_trades = trading_state.trade_outcomes[-50:]
            correct_predictions = sum(1 for trade in recent_trades if trade.get('correct_prediction', False))
            accuracy = correct_predictions / len(recent_trades) if recent_trades else 0.0

            # Update model accuracy tracking
            trading_state.model_accuracy['current'] = accuracy
            trading_state.meta_model_accuracy_history.append({
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'trade_count': len(recent_trades)
            })

            approved = accuracy >= self.min_accuracy

            # Additional risk criteria
            sharpe_ratio = trading_state.risk_metrics.get('sharpe_ratio', 0)
            drawdown = trading_state.risk_metrics.get('max_drawdown', 0)

            if approved and sharpe_ratio < config.SHARPE_RATIO_MIN:
                approved = False
                logger.warning(f"⚠️ Meta-model approval denied: Low Sharpe ratio {sharpe_ratio:.2f}")

            if approved and drawdown > config.MAX_DAILY_DRAWDOWN:
                approved = False
                logger.warning(f"⚠️ Meta-model approval denied: High drawdown {drawdown:.2%}")

            trading_state.meta_model_approved = approved

            self.approval_history.append({
                'timestamp': datetime.now(),
                'approved': approved,
                'accuracy': accuracy,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': drawdown
            })

            if approved:
                logger.deduped_log(f"✅ Meta-model approved - Accuracy: {accuracy:.3f}, Trades: {len(recent_trades)}")
            else:
                logger.warning(f"❌ Meta-model approval denied - Accuracy: {accuracy:.3f} < {self.min_accuracy:.3f}")

            return approved

        except Exception as e:
            pass
            logger.error(f"❌ Meta-model evaluation failed: {e}")
            return False

# === Global Instances ===
ensemble_model = DualHorizonEnsembleModel()
meta_approval_system = MetaModelApprovalSystem()

# === Exports ===
__all__ = [
    "train_all_models",
    "get_model",
    "get_scaler",
    "save_model",
    "save_scaler",
    "predict_weighted_proba",
    "DualHorizonEnsembleModel",
    "MetaModelApprovalSystem"
]
