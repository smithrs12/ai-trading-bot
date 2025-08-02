from config import config
from trading_state import trading_state
from redis_cache import redis_cache
from key_utils import redis_key
import logger
from datetime import datetime

class MetaApprovalSystem:
    def __init__(self):
        self.performance_log = {}  # {ticker: [{timestamp, accuracy, samples}]}

    def evaluate_model_performance(self):
        """
        Evaluate all tickers and log those underperforming.
        """
        for ticker, accuracy in trading_state.model_accuracy.items():
            sample_count = trading_state.model_sample_count.get(ticker, 0)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "samples": sample_count
            }

            if ticker not in self.performance_log:
                self.performance_log[ticker] = []
            self.performance_log[ticker].append(entry)

            if accuracy < config.META_MODEL_MIN_ACCURACY or sample_count < config.META_MODEL_MIN_TRADES:
                logger.warning(f"⚠️ {ticker} flagged — accuracy: {accuracy:.2f}, samples: {sample_count}")
            else:
                logger.deduped_log("info", f"✅ {ticker} model approved — accuracy: {accuracy:.2f}, samples: {sample_count}")

            try:
                redis_cache.set(redis_key("MODEL_PERF", ticker), self.performance_log[ticker], ttl_seconds=86400)
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist performance log for {ticker}: {e}")

    def is_model_approved(self, ticker: str, proba: float) -> bool:
        """
        Approves or rejects a model's prediction based on past accuracy and confidence.
        """
        try:
            cache_key = redis_key("META_APPROVAL", ticker)
            cached = redis_cache.get(cache_key)
            if cached is not None:
                return cached

            accuracy = trading_state.model_accuracy.get(ticker, 0)
            sample_count = trading_state.model_sample_count.get(ticker, 0)

            approved = (
                accuracy >= config.META_MODEL_MIN_ACCURACY and
                sample_count >= config.META_MODEL_MIN_TRADES and
                proba >= config.ENSEMBLE_CONFIDENCE_THRESHOLD
            )

            redis_cache.set(cache_key, approved, ttl_seconds=300)

            if not approved:
                logger.deduped_log("warn", f"🚫 Meta-model rejected {ticker} — acc: {accuracy:.2f}, samples: {sample_count}, proba: {proba:.2f}")

            return approved
        except Exception as e:
            logger.warning(f"⚠️ Meta approval check failed for {ticker}: {e}")
            return False

        def should_execute_trade(self, prediction_confidence: float) -> bool:
        """Determine if trade should be executed based on meta-model approval"""
        try:
            pass
            # Check if meta-model is approved
            if not trading_state.meta_model_approved:
                logger.deduped_log("info", "⏸️ Trade blocked: Meta-model not approved")
                return False

            # Check confidence threshold
            if prediction_confidence < config.ENSEMBLE_CONFIDENCE_THRESHOLD:
                logger.deduped_log("info", f"⏸️ Trade blocked: Low confidence {prediction_confidence:.3f} < {config.ENSEMBLE_CONFIDENCE_THRESHOLD:.3f}")
                return False

            return True

        except Exception as e:
            pass
            logger.error(f"❌ Trade execution check failed: {e}")
            return False

meta_approval_system = MetaApprovalSystem()
from config import config
from trading_state import trading_state
from globals import redis_cache, redis_key
from logger import logger
from datetime import datetime

class MetaApprovalSystem:
    def __init__(self):
        self.performance_log = {}  # {ticker: [{timestamp, accuracy, samples}]}

    def evaluate_model_performance(self):
        """
        Evaluate all tickers and log those underperforming.
        """
        for ticker, accuracy in trading_state.model_accuracy.items():
            sample_count = trading_state.model_sample_count.get(ticker, 0)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "samples": sample_count
            }

            if ticker not in self.performance_log:
                self.performance_log[ticker] = []
            self.performance_log[ticker].append(entry)

            if accuracy < config.META_MODEL_MIN_ACCURACY or sample_count < config.META_MODEL_MIN_TRADES:
                logger.warning(f"⚠️ {ticker} flagged — accuracy: {accuracy:.2f}, samples: {sample_count}")
            else:
                logger.deduped_log("info", f"✅ {ticker} model approved — accuracy: {accuracy:.2f}, samples: {sample_count}")

            try:
                redis_cache.set(redis_key("MODEL_PERF", ticker), self.performance_log[ticker], ttl_seconds=86400)
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist performance log for {ticker}: {e}")

    def is_model_approved(self, ticker: str, proba: float) -> bool:
        """
        Approves or rejects a model's prediction based on past accuracy and confidence.
        """
        try:
            cache_key = redis_key("META_APPROVAL", ticker)
            cached = redis_cache.get(cache_key)
            if cached is not None:
                return cached

            accuracy = trading_state.model_accuracy.get(ticker, 0)
            sample_count = trading_state.model_sample_count.get(ticker, 0)

            approved = (
                accuracy >= config.META_MODEL_MIN_ACCURACY and
                sample_count >= config.META_MODEL_MIN_TRADES and
                proba >= config.ENSEMBLE_CONFIDENCE_THRESHOLD
            )

            redis_cache.set(cache_key, approved, ttl_seconds=300)

            if not approved:
                logger.deduped_log("warn", f"🚫 Meta-model rejected {ticker} — acc: {accuracy:.2f}, samples: {sample_count}, proba: {proba:.2f}")

            return approved
        except Exception as e:
            logger.warning(f"⚠️ Meta approval check failed for {ticker}: {e}")
            return False

    def should_execute_trade(self, prediction_confidence: float) -> bool:
        """Determine if trade should be executed based on meta-model approval"""
        try:
            pass
            # Check if meta-model is approved
            if not trading_state.meta_model_approved:
                logger.deduped_log("info", "⏸️ Trade blocked: Meta-model not approved")
                return False

            # Check confidence threshold
            if prediction_confidence < config.ENSEMBLE_CONFIDENCE_THRESHOLD:
                logger.deduped_log("info", f"⏸️ Trade blocked: Low confidence {prediction_confidence:.3f} < {config.ENSEMBLE_CONFIDENCE_THRESHOLD:.3f}")
                return False

            return True

        except Exception as e:
            pass
            logger.error(f"❌ Trade execution check failed: {e}")
            return False

meta_approval_system = MetaApprovalSystem()
