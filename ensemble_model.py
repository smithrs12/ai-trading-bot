import random
from config import config
from trading_state import trading_state
from main_user_isolated import redis_cache, redis_key
import logger

class EnsembleModel:
    def __init__(self):
        self.models = {}

    def train_dual_horizon_ensemble(self, tickers):
        """
        Simulates training multiple models per ticker (short + medium term).
        """
        print(f"📈 Training ensemble on {len(tickers)} tickers (Dual Horizon)")
        self.models = {
            ticker: {
                "short_term": f"short_model_{ticker}",
                "medium_term": f"medium_model_{ticker}"
            }
            for ticker in tickers
        }

        for ticker in tickers:
            acc = round(random.uniform(0.60, 0.95), 4)
            trading_state.model_accuracy[ticker] = acc
            trading_state.model_sample_count[ticker] = random.randint(100, 500)

        logger.deduped_log("info", f"✅ Ensemble models trained for {len(tickers)} tickers")

    def predict_weighted_proba(self, ticker: str) -> float:
        """
        Combines short- and medium-term model confidence scores.
        Caches prediction per ticker.
        """
        try:
            cache_key = redis_key("PREDICT_PROBA", ticker)
            cached = redis_cache.get(cache_key)
            if cached is not None:
                return cached

            if ticker not in self.models:
                logger.logger.warning(f"⚠️ No ensemble model found for {ticker}")
                return 0.5

            # Simulated probabilities
            short_conf = round(random.uniform(0.5, 0.95), 4)
            medium_conf = round(random.uniform(0.5, 0.95), 4)

            # Store for transparency
            trading_state.model_confidence_snapshot[ticker] = {
                "short_term": short_conf,
                "medium_term": medium_conf,
                "timestamp": config.get_now_str()
            }

            weighted_conf = round((short_conf * 0.6 + medium_conf * 0.4), 4)
            redis_cache.set(cache_key, weighted_conf, ttl_seconds=300)

            logger.deduped_log("debug", f"📊 {ticker} | Short: {short_conf:.2f}, Medium: {medium_conf:.2f}, Weighted: {weighted_conf:.2f}")
            return weighted_conf
        except Exception as e:
            logger.logger.error(f"❌ predict_weighted_proba failed for {ticker}: {e}")
            return 0.5

    def retrain_meta_model(self):
        """
        Placeholder logic for meta-model retraining.
        """
        print("🔄 Retraining meta model (placeholder)")
        logger.deduped_log("info", "🔁 Meta model retraining complete")

ensemble_model = EnsembleModel()
