# trading_worker.py

import os
import time
import json
from datetime import datetime, timedelta

from config import config
from trading_state import trading_state
from main_user_isolated import market_status, redis_cache
from ensemble_model import ensemble_model
from meta_approval_system import meta_approval_system
from logger import deduped_log, logger
from technical_indicators import get_indicator_snapshot, passes_all_filters
from market_regime import regime_detector
import api_manager

# These need to exist in your modules:
from execution_manager import ultra_advanced_trading_logic, perform_eod_liquidation
from watchlist_optimizer import optimize_watchlist  # If separate module

def main_loop(user_id):
    """Main 24/7 trading loop with market awareness and resilience"""
    last_heartbeat = datetime.min
    last_watchlist_optimization = datetime.min
    last_model_retrain = datetime.min
    last_eod_check = datetime.now().date()
    loop_count = 0

    while True:
        try:
            current_time = datetime.now()
            loop_count += 1

            # === Market Status ===
            market_open = market_status.is_market_open()
            trading_window = market_status.is_in_trading_window()
            near_eod = market_status.is_near_eod()

            current_regime = regime_detector.detect_market_regime()
            if current_regime != getattr(trading_state, "market_regime", None):
                logger.deduped_log("info", f"🔄 Regime shift: {trading_state.market_regime} → {current_regime}")
            trading_state.market_regime = current_regime

            # === Periodic Heartbeat ===
            if (current_time - last_heartbeat).total_seconds() > 60:
                # heartbeat_monitor.send_heartbeat()  # Optional
                last_heartbeat = current_time

            logger.deduped_log("info", f"📊 Market: Open={market_open}, Window={trading_window}, NearEOD={near_eod}, Regime={current_regime}")

            # === Market Open Logic ===
            if market_open:
                # === Smart EOD Handling ===
                if near_eod and not trading_state.eod_liquidation_triggered:
                    perform_eod_liquidation()

                elif trading_window:
                    logger.deduped_log("info", "🚀 Executing intraday strategy...")

                    # === Dynamic Watchlist Refresh ===
                    if (current_time - last_watchlist_optimization).total_seconds() >= config.DYNAMIC_WATCHLIST_REFRESH_HOURS * 3600:
                        optimized = optimize_watchlist()
                        trading_state.qualified_watchlist = optimized
                        last_watchlist_optimization = current_time
                        logger.deduped_log("info", f"✅ Watchlist refreshed: {len(optimized)} tickers")

                    current_watchlist = trading_state.qualified_watchlist or trading_state.current_watchlist
                    meta_approval_system.evaluate_model_performance()

                    trades_executed = 0
                    for ticker in current_watchlist:
                        try:
                            snapshot = get_indicator_snapshot(ticker)
                            if snapshot is None or snapshot.empty:
                                continue

                            if not passes_all_filters(ticker, data=snapshot, regime=current_regime):
                                continue

                            if ultra_advanced_trading_logic(ticker):
                                trades_executed += 1

                            time.sleep(0.75)  # Avoid API bursts

                        except Exception as e:
                            logger.error(f"❌ Error trading {ticker}: {e}")
                            continue

                    logger.deduped_log("info", f"📈 Cycle finished: {trades_executed} trades")

                    # === Model Retraining Logic ===
                    if (current_time - last_model_retrain).total_seconds() > config.MODEL_RETRAIN_FREQUENCY_HOURS * 3600:
                        tickers_for_training = trading_state.qualified_watchlist or trading_state.current_watchlist[:config.MIN_TICKERS_FOR_TRAINING]
                        ensemble_model.train_dual_horizon_ensemble(tickers_for_training)
                        trading_state.models_trained = True
                        last_model_retrain = current_time
                        logger.deduped_log("info", "🔄 Ensemble retrained")

                else:
                    # === Window Not Open: Do Maintenance ===
                    logger.deduped_log("info", "🕓 Waiting for trading window...")
                    trading_state.update_ultra_advanced_risk_metrics()

                    # Expire stale sentiment cache
                    cutoff = current_time.timestamp() - 3600
                    trading_state.sentiment_cache = {
                        k: v for k, v in trading_state.sentiment_cache.items()
                        if isinstance(v, dict) and 'timestamp' in v and v['timestamp'].timestamp() > cutoff
                    }

            # === Market Closed Logic ===
            else:
                logger.deduped_log("info", "🌙 Market closed — performing overnight ops")

                if current_time.date() != last_eod_check:
                    trading_state.reset_daily()
                    last_eod_check = current_time.date()
                    logger.deduped_log("info", "🌄 Daily reset completed")

                if loop_count % 10 == 0:
                    try:
                        backup_data = {
                            'trade_outcomes': trading_state.trade_outcomes[-100:],
                            'risk_metrics': trading_state.risk_metrics,
                            'model_accuracy': trading_state.model_accuracy,
                            'watchlist_performance': trading_state.watchlist_performance,
                        }

                        os.makedirs("backups", exist_ok=True)
                        path = f"backups/backup_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                        with open(path, "w") as f:
                            json.dump(backup_data, f, indent=2, default=str)

                        logger.deduped_log("info", f"💾 Backup saved to: {path}")

                        qualified = trading_state.qualified_watchlist or trading_state.current_watchlist
                        if len(qualified) >= config.MIN_TICKERS_FOR_TRAINING:
                            ensemble_model.retrain_meta_model()
                            ensemble_model.train_dual_horizon_ensemble(qualified[:config.MIN_TICKERS_FOR_TRAINING])
                            logger.deduped_log("info", "✅ Overnight model retraining done.")
                        else:
                            logger.warning("⚠️ Insufficient tickers for retraining")

                    except Exception as e:
                        logger.error(f"❌ Backup or retrain failed: {e}")

                # === Sleep if far from open ===
                if market_status.get_time_until_market_open().total_seconds() > 3600:
                    logger.deduped_log("info", "😴 Sleeping 5 min...")
                    time.sleep(300)
                    continue

            # === Poll Every 30s ===
            for _ in range(30):
                time.sleep(1)

        except KeyboardInterrupt:
            logger.deduped_log("info", "🛑 Graceful shutdown triggered")
            break
        except Exception as e:
            logger.error(f"❌ Fatal error in main loop: {e}")

# === ENTRYPOINT ===
if __name__ == "__main__":
    user_id = os.getenv("USER_SESSION_ID", "background-worker")
    main_loop(user_id)
