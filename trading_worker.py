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
    """Main 24/7 trading loop with market awareness"""
    last_heartbeat = datetime.min
    last_watchlist_optimization = datetime.min
    last_model_retrain = datetime.min
    last_eod_check = datetime.now().date()
    loop_count = 0

    while True:
        try:
            current_time = datetime.now()
            loop_count += 1

            # Always update status and regime
            market_open = market_status.is_market_open()
            trading_window = market_status.is_in_trading_window()
            near_eod = market_status.is_near_eod()

            last_regime = getattr(trading_state, "market_regime", None)
            current_regime = regime_detector.detect_market_regime()

            if current_regime != last_regime:
                logger.deduped_log("info", f"🔄 Market regime changed: {last_regime or 'unknown'} → {current_regime}")

            trading_state.market_regime = current_regime

            # Optional heartbeat
            if (current_time - last_heartbeat).total_seconds() >= 60:
                # heartbeat_monitor.send_heartbeat()  # Optional module
                last_heartbeat = current_time

            logger.deduped_log("info", f"📊 Market Status: Open={market_open}, Trading Window={trading_window}, Near EOD={near_eod}")

            if market_open:
                if near_eod and not trading_state.eod_liquidation_triggered:
                    perform_eod_liquidation()

                elif trading_window:
                    logger.deduped_log("info", "📈 In trading window - executing trading logic")

                    if (current_time - last_watchlist_optimization).total_seconds() >= config.DYNAMIC_WATCHLIST_REFRESH_HOURS * 3600:
                        optimized_watchlist = optimize_watchlist()
                        trading_state.qualified_watchlist = optimized_watchlist
                        last_watchlist_optimization = current_time
                        logger.deduped_log("info", f"✅ Watchlist optimized: {len(optimized_watchlist)} tickers")

                    current_watchlist = trading_state.qualified_watchlist or trading_state.current_watchlist
                    meta_approval_system.evaluate_model_performance()

                    trades_executed = 0
                    for ticker in current_watchlist:
                        try:
                            features_df = get_indicator_snapshot(ticker)
                            if features_df is None or features_df.empty:
                                continue

                            current_regime = getattr(trading_state, "market_regime", "neutral")

                            if not passes_all_filters(ticker, data=features_df, regime=current_regime):
                                continue

                            if ultra_advanced_trading_logic(ticker):
                                trades_executed += 1

                            time.sleep(1)

                        except Exception as e:
                            logger.error(f"❌ Error processing {ticker}: {e}")
                            continue

                    logger.deduped_log("info", f"📊 Trading cycle complete: {trades_executed} trades executed")

                    if (current_time - last_model_retrain).total_seconds() >= config.MODEL_RETRAIN_FREQUENCY_HOURS * 3600:
                        logger.deduped_log("info", "🔄 Periodic model retraining...")
                        qualified_tickers = trading_state.qualified_watchlist or trading_state.current_watchlist[:config.MIN_TICKERS_FOR_TRAINING]
                        ensemble_model.train_dual_horizon_ensemble(qualified_tickers)
                        trading_state.models_trained = True
                        last_model_retrain = current_time

                else:
                    time_until_trading = market_status.get_time_until_market_open()
                    logger.deduped_log("info", f"⏰ Waiting for trading window to open: {time_until_trading}")

                    logger.deduped_log("info", "🔧 Performing maintenance tasks...")
                    trading_state.update_ultra_advanced_risk_metrics()

                    # Expire old sentiment cache entries
                    now_ts = current_time.timestamp()
                    for ticker, cache in list(trading_state.sentiment_cache.items()):
                        if isinstance(cache, dict) and 'timestamp' in cache:
                            if (now_ts - cache['timestamp'].timestamp()) > 3600:
                                del trading_state.sentiment_cache[ticker]

            else:
                logger.deduped_log("info", "🌙 Market closed - performing overnight maintenance")

                if current_time.date() != last_eod_check:
                    trading_state.reset_daily()
                    last_eod_check = current_time.date()
                    logger.deduped_log("info", "🌅 New trading day - state reset")

                if loop_count % 10 == 0:
                    logger.deduped_log("info", "🔧 Performing extended maintenance...")
                    try:
                        backup_data = {
                            'trade_outcomes': trading_state.trade_outcomes[-100:],
                            'risk_metrics': trading_state.risk_metrics,
                            'model_accuracy': trading_state.model_accuracy,
                            'watchlist_performance': trading_state.watchlist_performance
                        }

                        os.makedirs('backups', exist_ok=True)
                        backup_filename = f"backups/backup_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                        with open(backup_filename, 'w') as f:
                            json.dump(backup_data, f, indent=2, default=str)

                        logger.deduped_log("info", f"💾 Backup complete: {backup_filename}")

                        qualified = trading_state.qualified_watchlist or trading_state.current_watchlist
                        if len(qualified) >= config.MIN_TICKERS_FOR_TRAINING:
                            ensemble_model.retrain_meta_model()
                            ensemble_model.train_dual_horizon_ensemble(qualified[:config.MIN_TICKERS_FOR_TRAINING])
                            logger.deduped_log("info", "✅ Daily model retraining completed.")
                        else:
                            logger.warning(f"⚠️ Not enough tickers to retrain: {len(qualified)}")
                    except Exception as e:
                        logger.error(f"❌ Maintenance or retraining failed: {e}")

                time_until_open = market_status.get_time_until_market_open()
                if time_until_open.total_seconds() > 3600:
                    logger.deduped_log("info", f"⏰ Sleeping 5 minutes until market opens: {time_until_open}")
                    time.sleep(300)
                    continue

            for _ in range(30):
                time.sleep(1)

        except KeyboardInterrupt:
            logger.deduped_log("info", "🛑 Shutdown requested by user")
            break
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")

# === ENTRYPOINT ===
if __name__ == "__main__":
    user_id = os.getenv("USER_SESSION_ID", "background-worker")
    main_loop(user_id)
