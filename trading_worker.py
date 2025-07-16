# trading_worker.py

from main_user_isolated import main_loop
import os

user_id = os.getenv("USER_SESSION_ID", "background-worker")
main_loop(user_id)
