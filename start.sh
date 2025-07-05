#!/bin/bash

MODE=${MODE:-main}

if [[ "$MODE" == "main" ]]; then
    exec python main.py
elif [[ "$MODE" == "worker" ]]; then
    exec python worker.py
elif [[ "$MODE" == "dashboard" ]]; then
    exec streamlit run dashboard.py --server.port=5000 --server.enableCORS=false
else
    echo "Unknown mode: $MODE"
    exit 1
fi
