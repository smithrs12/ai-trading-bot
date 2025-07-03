#!/bin/bash

echo "🚀 AI Trading Bot Setup Check"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Run setup check
echo "🔧 Running setup validation..."
python3 setup_check.py

echo "✅ Setup check complete!"
