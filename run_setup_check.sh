#!/bin/bash

# AI Trading Bot Setup Check Script
echo "🚀 AI Trading Bot - Setup Check"
echo "================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if virtual environment is recommended
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not detected. Recommended to use venv:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    echo ""
fi

# Install requirements
echo "📦 Installing requirements..."
if pip install -r requirements.txt; then
    echo "✅ Requirements installed successfully"
else
    echo "❌ Failed to install requirements"
    exit 1
fi

# Run Python setup check
echo "🔧 Running comprehensive setup check..."
if python3 setup_check.py; then
    echo ""
    echo "🎉 Setup check completed!"
    echo ""
    echo "Next steps:"
    echo "1. Set up your environment variables in .env file"
    echo "2. Configure your Alpaca API keys"
    echo "3. Run: python3 main.py"
else
    echo "❌ Setup check failed. Please fix the issues above."
    exit 1
fi
