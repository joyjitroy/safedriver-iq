#!/bin/bash
# Quick Start Script for SafeDriver-IQ Project

echo "=========================================="
echo "SafeDriver-IQ: Setup & Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
if ! python -c "import pandas" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Available Commands:"
echo ""
echo "1. Start Jupyter Notebook:"
echo "   jupyter notebook notebooks/01_data_exploration.ipynb"
echo ""
echo "2. Run data loader test:"
echo "   python src/data_loader.py"
echo ""
echo "3. Start Streamlit dashboard (after model training):"
echo "   streamlit run app/streamlit_app.py"
echo ""
echo "4. Run tests:"
echo "   pytest tests/"
echo ""
echo "=========================================="
echo ""
echo "To get started, run:"
echo "  source venv/bin/activate"
echo "  jupyter notebook"
echo ""
