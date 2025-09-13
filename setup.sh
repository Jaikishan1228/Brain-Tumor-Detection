#!/bin/bash

# Brain Tumor Detection Setup Script
echo "🧠 Brain Tumor Detection System Setup"
echo "====================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
echo "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION found"
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION found"
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Python version is 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/MacOS
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/{tumor,no_tumor}
mkdir -p models
mkdir -p logs
mkdir -p temp
echo "✅ Directories created"

# Create data placeholder files
echo "Creating placeholder files..."
touch data/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
echo "✅ Placeholder files created"

# Check for GPU support
echo "Checking GPU support..."
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
if tf.test.is_gpu_available():
    print('✅ GPU available')
    print('GPU devices:', tf.config.list_physical_devices('GPU'))
else:
    print('⚠️  GPU not available, using CPU')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Add your dataset to the data directory:"
echo "   data/"
echo "   ├── tumor/"
echo "   │   ├── image1.jpg"
echo "   │   └── ..."
echo "   └── no_tumor/"
echo "       ├── image1.jpg"
echo "       └── ..."
echo ""
echo "3. Train a model:"
echo "   python src/train_model.py --data_path data --epochs 50"
echo ""
echo "4. Run the web application:"
echo "   streamlit run src/web_app.py"
echo ""
echo "5. Or start the API server:"
echo "   python src/api_server.py"
echo ""
echo "6. Run tests:"
echo "   python -m pytest tests/"
echo ""
echo "📖 For more information, see the README.md file"