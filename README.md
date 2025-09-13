# Brain Tumor Detection System

A comprehensive machine learning system for detecting brain tumors in medical images using deep learning techniques.

## 🧠 Overview

This project implements a state-of-the-art brain tumor detection system that can:
- Classify brain MRI images as tumor/no-tumor
- Provide interpretable results using Grad-CAM visualization
- Offer both web interface and API access
- Support multiple medical image formats (DICOM, NIfTI, JPEG, PNG)

## 🚀 Features

### Core Functionality
- **Deep Learning Model**: CNN-based classification with transfer learning
- **Image Preprocessing**: Automated preprocessing pipeline for medical images
- **Model Interpretability**: Grad-CAM visualization for explainable AI
- **Multiple Formats**: Support for DICOM, NIfTI, and standard image formats

### User Interfaces
- **Web Application**: User-friendly Streamlit interface for image upload and prediction
- **REST API**: RESTful API for programmatic access
- **Batch Processing**: Capability to process multiple images at once

### Development Features
- **Model Training**: Complete training pipeline with validation
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score
- **Visualization**: Training progress monitoring and result visualization
- **Testing**: Unit tests and model validation

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.12+
- See `requirements.txt` for complete dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaikishan1228/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Web Application
```bash
streamlit run src/web_app.py
```

### API Server
```bash
python src/api_server.py
```

### Training a Model
```bash
python src/train_model.py --data_path /path/to/dataset --epochs 50
```

### Batch Prediction
```bash
python src/predict.py --input_dir /path/to/images --output_dir /path/to/results
```

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── src/
│   ├── models/           # Model architectures
│   ├── data/            # Data processing utilities
│   ├── utils/           # Helper functions
│   ├── web_app.py       # Streamlit web interface
│   ├── api_server.py    # Flask API server
│   ├── train_model.py   # Training script
│   └── predict.py       # Prediction script
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── models/             # Saved model files
├── data/               # Dataset directory
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🔬 Model Architecture

The system uses a Convolutional Neural Network (CNN) with transfer learning:
- **Base Model**: Pre-trained models (ResNet50, VGG16, EfficientNet)
- **Custom Layers**: Additional layers for medical image classification
- **Input**: 224x224x3 RGB images (preprocessed from original medical images)
- **Output**: Binary classification (Tumor/No Tumor)

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification results

## 🎯 API Endpoints

### Prediction
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch image prediction

### Model Management
- `GET /model/info` - Model information
- `GET /health` - Health check

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.