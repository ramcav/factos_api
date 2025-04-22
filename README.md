# 🔍 Factos API - Fake News Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/) [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A robust REST API and interactive web interface for fake news detection using state-of-the-art machine learning models.

## 🌟 Features

- **Multiple ML Models**: Choose from various pre-trained models:
  - Logistic Regression (with text readability and sentiment features)
  - Random Forest (using TF-IDF features)
  - Naive Bayes (using TF-IDF features)
- **REST API**: Built with FastAPI for high performance and easy integration
- **Interactive UI**: User-friendly Streamlit interface for testing models
- **Real-time Analysis**: Get instant predictions with confidence scores
- **Model Insights**: View detailed model information and performance metrics
- **Docker Support**: Easy deployment with Docker containers

## 🚀 Getting Started

You can run the application either using Docker or directly on your machine.

### 🐳 Using Docker (Recommended)

1. **Prerequisites**
   - Docker
   - Docker Compose

2. **Running the Application**
   ```bash
   # Clone the repository
   git clone https://github.com/ramcav/factos_api.git
   cd factos_api

   # Start the services
   docker-compose up -d

   # View logs (optional)
   docker-compose logs -f
   ```

   The services will be available at:
   - API: http://localhost:8002
   - Streamlit UI: http://localhost:8501

3. **Stopping the Application**
   ```bash
   docker-compose down
   ```

### 💻 Local Installation

There are two ways to run the application locally: using command-line instructions or using the provided shell scripts.

#### Option 1: Command Line Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ramcav/factos_api.git
   cd factos_api
   ```

2. **Set up and activate virtual environment**

   ```bash
   # Create virtual environments for both services

   # For API
   cd api
   python -m venv venv

   # Activate virtual environment

   # On Unix/macOS:
   source venv/bin/activate

   # On Windows:
   .\venv\Scripts\activate

   # Install API dependencies
   pip install -r requirements.txt

   # Start the API server
   uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload
   ```

   In a new terminal:

   ```bash
   # For Streamlit app
   cd demo
   python -m venv venv

   # Activate virtual environment
   # On Unix/macOS:
   source venv/bin/activate

   # On Windows:
   .\venv\Scripts\activate

   # Install Streamlit app dependencies
   pip install -r requirements.txt

   # Start the Streamlit app
   streamlit run app.py
   ```

The services will be available at:

- API: http://localhost:8002
- Streamlit UI: http://localhost:8501

To deactivate the virtual environments when done:

```bash
deactivate
```

#### Option 2: Using Shell Scripts

This will create a virtual environment, install dependencies and start the services.

1. **Clone the repository**

   ```bash
   git clone https://github.com/ramcav/factos_api.git
   cd factos_api
   ```

2. **Make the scripts executable**

   ```bash
   chmod +x api/run.sh
   chmod +x demo/run.sh
   ```

3. **Start the services**

   In the first terminal:

   ```bash
   cd api
   ./run.sh
   ```

   In a second terminal:

   ```bash
   cd demo
   ./run.sh
   ```

The services will be available at:

- API: http://localhost:8002
- Streamlit UI: http://localhost:8501

#### Stopping the Application

For both methods:
1. Press `Ctrl+C` in each terminal to stop the services
2. If using virtual environments, run `deactivate` to exit them

## 🎯 Usage

### Via Web Interface

1. Open the Streamlit interface in your browser
2. Select a model from the sidebar
3. Enter or paste the text you want to analyze
4. Click "Analyze Text" to get the prediction
5. View results including:
   - Prediction (Fake/Real News)
   - Confidence Score
   - Model Details
   - Full Analysis Results

### Via API

The API provides the following endpoints:

```bash
GET /api/models
# List all available models

GET /api/models/{model_id}
# Get detailed information about a specific model

POST /api/models/{model_id}/predict
# Get prediction for input text
```

Example API request:
```bash
curl -X POST "http://localhost:8002/api/models/1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here"}'
```

## 🛠️ Architecture

- **API Layer**: FastAPI for robust REST endpoints
- **ML Models**: Scikit-learn based models for text classification
- **Frontend**: Streamlit for interactive user interface
- **Data Processing**: NLTK for text preprocessing and feature extraction

## 🔄 Environment Variables

The application supports the following environment variables:

- `API_URL`: API endpoint URL (default: "http://localhost:8002")

## 📊 Model Performance

| Model | Accuracy | Features |
|-------|----------|----------|
| Logistic Regression | 73% | Text readability, sentiment |
| Random Forest | 99% | TF-IDF |
| Naive Bayes | 92% | TF-IDF |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the amazing web framework
- Streamlit for the intuitive UI components
- Scikit-learn for machine learning tools
- NLTK for natural language processing capabilities
