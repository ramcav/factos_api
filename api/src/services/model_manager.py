import os
import pickle
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from sklearn.exceptions import NotFittedError
import numpy as np
import pathlib
# Import tensorflow and keras for loading the CNN model
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class ModelManager:
    def __init__(self, models_dir: str = None):
        # Use a more robust way to find the models directory
        if models_dir is None:
            # Get the directory of the current file
            current_file = pathlib.Path(__file__).resolve()
            # Navigate to project root (go up from src/services to api)
            project_root = current_file.parent.parent.parent
            self.models_dir = os.path.join(project_root, "models")
        else:
            self.models_dir = models_dir
            
        logger.info(f"Using models directory: {self.models_dir}")
        
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Create a fallback dummy model
        class DummyModel:
            def predict(self, X):
                # Always predict fake news (0)
                return np.zeros(len(X), dtype=int)
                
            def predict_proba(self, X):
                # Return probabilities with high confidence for fake news
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 0.9  # 90% confidence fake
                probs[:, 1] = 0.1  # 10% confidence real
                return probs
                
        self.dummy_model = DummyModel()
        
        # Load tokenizer for CNN model
        self.tokenizer = None
        self.max_length = 100  # Default max sequence length
        
        # Try to load tokenizer if it exists
        tokenizer_path = os.path.join(self.models_dir, "tokenizer.pickle")
        if os.path.exists(tokenizer_path):
            try:
                with open(tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Successfully loaded tokenizer")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {str(e)}")
        
        # Try to load max_length if it exists
        max_length_path = os.path.join(self.models_dir, "max_length.txt")
        if os.path.exists(max_length_path):
            try:
                with open(max_length_path, "r") as f:
                    self.max_length = int(f.read().strip())
                logger.info(f"Using max_length: {self.max_length}")
            except Exception as e:
                logger.error(f"Error loading max_length, using default: {str(e)}")
        
        self.load_models()
        
    def load_models(self) -> None:
        """Load all available models from the models directory."""
        models_info = {
            "1": {
                "id": 1,
                "name": "Logistic Regression",
                "description": "Logistic Regression model using text readability and sentiment features",
                "file_path": os.path.join(self.models_dir, "logistic_regression.pkl"),
                "type": "features",
                "required_features": [
                    "gunning_fog", "flesch_reading_ease", "compound_sentiment_score", 
                    "sentiment_score", "neutral_score", "negative_score", "positive_score",
                    "title_gunning_fog", "title_flesch_reading_ease", "title_compound_sentiment_score", 
                    "title_sentiment_score"
                ],
                "accuracy": 0.73
            },
            "2": {
                "id": 2,
                "name": "Random Forest",
                "description": "Random Forest model using TF-IDF features",
                "file_path": os.path.join(self.models_dir, "random_forest.pkl"),
                "type": "text",
                "accuracy": 0.99
            },
            "3": {
                "id": 3,
                "name": "Naive Bayes",
                "description": "Naive Bayes model using TF-IDF features",
                "file_path": os.path.join(self.models_dir, "naive_bayes.pkl"),
                "type": "text",
                "accuracy": 0.92
            },
            "4": {
                "id": 4,
                "name": "CNN",
                "description": "Deep Learning CNN model using word embeddings",
                "file_path": os.path.join(self.models_dir, "cnn.keras"),
                "type": "deep_learning",
                "accuracy": 0.999
            }
        }
        
        # First register all dummy models, then try to replace with real ones
        for model_id, model_info in models_info.items():
            # Register a dummy version first
            dummy_info = model_info.copy()
            dummy_info["model"] = self.dummy_model
            dummy_info["name"] += " (FALLBACK)"
            dummy_info["description"] += " - Using fallback model"
            self.models[model_id] = dummy_info
            
            # Now try to load the real model
            file_path = model_info["file_path"]
            if os.path.exists(file_path):
                try:
                    # Special handling for the CNN model
                    if model_info["type"] == "deep_learning":
                        if self.tokenizer is None:
                            logger.warning(f"Tokenizer not found for CNN model. Using fallback model.")
                            continue
                        
                        try:
                            loaded_model = keras.models.load_model(file_path)
                            logger.info(f"Successfully loaded CNN model: {model_info['name']}")
                            
                            # If successful, replace dummy with real model
                            model_info["model"] = loaded_model
                            self.models[model_id] = model_info
                        except Exception as e:
                            logger.error(f"CNN model failed to load: {str(e)}")
                    else:
                        # For scikit-learn models
                        with open(file_path, "rb") as f:
                            loaded_model = pickle.load(f)
                        
                        # For text models, test if they're properly fitted
                        if model_info["type"] == "text":
                            try:
                                # Test prediction on a simple string
                                _ = loaded_model.predict(["test"])
                                # If successful, replace dummy with real model
                                model_info["model"] = loaded_model
                                self.models[model_id] = model_info
                                logger.info(f"Successfully loaded model {model_id}: {model_info['name']}")
                            except Exception as e:
                                logger.error(f"Model {model_id} failed validation: {str(e)}")
                        else:
                            # For non-text models
                            model_info["model"] = loaded_model
                            self.models[model_id] = model_info
                            logger.info(f"Loaded model {model_id} without validation")
                            
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {str(e)}")
            else:
                logger.warning(f"Model file not found: {file_path}")
    
    def pad_sequences(self, sequences, max_length):
        """Pad sequences to the same length (utility for CNN model)"""
        padded_sequences = np.zeros((len(sequences), max_length))
        for i, seq in enumerate(sequences):
            if len(seq) > max_length:
                padded_sequences[i, :] = seq[:max_length]
            else:
                padded_sequences[i, :len(seq)] = seq
        return padded_sequences
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Return information about all available models."""
        return [
            {
                "id": info["id"],
                "name": info["name"],
                "description": info["description"],
                "type": info["type"],
                "accuracy": info["accuracy"]
            } 
            for info in self.models.values()
        ]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return information about a specific model."""
        if model_id in self.models:
            info = self.models[model_id]
            return {
                "id": info["id"],
                "name": info["name"],
                "description": info["description"],
                "type": info["type"],
                "accuracy": info["accuracy"],
                "input_format": "Plain text for prediction",
                "output_format": "Classification (0 for fake news, 1 for real news) with confidence score"
            }
        return None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for traditional ML models."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove stopwords
        words = [word for word in text.split() if word not in self.stop_words]
        
        # Remove non-alphabetic tokens
        words = [word for word in words if word.isalpha()]
        
        # Lemmatization
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Join words back into text
        return " ".join(words)
    
    def predict(self, model_id: str, text: str) -> Dict[str, Any]:
        """Use a model to predict if text is fake or real news."""
        if model_id not in self.models:
            return {"error": f"Model with ID {model_id} not found"}
        
        model_info = self.models[model_id]
        model = model_info["model"]
        
        try:
            if model_info["type"] == "deep_learning":
                # Check if tokenizer is available
                if self.tokenizer is None:
                    return {"error": "Tokenizer not available for CNN model"}
                
                # Preprocess text for CNN - fix the sequence handling
                sequences = self.tokenizer.texts_to_sequences([text])
                
                # The sequences is a list of lists - we need to handle it properly
                # Make sure we're properly padding the sequence
                if not sequences[0]:  # Handle empty sequence case
                    padded_data = np.zeros((1, self.max_length))
                else:
                    # Create a properly shaped input array
                    padded_data = np.zeros((1, self.max_length))
                    seq = sequences[0]
                    # Truncate if too long
                    if len(seq) > self.max_length:
                        padded_data[0, :] = seq[:self.max_length]
                    else:
                        # Pad if too short
                        padded_data[0, :len(seq)] = seq
                
                # Get prediction with the properly formatted input
                raw_prediction = model.predict(padded_data, verbose=0)[0][0]
                prediction = 1 if raw_prediction >= 0.5 else 0
                confidence = float(raw_prediction) if prediction == 1 else float(1 - raw_prediction)
                
                result = {
                    "prediction": prediction,
                    "prediction_label": "Real News" if prediction == 1 else "Fake News",
                    "confidence": confidence,
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "raw_score": float(raw_prediction)  # Include raw score for debugging
                }
                
            elif model_info["type"] == "text":
                # For TF-IDF based models (Random Forest and Naive Bayes)
                processed_text = self.preprocess_text(text)
                
                # These models contain the TF-IDF vectorizer in their pipeline
                prediction = model.predict([processed_text])[0]
                probabilities = model.predict_proba([processed_text])[0]
                
                result = {
                    "prediction": int(prediction),
                    "prediction_label": "Real News" if prediction == 1 else "Fake News",
                    "confidence": float(probabilities[1]) if prediction == 1 else float(probabilities[0]),
                    "model_id": model_info["id"],
                    "model_name": model_info["name"]
                }
                
            else:
                # For Logistic Regression model (needs features extraction)
                # This is a simplified version - real implementation would need to calculate all required features
                return {
                    "error": "Feature-based models require additional preprocessing not implemented in this demo API"
                }
                
        except Exception as e:
            logger.error(f"Error during prediction with model {model_id}: {str(e)}")
            return {"error": f"Model prediction failed: {str(e)}"}
            
        return result

# Create a singleton instance
model_manager = ModelManager()
