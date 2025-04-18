import os
import pickle
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class ModelManager:
    def __init__(self, models_dir: str = "api/models"):
        self.models_dir = models_dir
        self.models: Dict[str, Dict[str, Any]] = {}
        self.load_models()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
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
            }
        }
        
        for model_id, model_info in models_info.items():
            file_path = model_info["file_path"]
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        model_info["model"] = pickle.load(f)
                    self.models[model_id] = model_info
                except Exception as e:
                    print(f"Error loading model {model_id}: {e}")
    
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
        """Preprocess text for prediction."""
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
        
        if model_info["type"] == "text":
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
            
        return result

# Create a singleton instance
model_manager = ModelManager()