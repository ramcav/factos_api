#!/usr/bin/env python3
"""
Logistic Regression model for fake news detection.
This script handles the entire pipeline from loading data to saving the trained model.
"""

import os
import sys
import pickle
import json
import datetime
import pandas as pd
import numpy as np
import nltk
import kagglehub
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure required NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Add this class at the module level (outside of any function)
class FeatureSelector:
    """
    Transformer that selects specified features from a DataFrame.
    This follows scikit-learn transformer interface.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Select only the needed features
            return X[self.feature_names]
        return X
        
    def fit(self, X, y=None):
        return self
    
    # These are needed for proper pickling in scikit-learn
    def get_params(self, deep=True):
        return {"feature_names": self.feature_names}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def load_data():
    """Load and merge the fake news datasets"""
    print("Loading datasets...")
    
    # Download latest version from Kaggle
    try:
        path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets") + "/News _dataset"
        print(f"Path to dataset files: {path}")
        
        fake_ds = pd.read_csv(path + "/Fake.csv")
        true_ds = pd.read_csv(path + "/True.csv")
        
        # Add labels
        fake_ds['label'] = 0
        true_ds['label'] = 1
        
        # Merge datasets
        dataset = pd.concat([fake_ds, true_ds])
        return dataset
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def extract_features(dataset):
    """
    Extract numerical features from text as used in the notebook:
    - readability metrics
    - sentiment analysis
    """
    print("Extracting features...")
    
    # Calculate readability metrics
    print("- Calculating readability metrics...")
    dataset['gunning_fog'] = dataset['text'].apply(lambda x: textstat.gunning_fog(x))
    dataset['flesch_reading_ease'] = dataset['text'].apply(lambda x: textstat.flesch_reading_ease(x))
    dataset['title_gunning_fog'] = dataset['title'].apply(lambda x: textstat.gunning_fog(x))
    dataset['title_flesch_reading_ease'] = dataset['title'].apply(lambda x: textstat.flesch_reading_ease(x))
    
    # Calculate sentiment metrics
    print("- Calculating sentiment metrics...")
    sia = SentimentIntensityAnalyzer()
    dataset['compound_sentiment_score'] = dataset['text'].apply(
        lambda x: sia.polarity_scores(x)['compound'])
    dataset['sentiment_score'] = dataset['text'].apply(
        lambda x: sia.polarity_scores(x)['pos'] - sia.polarity_scores(x)['neg'])
    dataset['neutral_score'] = dataset['text'].apply(
        lambda x: sia.polarity_scores(x)['neu'])
    dataset['negative_score'] = dataset['text'].apply(
        lambda x: sia.polarity_scores(x)['neg'])
    dataset['positive_score'] = dataset['text'].apply(
        lambda x: sia.polarity_scores(x)['pos'])
    
    dataset['title_compound_sentiment_score'] = dataset['title'].apply(
        lambda x: sia.polarity_scores(x)['compound'])
    dataset['title_sentiment_score'] = dataset['title'].apply(
        lambda x: sia.polarity_scores(x)['pos'] - sia.polarity_scores(x)['neg'])
    
    return dataset

def train_logistic_regression_model(dataset):
    """Train the Logistic Regression model using numerical features"""
    print("Training Logistic Regression model...")
    
    # Drop any rows with missing values
    dataset = dataset.dropna()
    
    # Use only numerical features as in the notebook
    features = dataset[[
        'gunning_fog', 'flesch_reading_ease',
        'compound_sentiment_score', 'sentiment_score',
        'neutral_score', 'negative_score', 'positive_score',
        'title_gunning_fog', 'title_flesch_reading_ease',
        'title_compound_sentiment_score', 'title_sentiment_score'
    ]]
    
    y = dataset['label']
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = lr_model.predict(X_test)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature names for the selector
    feature_names = features.columns.tolist()
    
    # Create the pipeline using the module-level FeatureSelector class
    lr_pipeline = Pipeline([
        ('features', FeatureSelector(feature_names)),
        ('clf', lr_model)
    ])
    
    # Test with a sample row
    sample = dataset.iloc[0:1]
    test_pred = lr_pipeline.predict(sample)
    print(f"Pipeline test prediction: {test_pred}")
    
    return lr_pipeline, classification_report(y_test, y_pred, output_dict=True), feature_names

def save_model(model, model_name, metrics=None, feature_names=None):
    """Save the model and its metadata"""
    print(f"Saving {model_name} model...")
    
    model_dir = "../api/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Define file paths
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "model_type": "LogisticRegression",
        "description": "Logistic Regression model for fake news detection using numerical features",
        "features_used": feature_names if feature_names else "Unknown",
        "saved_on": datetime.datetime.now().isoformat(),
        "saved_by": os.getenv("USER", "unknown"),
    }
    
    # Add pipeline components to metadata
    if isinstance(model, Pipeline):
        metadata["pipeline_steps"] = [
            {"name": name, "type": type(step).__name__}
            for name, step in model.named_steps.items()
        ]
        
        # Add Logistic Regression specific details
        if "clf" in model.named_steps and isinstance(model.named_steps["clf"], LogisticRegression):
            lr = model.named_steps["clf"]
            metadata.update({
                "max_iter": lr.max_iter,
                "C": lr.C,
                "random_state": lr.random_state if lr.random_state is not None else "None"
            })
    
    # Add performance metrics if available
    if metrics:
        metadata["performance"] = metrics
    
    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    """Main function to orchestrate the model building and saving process"""
    # Load data
    dataset = load_data()
    print(f"Loaded dataset with {len(dataset)} rows")
    
    # Extract features
    processed_data = extract_features(dataset)
    print(f"Processed data: {processed_data.shape}")
    
    # Train model
    model, metrics, feature_names = train_logistic_regression_model(processed_data)
    
    # Save model
    save_model(model, "logistic_regression", metrics, feature_names)
    
    print("\nLogistic Regression model processing complete!")

if __name__ == "__main__":
    main()