#!/usr/bin/env python3
"""
Naive Bayes model for fake news detection.
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure required NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

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

def preprocess_data(dataset):
    """Preprocess the text data"""
    print("Preprocessing data...")
    
    # Drop rows with missing values
    dataset = dataset.dropna()
    
    # First tokenize
    dataset['text_tokens'] = dataset['text'].apply(word_tokenize)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    dataset['text_tokens'] = dataset['text_tokens'].apply(
        lambda tokens: [word.lower() for word in tokens if word.lower() not in stop_words]
    )
    
    # Remove non-alphabetic tokens
    dataset['text_tokens'] = dataset['text_tokens'].apply(
        lambda tokens: [word for word in tokens if word.isalpha()]
    )
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    dataset['text_tokens'] = dataset['text_tokens'].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
    )
    
    # For display or further text-based processing, join back to strings
    dataset['text_clean'] = dataset['text_tokens'].apply(lambda tokens: ' '.join(tokens))
    
    return dataset

def train_naive_bayes_model(dataset):
    """Train the Naive Bayes model"""
    print("Training Naive Bayes model...")
    
    # Create and fit the TF-IDF vectorizer 
    tfidf = TfidfVectorizer(max_features=1000)  # Using 1000 features as in the notebook
    X_text = tfidf.fit_transform(dataset['text_clean'])
    y = dataset['label']
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )
    
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = nb_model.predict(X_test)
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create pipeline with fitted components
    nb_pipeline = Pipeline([
        ('tfidf', tfidf),  # Use the fitted vectorizer
        ('clf', nb_model)
    ])
    
    # Test the pipeline
    test_pred = nb_pipeline.predict(["test article"])
    print(f"Pipeline test prediction: {test_pred}")
    
    return nb_pipeline, classification_report(y_test, y_pred, output_dict=True)

def save_model(model, model_name, metrics=None):
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
        "model_type": "NaiveBayes",
        "description": "Multinomial Naive Bayes model for fake news detection",
        "features_used": "text content (TF-IDF)",
        "saved_on": datetime.datetime.now().isoformat(),
        "saved_by": os.getenv("USER", "unknown"),
    }
    
    # Add pipeline components to metadata
    if isinstance(model, Pipeline):
        metadata["pipeline_steps"] = [
            {"name": name, "type": type(step).__name__}
            for name, step in model.named_steps.items()
        ]
        
        # Add vectorizer details
        if "tfidf" in model.named_steps and isinstance(model.named_steps["tfidf"], TfidfVectorizer):
            tfidf = model.named_steps["tfidf"]
            metadata["tfidf_max_features"] = tfidf.max_features
            metadata["vocabulary_size"] = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0
    
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
    
    # Preprocess data
    processed_data = preprocess_data(dataset)
    print(f"Preprocessed data: {processed_data.shape}")
    
    # Train model
    model, metrics = train_naive_bayes_model(processed_data)
    
    # Save model
    save_model(model, "naive_bayes", metrics)
    
    print("\nNaive Bayes model processing complete!")

if __name__ == "__main__":
    main()