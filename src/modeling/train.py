import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.health_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.fraud_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()

    def load_data(self):
        """Load processed data"""
        logger.info("Loading processed data...")
        data_path = os.path.join(self.processed_data_dir, 'processed_data.csv')
        df = pd.read_csv(data_path)
        return df

    def prepare_features(self, df):
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        # Select feature columns (excluding target variables)
        feature_columns = [col for col in df.columns 
                         if not col.startswith('is_') and col != 'symbol']
        
        # Scale features
        X = self.scaler.fit_transform(df[feature_columns])
        
        return X, feature_columns

    def train_health_model(self, X, y):
        """Train business health assessment model"""
        logger.info("Training business health assessment model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.health_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.health_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Business Health Model Report:\n{report}")
        
        return X_test, y_test, y_pred

    def train_fraud_model(self, X, y):
        """Train fraud detection model"""
        logger.info("Training fraud detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.fraud_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.fraud_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Fraud Detection Model Report:\n{report}")
        
        return X_test, y_test, y_pred

    def plot_feature_importance(self, model, feature_columns, title):
        """Plot feature importance"""
        logger.info(f"Plotting feature importance for {title}...")
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {title}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_columns[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.models_dir, f'feature_importance_{title.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {plot_path}")

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix"""
        logger.info(f"Plotting confusion matrix for {title}...")
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = os.path.join(self.models_dir, f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plot_path}")

    def save_models(self):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        # Save models using joblib
        import joblib
        
        health_model_path = os.path.join(self.models_dir, 'health_model.joblib')
        fraud_model_path = os.path.join(self.models_dir, 'fraud_model.joblib')
        
        joblib.dump(self.health_model, health_model_path)
        joblib.dump(self.fraud_model, fraud_model_path)
        
        logger.info(f"Saved models to {self.models_dir}")

    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting training pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, feature_columns = self.prepare_features(df)
        
        # Train and evaluate health model
        X_test_health, y_test_health, y_pred_health = self.train_health_model(
            X, df['is_healthy']
        )
        
        # Train and evaluate fraud model
        X_test_fraud, y_test_fraud, y_pred_fraud = self.train_fraud_model(
            X, df['is_fraud']
        )
        
        # Plot feature importance
        self.plot_feature_importance(
            self.health_model, 
            feature_columns, 
            "Business Health Assessment"
        )
        self.plot_feature_importance(
            self.fraud_model, 
            feature_columns, 
            "Fraud Detection"
        )
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            y_test_health, 
            y_pred_health, 
            "Business Health Assessment"
        )
        self.plot_confusion_matrix(
            y_test_fraud, 
            y_pred_fraud, 
            "Fraud Detection"
        )
        
        # Save models
        self.save_models()
        
        logger.info("Completed training pipeline")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training() 