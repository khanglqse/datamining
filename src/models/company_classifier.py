import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('company_classification.log')
    ]
)
logger = logging.getLogger(__name__)

class CompanyClassifier:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.results_dir = os.path.join(self.data_dir, 'results')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        
    def load_data(self):
        """Load and prepare data for classification"""
        try:
            # Load company clusters
            clusters_df = pd.read_csv(os.path.join(self.results_dir, 'company_clusters.csv'))
            
            # Map cluster numbers to labels
            cluster_labels = {
                0: 'Healthy',
                1: 'Normal',
                2: 'Risky'
            }
            clusters_df['label'] = clusters_df['0'].map(cluster_labels)
            clusters_df = clusters_df.rename(columns={'Unnamed: 0': 'company'})
            
            # Remove any duplicate companies
            clusters_df = clusters_df.drop_duplicates(subset=['company'])
            
            # Load financial data
            all_data = {}
            processed_files = glob.glob(os.path.join(self.processed_dir, 'processed_*.csv'))
            
            for file_path in processed_files:
                symbol = os.path.basename(file_path).split('_')[1].split('.')[0]
                if symbol in clusters_df['company'].values:
                    df = pd.read_csv(file_path, index_col=0)
                    
                    # Calculate average values for each metric across years
                    metrics = {}
                    for metric in df.index:
                        # Take the first set of years (2021-2024)
                        values = df.loc[metric].iloc[:4]
                        metrics[metric] = values.mean()
                    
                    all_data[symbol] = metrics
            
            # Convert to DataFrame
            features_df = pd.DataFrame(all_data).T
            
            # Select features for classification
            features_for_classification = [
                "roa",
                "roe",
                "gross_margin",
                "net_profit_margin",
                "revenue_growth",
                "net_income",
                "gross_profit",
                "debt_ratio",
                "current_ratio",
                "revenue"
            ]
            
            # Filter available features
            available_features = [f for f in features_for_classification if f in features_df.columns]
            features_df = features_df[available_features]
            
            # Merge with labels
            data = features_df.merge(clusters_df[['company', 'label']], 
                                   left_index=True, 
                                   right_on='company')
            
            logger.info(f"Loaded data with shape: {data.shape}")
            logger.info(f"Available features: {available_features}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def train_model(self, data):
        """Train the classification model"""
        try:
            # Prepare features and target
            X = data.drop(['company', 'label'], axis=1)
            y = data['label']
            
            # Print class distribution
            logger.info("\nClass distribution:")
            class_dist = y.value_counts()
            logger.info(class_dist)
            
            # Split data with a smaller test size due to small sample size
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with adjusted parameters for small sample size
            self.model = RandomForestClassifier(
                n_estimators=100,  # Increased number of trees
                max_depth=10,      # Increased depth for better feature separation
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            # Log detailed metrics
            logger.info("\nModel Evaluation Results:")
            
            # Calculate and log classification report
            logger.info("\nClassification Report:")
            report = classification_report(y_test, y_pred, digits=4)
            logger.info("\n" + report)
            
            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Create and save confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.model.classes_,
                       yticklabels=self.model.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.results_dir, 'confusion_matrix.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"\nConfusion Matrix plot saved to: {plot_path}")
            
            # Print confusion matrix in console
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + "="*50)
            logger.info("True vs Predicted (rows=true, columns=predicted):")
            logger.info("="*50)
            
            # Create header row
            header = " " * 10 + " | " + " | ".join([f"{label:^10}" for label in self.model.classes_])
            logger.info(header)
            logger.info("-" * len(header))
            
            # Create each row of the matrix
            for i, true_label in enumerate(self.model.classes_):
                row = f"{true_label:^10} | " + " | ".join([f"{count:^10}" for count in cm[i]])
                logger.info(row)
            
            logger.info("="*50)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logger.info("\nFeature Importance:")
            logger.info("\n" + feature_importance.to_string(index=False))
            
            # Save model and scaler
            joblib.dump(self.model, os.path.join(self.models_dir, 'company_classifier.joblib'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'company_scaler.joblib'))
            
            logger.info("\nModel training completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict_company_type(self, company_data):
        """Predict company type for new data"""
        try:
            if self.model is None or self.scaler is None:
                logger.error("Model not trained. Please train the model first.")
                return None
            
            # Scale the data
            scaled_data = self.scaler.transform(company_data)
            
            # Make prediction
            prediction = self.model.predict(scaled_data)
            probabilities = self.model.predict_proba(scaled_data)
            
            return prediction[0], dict(zip(self.model.classes_, probabilities[0]))
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        classifier = CompanyClassifier()
        data = classifier.load_data()
        if data is not None:
            classifier.train_model(data)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}") 