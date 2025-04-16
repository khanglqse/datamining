import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import logging
import sys
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('financial_clustering.log')
    ]
)
logger = logging.getLogger(__name__)

class FinancialClusterer:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.results_dir = os.path.join(self.data_dir, 'results')
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.kmeans_model = None
        self.company_clusters = {}
        
    def load_processed_data(self):
        """Load all processed financial data files"""
        all_data = {}
        
        # Get all processed data files
        processed_files = glob.glob(os.path.join(self.processed_dir, 'processed_*.csv'))
        
        if not processed_files:
            logger.error("No processed data files found")
            return None
            
        for file_path in processed_files:
            try:
                # Extract company symbol from filename
                symbol = os.path.basename(file_path).split('_')[1].split('.')[0]
                
                # Load data
                df = pd.read_csv(file_path, index_col=0)
                
                # Fix the column headers issue (if needed)
                if len(df.columns) >= 6 and list(df.columns)[0:3] == list(df.columns)[3:6]:
                    # This handles the duplicate columns issue
                    # Assuming first 3 are one set and second 3 are another
                    df1 = df.iloc[:, 0:3]
                    df2 = df.iloc[:, 3:6]
                    
                    # Rename columns to avoid confusion
                    df1.columns = [f"orig_{year}" for year in df1.columns]
                    df2.columns = [f"ratio_{year}" for year in df2.columns]
                    
                    # Combine the data
                    df = pd.concat([df1, df2], axis=1)
                
                all_data[symbol] = df
                logger.info(f"Loaded data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {str(e)}")
                
        return all_data
    
    def prepare_clustering_data(self, all_data):
        """Prepare data for clustering by averaging across years"""
        clustering_data = {}
        
        for symbol, df in all_data.items():
            # Average the values across years to get a single profile per company
            avg_values = df.mean(axis=1)
            clustering_data[symbol] = avg_values
            
        # Convert to DataFrame
        clustering_df = pd.DataFrame(clustering_data)
        
        # Transpose to have companies as rows and metrics as columns
        clustering_df = clustering_df.T
        
        # Check for NaN values
        nan_count = clustering_df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in clustering data")
            
            # Strategy 1: Fill NaN values with column means
            clustering_df = clustering_df.fillna(clustering_df.mean())
            
            # Check if there are still NaN values (e.g., columns with all NaN)
            remaining_nan = clustering_df.isna().sum().sum()
            if remaining_nan > 0:
                logger.warning(f"Still have {remaining_nan} NaN values after filling with means")
                # Strategy 2: Fill remaining NaN values with 0
                clustering_df = clustering_df.fillna(0)
        
        logger.info(f"Prepared clustering data with shape: {clustering_df.shape}")
        return clustering_df
    
    def determine_optimal_clusters(self, data, max_clusters=10):
        """Determine the optimal number of clusters using the elbow method and silhouette score"""
        inertia_values = []
        silhouette_values = []
        # Plot elbow method
        plt.figure(figsize=(12, 5))
        k_range = range(2, min(max_clusters + 1, len(data)))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertia_values.append(kmeans.inertia_)
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_values.append(silhouette_score(data, kmeans.labels_))
            else:
                silhouette_values.append(0)
        
        plt.subplot(1, 2, 1)
        plt.plot(list(k_range), inertia_values, 'bo-')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(list(k_range), silhouette_values, 'ro-')
        plt.title('Silhouette Score Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'optimal_clusters.png'))
        plt.close()
        
        # Find the optimal number of clusters - simplistic approach
        # You might want to manually inspect the plots for better decision
        optimal_k = k_range[silhouette_values.index(max(silhouette_values))]
        logger.info(f"Optimal number of clusters determined: {optimal_k}")
        return optimal_k
    
    def cluster_companies(self, data, n_clusters=None):
        """Cluster companies using KMeans"""
        try:
            # Final check for NaN values before clustering
            if data.isna().any().any():
                logger.warning("Still found NaN values before clustering. Applying final fixes.")
                
                # First try to fill NaNs with column medians (more robust than mean for outliers)
                data = data.fillna(data.median())
                
                # If there are still NaNs (columns with all NaNs), fill with zeros
                if data.isna().any().any():
                    logger.warning("Filling remaining NaNs with zeros")
                    data = data.fillna(0)
                    
                # Log columns that had NaN values
                nan_cols = data.columns[data.isna().any()].tolist()
                if nan_cols:
                    logger.warning(f"Columns with NaN values: {nan_cols}")
            
            # Determine optimal number of clusters if not provided
            if n_clusters is None:
                n_clusters = self.determine_optimal_clusters(data)
            
            # Apply KMeans clustering
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans_model.fit_predict(data)
            
            # Store cluster assignments
            self.company_clusters = {
                company: cluster for company, cluster in zip(data.index, clusters)
            }
            
            # Count companies per cluster
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            logger.info(f"Companies per cluster: {cluster_counts.to_dict()}")
            return clusters
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def visualize_clusters(self, data, clusters):
        """Visualize clusters using PCA for dimensionality reduction"""
        try:
            # Apply PCA to reduce dimensionality to 2 components
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            logger.info(f"PCA explained variance: {explained_variance}")
            
            # Plot clusters
            plt.figure(figsize=(10, 8))
            unique_clusters = set(clusters)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_id, color in zip(sorted(unique_clusters), colors):
                cluster_points = reduced_data[clusters == cluster_id]
                plt.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1], 
                    s=50, 
                    c=[color], 
                    label=f'Cluster {cluster_id}'
                )
            
            # Add company labels
            for i, company in enumerate(data.index):
                plt.annotate(
                    company, 
                    (reduced_data[i, 0], reduced_data[i, 1]),
                    fontsize=8
                )
            
            # Add cluster centers (projected into PCA space)
            centers = pca.transform(self.kmeans_model.cluster_centers_)
            plt.scatter(
                centers[:, 0], 
                centers[:, 1], 
                s=200, 
                c='black', 
                marker='X', 
                label='Cluster Centers'
            )
            
            plt.title('Company Clusters Visualization using PCA')
            plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
            plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, 'company_clusters.png'))
            plt.close()
            
            logger.info("Cluster visualization complete")
        except Exception as e:
            logger.error(f"Error during cluster visualization: {str(e)}")
    
    def analyze_clusters(self, data, clusters):
        """Analyze the characteristics of each cluster"""
        try:
            # Create a DataFrame with cluster assignments
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = data_with_clusters.groupby('cluster').mean()
            
            # Save cluster statistics
            cluster_stats.to_csv(os.path.join(self.results_dir, 'cluster_statistics.csv'))
            
            # Calculate the most distinguishing features for each cluster
            all_clusters_stats = {}
            
            for cluster_id in set(clusters):
                # Get data for this cluster
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id].drop('cluster', axis=1)
                
                # Get data for other clusters
                other_clusters_data = data_with_clusters[data_with_clusters['cluster'] != cluster_id].drop('cluster', axis=1)
                
                # Calculate mean values
                cluster_mean = cluster_data.mean()
                other_mean = other_clusters_data.mean()
                
                # Calculate differences
                diff = cluster_mean - other_mean
                
                # Sort by absolute difference to find most distinguishing features
                distinguishing_features = diff.abs().sort_values(ascending=False)
                all_clusters_stats[cluster_id] = {
                    'top_features': distinguishing_features.index[:5].tolist(),
                    'feature_values': {feature: cluster_mean[feature] for feature in distinguishing_features.index[:5]}
                }
            
            # Print and save cluster analysis
            with open(os.path.join(self.results_dir, 'cluster_analysis.txt'), 'w') as f:
                for cluster_id, stats in all_clusters_stats.items():
                    cluster_info = f"\nCluster {cluster_id} ({sum(clusters == cluster_id)} companies)\n"
                    cluster_info += "Most distinguishing features:\n"
                    for feature in stats['top_features']:
                        value = stats['feature_values'][feature]
                        cluster_info += f"  - {feature}: {value:.4f}\n"
                    logger.info(cluster_info)
                    f.write(cluster_info)
            
            logger.info("Cluster analysis complete")
        except Exception as e:
            logger.error(f"Error during cluster analysis: {str(e)}")
    
    def run_clustering(self, custom_n_clusters=None):
        """Run the complete clustering process"""
        try:
            # Load data
            all_data = self.load_processed_data()
            if not all_data:
                logger.error("No data to cluster")
                return False
            
            # Prepare data for clustering
            clustering_data = self.prepare_clustering_data(all_data)
            logger.info(f"Prepared clustering data with shape: {clustering_data.shape}")
            
            # Run clustering
            clusters = self.cluster_companies(clustering_data, n_clusters=custom_n_clusters)
            if clusters is None:
                return False
            
            # Visualize clusters
            self.visualize_clusters(clustering_data, clusters)
            
            # Analyze clusters
            self.analyze_clusters(clustering_data, clusters)
            
            # Save cluster assignments
            pd.Series(self.company_clusters).to_csv(os.path.join(self.results_dir, 'company_clusters.csv'))
            
            logger.info("Clustering process completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error in clustering process: {str(e)}")
            return False
                
if __name__ == "__main__":
    try:
        clusterer = FinancialClusterer()
        clusterer.run_clustering()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")