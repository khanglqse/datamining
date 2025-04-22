import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def load_company_names():
    """Load company names from all_companies.csv"""
    try:
        company_names_df = pd.read_csv('data/raw/all_companies.csv')
        return dict(zip(company_names_df['symbol'], company_names_df['name']))
    except Exception as e:
        print(f"Error loading company names: {str(e)}")
        return {}

def process_company_data(file_path):
    """Process a single company's data file"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Transpose the data to have years as columns
        df = df.set_index('Unnamed: 0').T
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index, format='%Y')
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Initialize default values for missing columns
        default_values = {
            'revenue': 0.0,
            'gross_profit': 0.0,
            'net_income': 0.0,
            'total_assets': 0.0,
            'total_debt': 0.0,
            'current_assets': 0.0,
            'current_liabilities': 0.0,
            'equity': 0.0
        }
        
        # Add missing columns with default values
        for col, default in default_values.items():
            if col not in df.columns:
                df[col] = default
                print(f"Warning: Column '{col}' not found in {file_path}, using default value {default}")
        
        # Calculate additional metrics with error handling
        try:
            df['debt_ratio'] = df['total_debt'] / df['total_assets'].replace(0, 1)
        except:
            df['debt_ratio'] = 0.0
            
        try:
            df['current_ratio'] = df['current_assets'] / df['current_liabilities'].replace(0, 1)
        except:
            df['current_ratio'] = 0.0
            
        try:
            df['gross_margin'] = df['gross_profit'] / df['revenue'].replace(0, 1)
        except:
            df['gross_margin'] = 0.0
            
        try:
            df['net_profit_margin'] = df['net_income'] / df['revenue'].replace(0, 1)
        except:
            df['net_profit_margin'] = 0.0
            
        try:
            df['revenue_growth'] = df['revenue'].pct_change().fillna(0)
        except:
            df['revenue_growth'] = 0.0
            
        try:
            df['net_income_growth'] = df['net_income'].pct_change().fillna(0)
        except:
            df['net_income_growth'] = 0.0
            
        try:
            df['roa'] = df['net_income'] / df['total_assets'].replace(0, 1)
        except:
            df['roa'] = 0.0
            
        try:
            # Calculate total equity if not present
            if 'total_equity' not in df.columns:
                df['total_equity'] = df['total_assets'] - df['total_debt']
            df['roe'] = df['net_income'] / df['total_equity'].replace(0, 1)
        except:
            df['roe'] = 0.0
        
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def calculate_health_score(metrics):
    """Calculate health score based on Altman Z-Score"""
    try:
        # Extract required metrics
        total_assets = float(metrics.get('total_assets', 0))
        current_assets = float(metrics.get('current_assets', 0))
        current_liabilities = float(metrics.get('current_liabilities', 0))
        total_debt = float(metrics.get('total_debt', 0))
        net_income = float(metrics.get('net_income', 0))
        revenue = float(metrics.get('revenue', 0))
        ebit = float(metrics.get('profit_before_tax', 0))  # Using profit before tax as EBIT proxy
        
        # Calculate components
        # X1 = Working Capital / Total Assets
        working_capital = current_assets - current_liabilities
        x1 = working_capital / total_assets if total_assets != 0 else 0
        
        # X2 = Retained Earnings / Total Assets
        # Using net income as retained earnings proxy
        x2 = net_income / total_assets if total_assets != 0 else 0
        
        # X3 = EBIT / Total Assets
        x3 = ebit / total_assets if total_assets != 0 else 0
        
        # X4 = Market Value of Equity / Total Liabilities
        # Using book value of equity as market value proxy
        equity = total_assets - total_debt
        x4 = equity / total_debt if total_debt != 0 else 0
        
        # X5 = Sales / Total Assets
        x5 = revenue / total_assets if total_assets != 0 else 0
        
        # Calculate Z-Score
        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
        
        # Convert Z-Score to health score (0-100)
        # Z-Score interpretation:
        # Z > 2.99: Safe Zone
        # 1.81 < Z < 2.99: Grey Zone
        # Z < 1.81: Distress Zone
        if z_score > 2.99:
            # Map to 70-100 range
            health_score = 70 + ((z_score - 2.99) / (5 - 2.99)) * 30
        elif z_score > 1.81:
            # Map to 40-70 range
            health_score = 40 + ((z_score - 1.81) / (2.99 - 1.81)) * 30
        else:
            # Map to 0-40 range
            health_score = (z_score / 1.81) * 40
        
        # Ensure score is within 0-100 range
        health_score = max(0, min(100, health_score))
        
        return round(health_score, 2)
        
    except Exception as e:
        print(f"Error calculating health score: {str(e)}")
        return 0.0

def calculate_company_scores(df):
    """Calculate health scores for a company"""
    if df is None or df.empty:
        return None
    
    try:
        # Get the latest year's data
        latest_data = df.iloc[-1]
        
        # Handle NaN values and prevent division by zero
        metrics = {
            'roa': latest_data.get('roa'),
            'roe': latest_data.get('roe'),
            'current_ratio': latest_data.get('current_ratio'),
            'debt_ratio': latest_data.get('debt_ratio'),
            'revenue_growth': latest_data.get('revenue_growth'),
            'gross_margin': latest_data.get('gross_margin'),
            'net_profit_margin': latest_data.get('net_profit_margin')
        }
        
        # Replace NaN and Infinity values with 0
        metrics = {k: 0 if pd.isna(v) or np.isinf(v) else float(v) for k, v in metrics.items()}
        
        # Calculate health score
        health_score = calculate_health_score(metrics)
        
        # Determine risk level
        if health_score >= 70:
            risk_level = "Low"
        elif health_score >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            'health_score': health_score,
            'risk_level': risk_level,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error calculating scores: {str(e)}")
        return None

def main():
    # Create output directories if they don't exist
    os.makedirs('data/display/processed', exist_ok=True)
    os.makedirs('data/display/scores', exist_ok=True)
    os.makedirs('data/display/metrics', exist_ok=True)
    os.makedirs('data/display/trends', exist_ok=True)
    
    # Load company names
    company_names = load_company_names()
    
    # Process each company's data
    all_scores = []
    unscaled_dir = 'data/unscaled'
    
    for filename in os.listdir(unscaled_dir):
        if filename.endswith('.csv'):
            company_symbol = filename.replace('.csv', '')
            file_path = os.path.join(unscaled_dir, filename)
            
            print(f"Processing {company_symbol}...")
            
            # Process company data
            df = process_company_data(file_path)
            if df is not None:
                try:
                    # Save processed data
                    processed_path = os.path.join('data/display/processed', f"{company_symbol}_processed.csv")
                    df.to_csv(processed_path)
                    
                    # Save metrics data
                    metrics = df.iloc[-1].to_dict()  # Get latest year's metrics
                    metrics_path = os.path.join('data/display/metrics', f"{company_symbol}_metrics.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    
                    # Save trend data
                    trend_data = {
                        'years': df.index.strftime('%Y').tolist(),
                        'revenue': df['revenue'].tolist(),
                        'net_income': df['net_income'].tolist(),
                        'roa': df['roa'].tolist(),
                        'roe': df['roe'].tolist(),
                        'debt_ratio': df['debt_ratio'].tolist(),
                        'current_ratio': df['current_ratio'].tolist()
                    }
                    trend_path = os.path.join('data/display/trends', f"{company_symbol}_trends.json")
                    with open(trend_path, 'w') as f:
                        json.dump(trend_data, f, indent=4)
                    
                    # Calculate scores
                    scores = calculate_company_scores(df)
                    if scores:
                        scores['symbol'] = company_symbol
                        scores['name'] = company_names.get(company_symbol, company_symbol)
                        all_scores.append(scores)
                except Exception as e:
                    print(f"Error saving data for {company_symbol}: {str(e)}")
                    continue
    
    # Save all scores
    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        scores_path = os.path.join('data/display/scores', 'company_scores.csv')
        scores_df.to_csv(scores_path, index=False)
        
        # Also save as JSON for easier frontend consumption
        scores_json = scores_df.to_dict('records')
        scores_json_path = os.path.join('data/display/scores', 'company_scores.json')
        with open(scores_json_path, 'w') as f:
            json.dump(scores_json, f, indent=4)
            
        print(f"Saved scores for {len(all_scores)} companies to {scores_path}")
        
        # Create leaderboard data
        leaderboard = {
            'Low Risk': scores_df[scores_df['risk_level'] == 'Low'].to_dict('records'),
            'Moderate Risk': scores_df[scores_df['risk_level'] == 'Moderate'].to_dict('records'),
            'High Risk': scores_df[scores_df['risk_level'] == 'High'].to_dict('records')
        }
        leaderboard_path = os.path.join('data/display/scores', 'leaderboard.json')
        with open(leaderboard_path, 'w') as f:
            json.dump(leaderboard, f, indent=4)
            
        print(f"Saved leaderboard data to {leaderboard_path}")

if __name__ == "__main__":
    main() 