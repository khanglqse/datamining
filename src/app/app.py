from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from numpy import inf
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load unscaled data
def load_unscaled_data():
    unscaled_dir = Path('data/unscaled')
    all_data = []
    
    for file in unscaled_dir.glob('*.csv'):
        symbol = file.stem
        df = pd.read_csv(file, index_col=0)
        df = df.transpose()  # Transpose to get years as index
        df['company'] = symbol
        all_data.append(df)
    
    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()

# Load company names
def load_company_names():
    try:
        companies_df = pd.read_csv('data/raw/all_companies.csv', encoding='utf-8')
        # Clean up the data - remove any rows with NaN values in name column
        companies_df = companies_df.dropna(subset=['name'])
        # Create dictionary with symbol as key and name as value
        return dict(zip(companies_df['symbol'], companies_df['name']))
    except FileNotFoundError:
        print("Warning: all_companies.csv not found")
        return {}
    except Exception as e:
        print(f"Error loading company names: {str(e)}")
        return {}

# Load preprocessed data
def load_company_scores():
    try:
        with open('data/display/scores/company_scores.json', 'r') as f:
            return pd.DataFrame(json.load(f))
    except Exception as e:
        print(f"Error loading company scores: {str(e)}")
        return pd.DataFrame()

# Load processed data
def load_processed_data():
    try:
        all_data = []
        processed_dir = 'data/display/processed'
        for filename in os.listdir(processed_dir):
            if filename.endswith('_processed.csv'):
                company_symbol = filename.replace('_processed.csv', '')
                file_path = os.path.join(processed_dir, filename)
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df['company'] = company_symbol
                all_data.append(df)
        return pd.concat(all_data) if all_data else pd.DataFrame()
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        return pd.DataFrame()

# Load metrics data
def load_company_metrics(company_symbol):
    try:
        metrics_path = os.path.join('data/display/metrics', f"{company_symbol}_metrics.json")
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics for {company_symbol}: {str(e)}")
        return {}

# Load trend data
def load_company_trends(company_symbol):
    try:
        trend_path = os.path.join('data/display/trends', f"{company_symbol}_trends.json")
        with open(trend_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading trends for {company_symbol}: {str(e)}")
        return {}

# Load data on startup
unscaled_data = load_unscaled_data()
company_names = load_company_names()
company_scores = load_company_scores()
processed_data = load_processed_data()

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'company_classifier.joblib')
scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'scaler.joblib')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    model = None
    scaler = None
    print("Warning: Model or scaler not found")

def get_risk_assessment(debt_ratio, current_ratio, roa, roe):
    """Calculate detailed risk assessment based on financial metrics"""
    # Debt risk assessment
    if debt_ratio > 0.8:
        debt_risk = "High"
        debt_explanation = "Company has very high debt levels, indicating significant financial risk"
    elif debt_ratio > 0.6:
        debt_risk = "Moderate"
        debt_explanation = "Company has elevated debt levels that should be monitored"
    else:
        debt_risk = "Low"
        debt_explanation = "Company maintains reasonable debt levels"
    
    # Liquidity risk assessment
    if current_ratio < 1.0:
        liquidity_risk = "High"
        liquidity_explanation = "Company may face short-term liquidity challenges"
    elif current_ratio < 1.5:
        liquidity_risk = "Moderate"
        liquidity_explanation = "Company has adequate but not strong liquidity"
    else:
        liquidity_risk = "Low"
        liquidity_explanation = "Company maintains strong liquidity position"
    
    # Profitability risk assessment
    if roa < 0:
        profitability_risk = "High"
        profitability_explanation = "Company is not generating positive returns on assets"
    elif roa < 0.05:
        profitability_risk = "Moderate"
        profitability_explanation = "Company's return on assets is below industry average"
    else:
        profitability_risk = "Low"
        profitability_explanation = "Company maintains strong profitability"
    
    # Overall risk assessment
    risk_count = sum(1 for risk in [debt_risk, liquidity_risk, profitability_risk] if risk == "High")
    if risk_count >= 2:
        overall_risk = "High"
    elif risk_count == 1:
        overall_risk = "Moderate"
    else:
        overall_risk = "Low"
    
    return {
        "overall_risk": overall_risk,
        "components": {
            "debt_risk": {
                "level": debt_risk,
                "explanation": debt_explanation,
                "value": debt_ratio
            },
            "liquidity_risk": {
                "level": liquidity_risk,
                "explanation": liquidity_explanation,
                "value": current_ratio
            },
            "profitability_risk": {
                "level": profitability_risk,
                "explanation": profitability_explanation,
                "value": roa
            }
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/companies')
def get_companies():
    companies = unscaled_data['company'].unique().tolist()
    return jsonify(companies)

@app.route('/api/company/<company_name>')
def get_company_data(company_name):
    try:
        metrics = load_company_metrics(company_name)
        trends = load_company_trends(company_name)
        
        if not metrics and not trends:
            return jsonify({"error": "Company not found"}), 404
            
        return jsonify({
            "metrics": metrics,
            "trends": trends
        })
    except Exception as e:
        print(f"Error in company data endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/metrics')
def get_metrics():
    metrics = {
        'financial': ['revenue', 'gross_profit', 'net_income', 'profit_before_tax'],
        'ratios': ['roa', 'roe', 'debt_ratio', 'current_ratio', 'gross_margin', 'net_profit_margin'],
        'growth': ['revenue_growth']
    }
    return jsonify(metrics)

@app.route('/api/trend/<company_name>')
def get_company_trend(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Create trend chart
    fig = go.Figure()
    
    # Add financial metrics
    financial_metrics = ['revenue', 'gross_profit', 'net_income']
    for metric in financial_metrics:
        if metric in company_data.columns:
            fig.add_trace(go.Scatter(
                x=company_data.index,
                y=company_data[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title=f"Financial Trends for {company_name}",
        xaxis_title="Year",
        yaxis_title="Value (VND)",
        showlegend=True
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/ratios/<company_name>')
def get_company_ratios(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Create ratios chart
    fig = go.Figure()
    
    # Add ratio metrics
    ratio_metrics = ['roa', 'roe', 'debt_ratio', 'current_ratio', 'gross_margin', 'net_profit_margin']
    for metric in ratio_metrics:
        if metric in company_data.columns:
            fig.add_trace(go.Scatter(
                x=company_data.index,
                y=company_data[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title=f"Financial Ratios for {company_name}",
        xaxis_title="Year",
        yaxis_title="Ratio",
        showlegend=True
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/risk-assessment/<company_name>')
def get_risk_assessment_endpoint(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Get latest year's data
    latest_data = company_data.iloc[-1]
    
    # Get risk assessment
    assessment = get_risk_assessment(
        float(latest_data['debt_ratio']),
        float(latest_data['current_ratio']),
        float(latest_data['roa']),
        float(latest_data['roe'])
    )
    
    # Add trend analysis
    trend_analysis = {
        'debt_ratio_trend': 'increasing' if company_data['debt_ratio'].iloc[-1] > company_data['debt_ratio'].iloc[0] else 'decreasing',
        'current_ratio_trend': 'increasing' if company_data['current_ratio'].iloc[-1] > company_data['current_ratio'].iloc[0] else 'decreasing',
        'roa_trend': 'improving' if company_data['roa'].iloc[-1] > company_data['roa'].iloc[0] else 'declining'
    }
    
    response = {
        "company": company_name,
        "assessment": assessment,
        "trend_analysis": trend_analysis,
        "latest_values": {
            "debt_ratio": float(latest_data['debt_ratio']),
            "current_ratio": float(latest_data['current_ratio']),
            "roa": float(latest_data['roa']),
            "roe": float(latest_data['roe'])
        }
    }
    
    return jsonify(response)

@app.route('/api/company-info/<company_name>')
def get_company_info(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Get latest year's data
    latest_data = company_data.iloc[-1]
    
    # Get full company name from the dictionary
    full_name = company_names.get(company_name, company_name)
    
    # Calculate key financial indicators
    financial_indicators = {
        'profitability': {
            'gross_margin': float(latest_data['gross_margin']),
            'net_profit_margin': float(latest_data['net_profit_margin']),
            'roa': float(latest_data['roa']),
            'roe': float(latest_data['roe'])
        },
        'liquidity': {
            'current_ratio': float(latest_data['current_ratio']),
            'working_capital': float(latest_data['current_assets']) - float(latest_data['current_liabilities'])
        },
        'leverage': {
            'debt_ratio': float(latest_data['debt_ratio']),
            'debt_to_equity': float(latest_data['total_debt']) / float(latest_data['equity'])
        },
        'growth': {
            'revenue_growth': float(latest_data['revenue_growth']),
            'profit_growth': (float(latest_data['net_income']) - float(company_data['net_income'].iloc[0])) / abs(float(company_data['net_income'].iloc[0]))
        }
    }
    
    response = {
        "company": company_name,
        "full_name": full_name,
        "latest_year": company_data.index[-1],
        "financial_indicators": financial_indicators,
        "key_metrics": {
            "revenue": float(latest_data['revenue']),
            "gross_profit": float(latest_data['gross_profit']),
            "net_income": float(latest_data['net_income']),
            "total_assets": float(latest_data['total_assets']),
            "total_debt": float(latest_data['total_debt']),
            "equity": float(latest_data['equity'])
        }
    }
    
    return jsonify(response)

@app.route('/api/radar/<company_name>')
def get_radar_chart(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Get latest year's data
    latest_data = company_data.iloc[-1]
    
    # Define metrics for radar chart
    metrics = {
        'Profitability': ['roa', 'roe', 'gross_margin', 'net_profit_margin'],
        'Liquidity': ['current_ratio', 'quick_ratio'],
        'Leverage': ['debt_ratio', 'debt_to_equity'],
        'Growth': ['revenue_growth', 'profit_growth']
    }
    
    # Create radar chart
    fig = go.Figure()
    
    # Add traces for each category
    for category, metric_list in metrics.items():
        values = []
        for metric in metric_list:
            if metric in latest_data:
                values.append(float(latest_data[metric]))
            else:
                values.append(0)  # Default to 0 if metric not found
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_list,
            fill='toself',
            name=category
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Adjust range based on your metrics
            )
        ),
        showlegend=True,
        title=f"Financial Metrics Radar Chart - {company_name}"
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/health-score/<company_name>')
def get_health_score(company_name):
    try:
        # Load company data
        company_data = unscaled_data[unscaled_data['company'] == company_name]
        if company_data.empty:
            return jsonify({"error": "Company not found"}), 404
        
        # Get latest year's data
        latest_data = company_data.iloc[-1]
        
        # Calculate Altman Z-Score components
        try:
            # X1 = Working Capital / Total Assets
            working_capital = float(latest_data['current_assets']) - float(latest_data['current_liabilities'])
            x1 = working_capital / float(latest_data['total_assets'])
            
            # X2 = Retained Earnings / Total Assets (using equity as proxy)
            x2 = float(latest_data['equity']) / float(latest_data['total_assets'])
            
            # X3 = EBIT / Total Assets (using profit_before_tax as proxy)
            x3 = float(latest_data['profit_before_tax']) / float(latest_data['total_assets'])
            
            # X4 = Market Value of Equity / Total Liabilities (using equity as proxy)
            x4 = float(latest_data['equity']) / float(latest_data['total_debt'])
            
            # X5 = Sales / Total Assets
            x5 = float(latest_data['revenue']) / float(latest_data['total_assets'])
            
            # Calculate Z-Score
            z_score = (
                1.2 * x1 +
                1.4 * x2 +
                3.3 * x3 +
                0.6 * x4 +
                1.0 * x5
            )
            
            # Map Z-Score to health score (1-100)
            if z_score > 2.99:
                health_score = min(100, 70 + (z_score - 2.99) * 10)  # Safe zone: 70-100
            elif z_score > 1.81:
                health_score = 40 + (z_score - 1.81) * 25  # Grey zone: 40-70
            else:
                health_score = max(1, z_score * 22)  # Distress zone: 1-40
            
            # Determine risk level
            if z_score > 2.99:
                risk_level = "Low"
            elif z_score > 1.81:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': f"Financial Health Score - {company_name}",
                    'font': {'size': 20}
                },
                gauge={
                    'axis': {'range': [1, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [1, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': health_score
                    }
                }
            ))
            
            # Add annotations for risk levels
            fig.add_annotation(
                x=0.5,
                y=0.3,
                text=f"Risk Level: {risk_level}",
                showarrow=False,
                font={'size': 16}
            )
            
            # Create response
            response = {
                "company": company_name,
                "health_score": round(health_score, 2),
                "z_score": round(z_score, 2),
                "risk_level": risk_level,
                "components": {
                    "working_capital_ratio": round(x1, 4),
                    "retained_earnings_ratio": round(x2, 4),
                    "ebit_ratio": round(x3, 4),
                    "equity_to_debt_ratio": round(x4, 4),
                    "sales_ratio": round(x5, 4)
                },
                "interpretation": {
                    "z_score": {
                        "> 2.99": "Safe Zone - Low risk of bankruptcy",
                        "1.81 - 2.99": "Grey Zone - Moderate risk",
                        "< 1.81": "Distress Zone - High risk of bankruptcy"
                    }
                },
                "chart": json.loads(fig.to_json())
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error calculating Z-Score for {company_name}: {str(e)}")
            return jsonify({"error": "Error calculating financial health score"}), 500
            
    except Exception as e:
        print(f"Error in health score endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ratios-comparison/<company_name>')
def get_ratios_comparison(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    latest_data = company_data.iloc[-1]
    
    # Calculate industry averages (this is a placeholder - you should calculate real averages)
    industry_avg = {
        'roa': 0.05,
        'roe': 0.15,
        'debt_ratio': 0.6,
        'current_ratio': 1.5
    }
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add company bars
    fig.add_trace(go.Bar(
        name='Company',
        x=['ROA', 'ROE', 'Debt Ratio', 'Current Ratio'],
        y=[
            float(latest_data['roa']),
            float(latest_data['roe']),
            float(latest_data['debt_ratio']),
            float(latest_data['current_ratio'])
        ]
    ))
    
    # Add industry average bars
    fig.add_trace(go.Bar(
        name='Industry Average',
        x=['ROA', 'ROE', 'Debt Ratio', 'Current Ratio'],
        y=[industry_avg['roa'], industry_avg['roe'], 
           industry_avg['debt_ratio'], industry_avg['current_ratio']]
    ))
    
    fig.update_layout(
        title='Financial Ratios Comparison',
        barmode='group',
        yaxis_title='Value'
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/growth-trend/<company_name>')
def get_growth_trend(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    # Create growth trend chart
    fig = go.Figure()
    
    # Add revenue growth line
    fig.add_trace(go.Scatter(
        x=company_data.index,
        y=company_data['revenue_growth'],
        name='Revenue Growth',
        line=dict(color='blue')
    ))
    
    # Add profit growth line
    fig.add_trace(go.Scatter(
        x=company_data.index,
        y=company_data['profit_growth'],
        name='Profit Growth',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Growth Metrics Trend',
        xaxis_title='Year',
        yaxis_title='Growth Rate',
        hovermode='x unified'
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/financial-structure/<company_name>')
def get_financial_structure(company_name):
    company_data = unscaled_data[unscaled_data['company'] == company_name]
    if company_data.empty:
        return jsonify({"error": "Company not found"}), 404
    
    latest_data = company_data.iloc[-1]
    
    # Create financial structure pie charts
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    
    # Assets composition
    fig.add_trace(go.Pie(
        labels=['Current Assets', 'Fixed Assets', 'Other Assets'],
        values=[
            float(latest_data['current_assets']),
            float(latest_data['fixed_assets']),
            float(latest_data['total_assets']) - float(latest_data['current_assets']) - float(latest_data['fixed_assets'])
        ],
        name='Assets'
    ), 1, 1)
    
    # Liabilities composition
    fig.add_trace(go.Pie(
        labels=['Current Liabilities', 'Long-term Debt', 'Equity'],
        values=[
            float(latest_data['current_liabilities']),
            float(latest_data['total_debt']) - float(latest_data['current_liabilities']),
            float(latest_data['equity'])
        ],
        name='Liabilities & Equity'
    ), 1, 2)
    
    fig.update_layout(
        title='Financial Structure Analysis',
        annotations=[
            dict(text='Assets', x=0.18, y=0.5, font_size=20, showarrow=False),
            dict(text='Liabilities & Equity', x=0.82, y=0.5, font_size=20, showarrow=False)
        ]
    )
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/leaderboard')
def get_leaderboard():
    try:
        # Get unique companies from unscaled data
        companies = unscaled_data['company'].unique()
        
        # Calculate health scores for all companies
        company_scores = []
        for company in companies:
            try:
                # Get company data
                company_data = unscaled_data[unscaled_data['company'] == company]
                if company_data.empty:
                    continue
                
                # Get latest year's data
                latest_data = company_data.iloc[-1]
                
                # Skip if any required denominator is zero
                if (float(latest_data['total_assets']) == 0 or 
                    float(latest_data['total_debt']) == 0):
                    continue
                
                # Calculate Altman Z-Score components
                try:
                    # X1 = Working Capital / Total Assets
                    working_capital = float(latest_data['current_assets']) - float(latest_data['current_liabilities'])
                    x1 = working_capital / float(latest_data['total_assets'])
                    
                    # X2 = Retained Earnings / Total Assets (using equity as proxy)
                    x2 = float(latest_data['equity']) / float(latest_data['total_assets'])
                    
                    # X3 = EBIT / Total Assets (using profit_before_tax as proxy)
                    x3 = float(latest_data['profit_before_tax']) / float(latest_data['total_assets'])
                    
                    # X4 = Market Value of Equity / Total Liabilities (using equity as proxy)
                    x4 = float(latest_data['equity']) / float(latest_data['total_debt'])
                    
                    # X5 = Sales / Total Assets
                    x5 = float(latest_data['revenue']) / float(latest_data['total_assets'])
                    
                    # Calculate Z-Score
                    z_score = (
                        1.2 * x1 +
                        1.4 * x2 +
                        3.3 * x3 +
                        0.6 * x4 +
                        1.0 * x5
                    )
                    
                    # Skip if Z-Score is invalid
                    if not np.isfinite(z_score):
                        continue
                    
                    # Map Z-Score to health score (1-100)
                    if z_score > 2.99:
                        health_score = min(100, 70 + (z_score - 2.99) * 10)  # Safe zone: 70-100
                    elif z_score > 1.81:
                        health_score = 40 + (z_score - 1.81) * 25  # Grey zone: 40-70
                    else:
                        health_score = max(1, z_score * 22)  # Distress zone: 1-40
                    
                    # Determine risk level
                    if z_score > 2.99:
                        risk_level = "Low"
                    elif z_score > 1.81:
                        risk_level = "Moderate"
                    else:
                        risk_level = "High"
                    
                    # Get company name
                    full_name = company_names.get(company, company)
                    
                    # Add to scores
                    company_scores.append({
                        'symbol': company,
                        'name': full_name,
                        'health_score': round(health_score, 2),
                        'risk_level': risk_level,
                        'metrics': {
                            'roa': float(latest_data['roa']),
                            'roe': float(latest_data['roe']),
                            'debt_ratio': float(latest_data['debt_ratio']),
                            'current_ratio': float(latest_data['current_ratio'])
                        }
                    })
                    
                except (ZeroDivisionError, ValueError) as e:
                    print(f"Skipping {company} due to calculation error: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing company {company}: {str(e)}")
                continue
        
        # Sort by health score in descending order
        company_scores.sort(key=lambda x: x['health_score'], reverse=True)
        
        # Get top 20 companies
        top_companies = company_scores[:20]
        
        return jsonify(top_companies)
    except Exception as e:
        print(f"Error in leaderboard endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 