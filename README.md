# Telecom Customer Retention Analysis

A machine learning project to predict customer churn and identify key factors driving customer retention in the telecommunications industry.

## Project Structure

- `create.py` - Generates synthetic telecom customer dataset
- `retention.py` - Performs churn prediction analysis using Random Forest
- `telecom_customer_data.csv` - Sample dataset with 1,000 customer records

## Dataset Features

**Customer Demographics:**
- Age, Gender, Region (Lagos, Abuja, Kano, Port Harcourt)

**Service Details:**
- Plan (Basic, Standard, Premium)
- Contract Length (12, 24, 36 months)
- Monthly Spend, Data Usage

**Behavioral Metrics:**
- Late Payments, Complaints, Satisfaction Score
- Churn (Target variable: 0 = Retained, 1 = Churned)

## Quick Start

1. **Generate Dataset:**
   ```bash
   python create.py
   ```

2. **Run Analysis:**
   ```bash
   python retention.py
   ```

## Output

- Classification report with precision, recall, and F1-scores
- Feature importance ranking showing top churn predictors
- Visualization of the most influential factors

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Key Insights

The model identifies which customer attributes most strongly predict churn, enabling targeted retention strategies and proactive customer management.