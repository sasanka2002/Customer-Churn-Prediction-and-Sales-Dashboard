import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import MLModels
from utils.visualizations import Visualizations

# Configure page
st.set_page_config(
    page_title="Customer Churn & Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = Visualizations()

# Main page
st.title("🔮 Customer Churn Prediction & Sales Dashboard")
st.markdown("---")

# Overview metrics
if hasattr(st.session_state.data_processor, 'customers') and st.session_state.data_processor.customers is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(st.session_state.data_processor.customers)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        if hasattr(st.session_state.data_processor, 'transactions') and st.session_state.data_processor.transactions is not None:
            total_revenue = st.session_state.data_processor.transactions['amount'].sum() if 'amount' in st.session_state.data_processor.transactions.columns else 0
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        else:
            st.metric("Total Revenue", "$0.00")
    
    with col3:
        if 'churn' in st.session_state.data_processor.customers.columns:
            churn_rate = st.session_state.data_processor.customers['churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")
    
    with col4:
        if hasattr(st.session_state.data_processor, 'transactions') and st.session_state.data_processor.transactions is not None:
            avg_transaction = st.session_state.data_processor.transactions['amount'].mean() if 'amount' in st.session_state.data_processor.transactions.columns else 0
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
        else:
            st.metric("Avg Transaction", "$0.00")

    st.markdown("---")
    
    # Quick insights
    st.subheader("📈 Quick Insights")
    
    if hasattr(st.session_state.data_processor, 'processed_features') and st.session_state.data_processor.processed_features is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Status:**")
            st.success("✅ Customer data loaded")
            if hasattr(st.session_state.data_processor, 'transactions') and st.session_state.data_processor.transactions is not None:
                st.success("✅ Transaction data loaded")
            if hasattr(st.session_state.data_processor, 'products') and st.session_state.data_processor.products is not None:
                st.success("✅ Product data loaded")
        
        with col2:
            st.write("**Available Features:**")
            features = st.session_state.data_processor.processed_features.columns.tolist()
            st.write(f"• {len(features)} features engineered")
            if hasattr(st.session_state.ml_models, 'lr_model') and st.session_state.ml_models.lr_model is not None:
                st.success("✅ Models trained")
else:
    st.info("👈 Please upload your data using the Data Upload page to get started.")
    
    st.subheader("🚀 Getting Started")
    st.write("""
    1. **Upload Data**: Go to the Data Upload page and upload your customer and transaction data
    2. **Churn Prediction**: View predictions and model performance metrics
    3. **Customer Segmentation**: Explore customer clusters and segmentation insights
    4. **Sales Analysis**: Analyze sales trends and performance metrics
    """)

st.markdown("---")
st.subheader("📋 Navigation")
st.write("""
Use the sidebar to navigate between different sections:
- **Data Upload**: Upload and preprocess your datasets
- **Churn Prediction**: Train ML models including XGBoost, LightGBM, and Deep Learning
- **Customer Segmentation**: K-Means clustering and DBSCAN anomaly detection
- **Sales Analysis**: Explore sales trends and insights
- **Sales Forecasting**: Prophet-based time series forecasting
""")
