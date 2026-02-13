import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

st.title("📁 Data Upload & Preprocessing")
st.markdown("Upload your customer, transaction, and product data files for analysis.")

# Initialize data processor if not exists
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

# File upload section
st.header("📤 Upload Data Files")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Data")
    customer_file = st.file_uploader(
        "Upload customer data (CSV)",
        type=['csv'],
        key="customer_upload",
        help="Required: Customer information including demographics, subscription details, etc."
    )
    
    if customer_file is not None:
        if st.session_state.data_processor.load_customer_data(customer_file):
            st.write("**Preview:**")
            st.dataframe(st.session_state.data_processor.customers.head())

with col2:
    st.subheader("Transaction Data")
    transaction_file = st.file_uploader(
        "Upload transaction data (CSV)",
        type=['csv'],
        key="transaction_upload",
        help="Optional but recommended: Transaction history with amounts, dates, and customer IDs"
    )
    
    if transaction_file is not None:
        if st.session_state.data_processor.load_transaction_data(transaction_file):
            st.write("**Preview:**")
            st.dataframe(st.session_state.data_processor.transactions.head())

with col3:
    st.subheader("Product Data")
    product_file = st.file_uploader(
        "Upload product data (CSV)",
        type=['csv'],
        key="product_upload",
        help="Optional: Product information for enhanced analysis"
    )
    
    if product_file is not None:
        if st.session_state.data_processor.load_product_data(product_file):
            st.write("**Preview:**")
            st.dataframe(st.session_state.data_processor.products.head())

# Data preprocessing section
if st.session_state.data_processor.customers is not None:
    st.markdown("---")
    st.header("🔧 Data Preprocessing")
    
    # Show data summary
    with st.expander("📊 Data Summary", expanded=True):
        summary = st.session_state.data_processor.get_data_summary()
        
        for dataset_name, dataset_info in summary.items():
            st.subheader(f"{dataset_name.title()} Dataset")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", dataset_info['shape'][0])
            with col2:
                st.metric("Columns", dataset_info['shape'][1])
            with col3:
                st.metric("Missing Values", dataset_info['missing_values'])
            
            if st.checkbox(f"Show {dataset_name} columns", key=f"show_{dataset_name}_cols"):
                st.write("**Columns:**", ", ".join(dataset_info['columns']))
    
    # Preprocessing controls
    st.subheader("🛠️ Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Automatic preprocessing includes:**")
        st.write("✅ Missing value imputation")
        st.write("✅ Feature engineering")
        st.write("✅ Categorical encoding")
        st.write("✅ Feature scaling")
    
    with col2:
        st.write("**Engineered features:**")
        st.write("• Purchase frequency")
        st.write("• Total amount spent")
        st.write("• Average transaction value")
        st.write("• Days since last purchase")
        st.write("• Engagement score")
        st.write("• Subscription tenure")
    
    # Preprocessing button
    if st.button("🚀 Start Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            if st.session_state.data_processor.preprocess_data():
                st.success("✅ Data preprocessing completed successfully!")
                
                # Show processed data summary
                if st.session_state.data_processor.processed_features is not None:
                    st.subheader("📈 Processed Features Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Features", len(st.session_state.data_processor.processed_features.columns))
                    with col2:
                        st.metric("Samples", len(st.session_state.data_processor.processed_features))
                    with col3:
                        complete_ratio = (1 - st.session_state.data_processor.processed_features.isnull().sum().sum() / 
                                        (len(st.session_state.data_processor.processed_features) * 
                                         len(st.session_state.data_processor.processed_features.columns))) * 100
                        st.metric("Data Completeness", f"{complete_ratio:.1f}%")
                    
                    # Show feature statistics
                    with st.expander("📊 Feature Statistics"):
                        st.dataframe(st.session_state.data_processor.processed_features.describe())

# Data quality checks
if st.session_state.data_processor.processed_features is not None:
    st.markdown("---")
    st.header("✅ Data Quality Checks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        missing_data = st.session_state.data_processor.processed_features.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) == 0:
            st.success("✅ No missing values found!")
        else:
            st.warning(f"⚠️ {len(missing_data)} columns have missing values")
            st.dataframe(missing_data.to_frame('Missing Count'))
    
    with col2:
        st.subheader("Data Types")
        dtype_counts = st.session_state.data_processor.processed_features.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"**{dtype}**: {count} columns")

# Navigation helper
if st.session_state.data_processor.processed_features is not None:
    st.markdown("---")
    st.info("✅ Data is ready for analysis! Navigate to the next sections:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("pages/2_Churn_Prediction.py", label="🔮 Churn Prediction", icon="🔮")
    with col2:
        st.page_link("pages/3_Customer_Segmentation.py", label="👥 Customer Segmentation", icon="👥")
    with col3:
        st.page_link("pages/4_Sales_Analysis.py", label="📈 Sales Analysis", icon="📈")

else:
    st.info("👆 Please upload your customer data and run preprocessing to continue.")
