# Customer Churn Prediction & Sales Dashboard

## Overview

This project is a Streamlit-based web application that helps businesses analyze customer behavior, predict churn probability, and visualize sales trends. The system combines data analytics and machine learning to provide actionable insights for customer retention and sales optimization.

**Core Purpose:** Enable businesses to identify at-risk customers, understand sales patterns, and make data-driven decisions to improve customer retention and revenue growth.

**Key Capabilities:**
- Upload and preprocess customer, transaction, and product data
- Train multiple machine learning models for churn prediction (Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks)
- Perform customer segmentation using K-Means and DBSCAN clustering
- Analyze sales trends and patterns across time periods
- Forecast future sales using Facebook Prophet
- Interactive visualizations and dashboards

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework:** Streamlit multi-page application
- **Main App (app.py):** Entry point displaying overview metrics and summary dashboard
- **Page Structure:** Modular page-based navigation using Streamlit's pages feature
  - Data Upload & Preprocessing
  - Churn Prediction & Model Training
  - Customer Segmentation
  - Sales Analysis
  - Sales Forecasting

**State Management:** Streamlit session state for persisting data processors, ML models, and visualizations across page navigation. This avoids reloading data and retraining models on every interaction.

**Design Choice Rationale:** Streamlit was chosen for rapid development and built-in UI components. The multi-page structure provides clear separation of concerns and intuitive navigation for different analytical workflows.

### Backend Architecture

**Core Components:**

1. **DataProcessor (utils/data_processor.py)**
   - Handles CSV file uploads for customers, transactions, and products
   - Performs ETL operations and data validation
   - Feature engineering (purchase frequency, engagement scores, subscription tenure)
   - Data preprocessing: missing value imputation, normalization, categorical encoding
   - Uses scikit-learn's StandardScaler and LabelEncoder

2. **MLModels (utils/ml_models.py)**
   - Manages multiple machine learning model types:
     - **Supervised Learning:** Logistic Regression, Random Forest, XGBoost, LightGBM
     - **Deep Learning:** Keras-based Artificial Neural Networks (ANNs) and Recurrent Neural Networks (RNNs)
     - **Unsupervised Learning:** K-Means clustering, DBSCAN for anomaly detection
   - Handles train-test splitting and model evaluation
   - Tracks model performance metrics (accuracy, precision, recall, F1-score)
   - Includes fallback for synthetic churn target creation when no churn column exists

3. **Visualizations (utils/visualizations.py)**
   - Creates interactive plots using Plotly and static plots with Matplotlib/Seaborn
   - Generates confusion matrices, feature importance charts, and various business analytics visualizations
   - Maintains consistent color palette across visualizations

**Data Flow:**
1. User uploads CSV files → DataProcessor validates and loads data
2. DataProcessor performs preprocessing → Creates processed_features dataframe
3. MLModels consumes processed features → Trains models and generates predictions
4. Visualizations renders results → Interactive charts displayed in Streamlit UI

**Design Choice Rationale:** 
- Separation into three utility classes provides modularity and reusability
- Session state persistence prevents redundant data loading and model retraining
- Multiple model types allow comparison and selection of best-performing approaches
- Flexible column detection handles varying CSV schemas from different data sources

### Data Storage

**Current Implementation:** In-memory pandas DataFrames stored in Streamlit session state

**Data Structures:**
- Customer data: Demographics, subscription details, engagement metrics
- Transaction data: Purchase history with amounts, dates, customer IDs
- Product data: Optional product catalog information

**Design Choice Rationale:** 
- In-memory storage is suitable for prototype/demo purposes and moderate data sizes
- Avoids database setup complexity for initial development
- Session state ensures data persists across page navigation within a user session

**Future Considerations:** For production use with large datasets, consider integrating a database (PostgreSQL, MySQL) or data warehouse solution.

### Machine Learning Pipeline

**Supervised Learning for Churn Prediction:**
1. **Data Preparation:** Train-test split (default 80/20)
2. **Model Training:** Sequential training of multiple model types
3. **Evaluation:** Cross-validation and performance metrics calculation
4. **Prediction:** Generate churn probabilities for individual customers

**Model Selection Strategy:**
- Start with interpretable models (Logistic Regression) for baseline
- Progress to ensemble methods (Random Forest) for improved accuracy
- Apply gradient boosting (XGBoost, LightGBM) for high performance
- Optional deep learning (ANNs, RNNs) for complex patterns

**Unsupervised Learning for Segmentation:**
- K-Means for general customer grouping (user-defined cluster count)
- DBSCAN for outlier/anomaly detection
- Feature-based clustering on behavioral and transactional attributes

**Time Series Forecasting:**
- Facebook Prophet for sales prediction
- Handles seasonality and trends automatically
- Configurable forecast periods

**Design Choice Rationale:**
- Multiple model options allow users to balance interpretability vs. accuracy
- Ensemble methods generally provide robust performance across diverse datasets
- Prophet simplifies time series forecasting without requiring extensive statistical expertise

## External Dependencies

### Python Libraries

**Data Processing & Analysis:**
- pandas: DataFrame operations and CSV handling
- numpy: Numerical computations and array operations

**Machine Learning & AI:**
- scikit-learn: Traditional ML models, preprocessing, metrics
- xgboost: Gradient boosting classifier
- lightgbm: Gradient boosting framework
- tensorflow/keras: Deep learning (ANNs, RNNs)
- prophet (fbprophet): Time series forecasting

**Visualization:**
- plotly: Interactive charts and graphs
- matplotlib: Static visualizations
- seaborn: Statistical data visualization

**Web Framework:**
- streamlit: Web application framework and UI components

### Data Requirements

**Input Data Format:** CSV files with expected columns:

1. **Customer Data** (Required):
   - Customer identifiers
   - Demographic information
   - Subscription/engagement metrics
   - Churn indicator (or synthesized from behavior)

2. **Transaction Data** (Recommended):
   - Transaction/purchase dates
   - Amount/revenue values
   - Customer identifiers for joining

3. **Product Data** (Optional):
   - Product identifiers and details
   - Used for enhanced analysis

**Design Choice Rationale:**
- CSV format provides universal compatibility with business systems
- Flexible column detection accommodates varying data schemas
- Graceful fallbacks when optional data is missing

### No Database Currently Required

This application operates entirely in-memory without requiring database setup. All data is loaded from CSV files and stored in Streamlit session state during user sessions.