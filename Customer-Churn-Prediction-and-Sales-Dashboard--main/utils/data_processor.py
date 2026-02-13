import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.customers = None
        self.transactions = None
        self.products = None
        self.processed_features = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_customer_data(self, uploaded_file):
        """Load and validate customer data"""
        try:
            self.customers = pd.read_csv(uploaded_file)
            st.success(f"Customer data loaded: {len(self.customers)} records")
            return True
        except Exception as e:
            st.error(f"Error loading customer data: {str(e)}")
            return False
    
    def load_transaction_data(self, uploaded_file):
        """Load and validate transaction data"""
        try:
            self.transactions = pd.read_csv(uploaded_file)
            # Convert date columns if they exist
            date_columns = ['date', 'transaction_date', 'created_at', 'timestamp']
            for col in date_columns:
                if col in self.transactions.columns:
                    self.transactions[col] = pd.to_datetime(self.transactions[col], errors='coerce')
            st.success(f"Transaction data loaded: {len(self.transactions)} records")
            return True
        except Exception as e:
            st.error(f"Error loading transaction data: {str(e)}")
            return False
    
    def load_product_data(self, uploaded_file):
        """Load and validate product data"""
        try:
            self.products = pd.read_csv(uploaded_file)
            st.success(f"Product data loaded: {len(self.products)} records")
            return True
        except Exception as e:
            st.error(f"Error loading product data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Comprehensive data preprocessing and feature engineering"""
        if self.customers is None:
            st.error("No customer data available for preprocessing")
            return False
        
        try:
            # Create a copy for processing
            processed_data = self.customers.copy()
            
            # Handle missing values in numerical columns
            numerical_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                imputer_num = SimpleImputer(strategy='median')
                processed_data[numerical_cols] = imputer_num.fit_transform(processed_data[numerical_cols])
            
            # Handle missing values in categorical columns
            categorical_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                processed_data[categorical_cols] = imputer_cat.fit_transform(processed_data[categorical_cols])
            
            # Feature Engineering
            self._engineer_features(processed_data)
            
            # Encode categorical variables
            for col in categorical_cols:
                if col != 'customer_id' and col in processed_data.columns:
                    le = LabelEncoder()
                    processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                    self.label_encoders[col] = le
            
            # Remove non-feature columns
            feature_cols = [col for col in processed_data.columns 
                          if col not in ['customer_id', 'id', 'name', 'email']]
            
            if feature_cols:
                features_for_scaling = processed_data[feature_cols].select_dtypes(include=[np.number])
                if not features_for_scaling.empty:
                    # Scale numerical features
                    scaled_features = self.scaler.fit_transform(features_for_scaling)
                    scaled_df = pd.DataFrame(scaled_features, columns=features_for_scaling.columns, 
                                           index=features_for_scaling.index)
                    
                    # Combine with categorical features
                    non_numerical = processed_data[feature_cols].select_dtypes(exclude=[np.number])
                    self.processed_features = pd.concat([scaled_df, non_numerical], axis=1)
                else:
                    self.processed_features = processed_data[feature_cols]
            else:
                st.warning("No suitable feature columns found for processing")
                return False
            
            st.success("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return False
    
    def _engineer_features(self, data):
        """Create engineered features"""
        try:
            # If transaction data is available, create transaction-based features
            if self.transactions is not None and 'customer_id' in data.columns:
                customer_id_col = 'customer_id'
                
                # Find matching customer ID column in transactions
                trans_id_cols = ['customer_id', 'user_id', 'id']
                trans_customer_col = None
                for col in trans_id_cols:
                    if col in self.transactions.columns:
                        trans_customer_col = col
                        break
                
                if trans_customer_col:
                    # Purchase frequency
                    purchase_freq = self.transactions.groupby(trans_customer_col).size().reset_index(name='purchase_frequency')
                    data = data.merge(purchase_freq, left_on=customer_id_col, right_on=trans_customer_col, how='left')
                    data['purchase_frequency'] = data['purchase_frequency'].fillna(0)
                    
                    # Total amount spent
                    if 'amount' in self.transactions.columns:
                        total_spent = self.transactions.groupby(trans_customer_col)['amount'].sum().reset_index(name='total_spent')
                        data = data.merge(total_spent, left_on=customer_id_col, right_on=trans_customer_col, how='left')
                        data['total_spent'] = data['total_spent'].fillna(0)
                        
                        # Average transaction value
                        data['avg_transaction_value'] = np.where(data['purchase_frequency'] > 0,
                                                               data['total_spent'] / data['purchase_frequency'], 0)
                    
                    # Days since last purchase
                    date_col = None
                    for col in ['date', 'transaction_date', 'created_at', 'timestamp']:
                        if col in self.transactions.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        last_purchase = self.transactions.groupby(trans_customer_col)[date_col].max().reset_index(name='last_purchase_date')
                        data = data.merge(last_purchase, left_on=customer_id_col, right_on=trans_customer_col, how='left')
                        data['days_since_last_purchase'] = (pd.Timestamp.now() - data['last_purchase_date']).dt.days
                        data['days_since_last_purchase'] = data['days_since_last_purchase'].fillna(365)  # Default to 1 year
            
            # Create engagement score based on available features
            engagement_features = []
            
            if 'purchase_frequency' in data.columns:
                engagement_features.append('purchase_frequency')
            if 'total_spent' in data.columns:
                engagement_features.append('total_spent')
            if 'days_since_last_purchase' in data.columns:
                # Invert days since last purchase (more recent = higher engagement)
                data['recency_score'] = np.where(data['days_since_last_purchase'] > 0,
                                               1 / (1 + data['days_since_last_purchase'] / 30), 0)
                engagement_features.append('recency_score')
            
            if engagement_features:
                # Normalize engagement features and create composite score
                for feature in engagement_features:
                    if data[feature].std() > 0:
                        data[f'{feature}_normalized'] = (data[feature] - data[feature].mean()) / data[feature].std()
                    else:
                        data[f'{feature}_normalized'] = 0
                
                normalized_features = [f'{feature}_normalized' for feature in engagement_features]
                data['engagement_score'] = data[normalized_features].mean(axis=1)
            
            # Subscription tenure (if date columns are available)
            date_columns = ['created_at', 'signup_date', 'registration_date', 'start_date']
            for col in date_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                        data['subscription_tenure_days'] = (pd.Timestamp.now() - data[col]).dt.days
                        data['subscription_tenure_days'] = data['subscription_tenure_days'].fillna(0)
                        break
                    except:
                        continue
            
            # Update the original data
            for col in data.columns:
                if col not in self.customers.columns:
                    self.customers[col] = data[col]
                    
        except Exception as e:
            st.warning(f"Feature engineering warning: {str(e)}")
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        summary = {}
        
        if self.customers is not None:
            summary['customers'] = {
                'shape': self.customers.shape,
                'columns': self.customers.columns.tolist(),
                'missing_values': self.customers.isnull().sum().sum(),
                'data_types': self.customers.dtypes.value_counts().to_dict()
            }
        
        if self.transactions is not None:
            summary['transactions'] = {
                'shape': self.transactions.shape,
                'columns': self.transactions.columns.tolist(),
                'missing_values': self.transactions.isnull().sum().sum(),
                'data_types': self.transactions.dtypes.value_counts().to_dict()
            }
        
        if self.products is not None:
            summary['products'] = {
                'shape': self.products.shape,
                'columns': self.products.columns.tolist(),
                'missing_values': self.products.isnull().sum().sum(),
                'data_types': self.products.dtypes.value_counts().to_dict()
            }
        
        return summary
