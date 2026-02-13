import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st

class MLModels:
    def __init__(self):
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.ann_model = None
        self.rnn_model = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_churn_data(self, processed_features, target_column='churn'):
        """Prepare data for churn prediction"""
        try:
            if processed_features is None:
                st.error("No processed features available")
                return False
            
            # Check if target column exists
            if target_column not in processed_features.columns:
                # Try to find a suitable target column
                possible_targets = ['churn', 'churned', 'is_churn', 'target', 'label']
                target_column = None
                for col in possible_targets:
                    if col in processed_features.columns:
                        target_column = col
                        break
                
                if target_column is None:
                    # Create a synthetic churn target based on engagement metrics
                    st.warning("No churn column found. Creating synthetic churn labels based on engagement metrics.")
                    return self._create_synthetic_churn_target(processed_features)
            
            # Prepare features and target
            X = processed_features.drop(columns=[target_column])
            y = processed_features[target_column]
            
            # Ensure binary classification
            if y.nunique() > 2:
                st.warning("Converting multi-class target to binary (churned vs not churned)")
                y = (y > y.median()).astype(int)
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.feature_names = X.columns.tolist()
            st.success(f"Data prepared for training: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
            return True
            
        except Exception as e:
            st.error(f"Error preparing churn data: {str(e)}")
            return False
    
    def _create_synthetic_churn_target(self, processed_features):
        """Create synthetic churn target based on available features"""
        try:
            # Create churn based on engagement metrics
            features_for_churn = []
            
            # Use engagement score if available
            if 'engagement_score' in processed_features.columns:
                features_for_churn.append('engagement_score')
            
            # Use purchase frequency if available
            if 'purchase_frequency' in processed_features.columns:
                features_for_churn.append('purchase_frequency')
            
            # Use recency if available
            if 'days_since_last_purchase' in processed_features.columns:
                features_for_churn.append('days_since_last_purchase')
            
            if features_for_churn:
                # Create composite score
                scores = processed_features[features_for_churn].copy()
                
                # Normalize scores
                for col in scores.columns:
                    if col == 'days_since_last_purchase':
                        # Higher days = higher churn probability
                        scores[col] = (scores[col] - scores[col].min()) / (scores[col].max() - scores[col].min() + 1e-8)
                    else:
                        # Lower engagement/frequency = higher churn probability
                        scores[col] = 1 - (scores[col] - scores[col].min()) / (scores[col].max() - scores[col].min() + 1e-8)
                
                churn_score = scores.mean(axis=1)
                # Create binary churn target (top 30% are considered churned)
                churn_threshold = churn_score.quantile(0.7)
                processed_features['churn'] = (churn_score >= churn_threshold).astype(int)
                
                # Prepare data
                X = processed_features.drop(columns=['churn'])
                y = processed_features['churn']
                
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                self.feature_names = X.columns.tolist()
                st.success("Synthetic churn target created and data prepared for training")
                return True
            else:
                st.error("Insufficient features to create churn target")
                return False
                
        except Exception as e:
            st.error(f"Error creating synthetic churn target: {str(e)}")
            return False
    
    def train_churn_models(self):
        """Train churn prediction models"""
        if self.X_train is None or self.y_train is None:
            st.error("No training data available")
            return False
        
        try:
            # Train Logistic Regression
            self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
            self.lr_model.fit(self.X_train, self.y_train)
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(self.X_train, self.y_train)
            
            st.success("Churn prediction models trained successfully")
            return True
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return False
    
    def train_advanced_models(self):
        """Train advanced ML models (XGBoost and LightGBM)"""
        if self.X_train is None or self.y_train is None:
            st.error("No training data available")
            return False
        
        try:
            # Train XGBoost
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            self.xgb_model.fit(self.X_train, self.y_train)
            
            # Train LightGBM
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self.lgb_model.fit(self.X_train, self.y_train)
            
            st.success("Advanced models (XGBoost, LightGBM) trained successfully")
            return True
            
        except Exception as e:
            st.error(f"Error training advanced models: {str(e)}")
            return False
    
    def train_deep_learning_models(self):
        """Train deep learning models (ANN and RNN)"""
        if self.X_train is None or self.y_train is None:
            st.error("No training data available")
            return False
        
        try:
            input_dim = self.X_train.shape[1]
            
            # Build ANN model
            self.ann_model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            self.ann_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train ANN
            self.ann_model.fit(
                self.X_train, self.y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Build RNN model (using time-based features if available)
            # Reshape data for RNN (samples, timesteps, features)
            X_train_rnn = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
            
            self.rnn_model = keras.Sequential([
                layers.LSTM(64, activation='relu', input_shape=(1, input_dim), return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            self.rnn_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train RNN
            self.rnn_model.fit(
                X_train_rnn, self.y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            st.success("Deep learning models (ANN, RNN) trained successfully")
            return True
            
        except Exception as e:
            st.error(f"Error training deep learning models: {str(e)}")
            return False
    
    def get_model_performance(self):
        """Get performance metrics for all trained models"""
        if self.lr_model is None or self.rf_model is None:
            return None
        
        try:
            # Predictions
            lr_pred = self.lr_model.predict(self.X_test)
            rf_pred = self.rf_model.predict(self.X_test)
            
            # Metrics
            metrics = {
                'Logistic Regression': {
                    'accuracy': accuracy_score(self.y_test, lr_pred),
                    'precision': precision_score(self.y_test, lr_pred, average='weighted'),
                    'recall': recall_score(self.y_test, lr_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, lr_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, lr_pred)
                },
                'Random Forest': {
                    'accuracy': accuracy_score(self.y_test, rf_pred),
                    'precision': precision_score(self.y_test, rf_pred, average='weighted'),
                    'recall': recall_score(self.y_test, rf_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, rf_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, rf_pred)
                }
            }
            
            # Add XGBoost metrics if trained
            if self.xgb_model is not None:
                xgb_pred = self.xgb_model.predict(self.X_test)
                metrics['XGBoost'] = {
                    'accuracy': accuracy_score(self.y_test, xgb_pred),
                    'precision': precision_score(self.y_test, xgb_pred, average='weighted'),
                    'recall': recall_score(self.y_test, xgb_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, xgb_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, xgb_pred)
                }
            
            # Add LightGBM metrics if trained
            if self.lgb_model is not None:
                lgb_pred = self.lgb_model.predict(self.X_test)
                metrics['LightGBM'] = {
                    'accuracy': accuracy_score(self.y_test, lgb_pred),
                    'precision': precision_score(self.y_test, lgb_pred, average='weighted'),
                    'recall': recall_score(self.y_test, lgb_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, lgb_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, lgb_pred)
                }
            
            # Add ANN metrics if trained
            if self.ann_model is not None:
                ann_pred_proba = self.ann_model.predict(self.X_test, verbose=0)
                ann_pred = (ann_pred_proba > 0.5).astype(int).flatten()
                metrics['ANN (Deep Learning)'] = {
                    'accuracy': accuracy_score(self.y_test, ann_pred),
                    'precision': precision_score(self.y_test, ann_pred, average='weighted'),
                    'recall': recall_score(self.y_test, ann_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, ann_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, ann_pred)
                }
            
            # Add RNN metrics if trained
            if self.rnn_model is not None:
                X_test_rnn = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
                rnn_pred_proba = self.rnn_model.predict(X_test_rnn, verbose=0)
                rnn_pred = (rnn_pred_proba > 0.5).astype(int).flatten()
                metrics['RNN (Deep Learning)'] = {
                    'accuracy': accuracy_score(self.y_test, rnn_pred),
                    'precision': precision_score(self.y_test, rnn_pred, average='weighted'),
                    'recall': recall_score(self.y_test, rnn_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, rnn_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, rnn_pred)
                }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return None
    
    def predict_churn_probabilities(self, data=None):
        """Get churn probabilities for customers"""
        if self.rf_model is None:
            st.error("No trained model available")
            return None
        
        try:
            if data is None:
                data = self.X_test
            
            probabilities = self.rf_model.predict_proba(data)[:, 1]  # Probability of churn
            predictions = self.rf_model.predict(data)
            
            results = pd.DataFrame({
                'churn_probability': probabilities,
                'churn_prediction': predictions
            })
            
            return results
            
        except Exception as e:
            st.error(f"Error predicting churn probabilities: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if self.rf_model is None or self.feature_names is None:
            return None
        
        try:
            importance = self.rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def perform_customer_segmentation(self, data, n_clusters=4):
        """Perform K-Means clustering for customer segmentation"""
        try:
            if data is None or data.empty:
                st.error("No data available for clustering")
                return None
            
            # Prepare data for clustering (use only numerical features)
            numerical_data = data.select_dtypes(include=[np.number])
            
            if numerical_data.empty:
                st.error("No numerical features available for clustering")
                return None
            
            # Perform K-Means clustering
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.kmeans_model.fit_predict(numerical_data)
            
            # Create results dataframe
            results = numerical_data.copy()
            results['cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = results.groupby('cluster').agg(['mean', 'count']).round(2)
            
            return {
                'clustered_data': results,
                'cluster_centers': self.kmeans_model.cluster_centers_,
                'cluster_stats': cluster_stats,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            st.error(f"Error performing customer segmentation: {str(e)}")
            return None
    
    def perform_anomaly_detection(self, data, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering for anomaly detection"""
        try:
            if data is None or data.empty:
                st.error("No data available for anomaly detection")
                return None
            
            # Prepare data for clustering (use only numerical features)
            numerical_data = data.select_dtypes(include=[np.number])
            
            if numerical_data.empty:
                st.error("No numerical features available for anomaly detection")
                return None
            
            # Standardize the data for DBSCAN
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_data)
            
            # Perform DBSCAN clustering
            self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = self.dbscan_model.fit_predict(scaled_data)
            
            # Create results dataframe
            results = numerical_data.copy()
            results['cluster'] = cluster_labels
            
            # Identify outliers (cluster label -1)
            outliers = results[results['cluster'] == -1]
            normal_clusters = results[results['cluster'] != -1]
            
            # Calculate statistics
            n_outliers = len(outliers)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            return {
                'clustered_data': results,
                'outliers': outliers,
                'normal_clusters': normal_clusters,
                'n_outliers': n_outliers,
                'n_clusters': n_clusters,
                'outlier_percentage': (n_outliers / len(results) * 100) if len(results) > 0 else 0
            }
            
        except Exception as e:
            st.error(f"Error performing anomaly detection: {str(e)}")
            return None
