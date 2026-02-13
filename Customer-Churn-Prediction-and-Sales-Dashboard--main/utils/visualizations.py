import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class Visualizations:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set2
    
    def plot_confusion_matrix(self, cm, model_name):
        """Create confusion matrix heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Churned', 'Churned'],
            y=['Not Churned', 'Churned'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=15):
        """Create feature importance plot"""
        top_features = feature_importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_churn_distribution(self, churn_data):
        """Create churn distribution plot"""
        churn_counts = churn_data.value_counts()
        
        fig = px.pie(
            values=churn_counts.values,
            names=['Not Churned', 'Churned'],
            title='Customer Churn Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def plot_churn_probability_distribution(self, probabilities):
        """Create churn probability distribution"""
        fig = px.histogram(
            x=probabilities,
            nbins=50,
            title='Churn Probability Distribution',
            labels={'x': 'Churn Probability', 'y': 'Number of Customers'}
        )
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                     annotation_text="50% Threshold")
        
        return fig
    
    def plot_cluster_analysis(self, clustered_data, feature1, feature2):
        """Create cluster scatter plot"""
        fig = px.scatter(
            clustered_data,
            x=feature1,
            y=feature2,
            color='cluster',
            title=f'Customer Segmentation: {feature1} vs {feature2}',
            labels={'cluster': 'Cluster'},
            color_discrete_sequence=self.color_palette
        )
        
        return fig
    
    def plot_cluster_statistics(self, cluster_stats):
        """Create cluster statistics visualization"""
        # Prepare data for plotting
        stats_data = []
        for cluster in cluster_stats.index:
            for feature in cluster_stats.columns.get_level_values(0).unique():
                stats_data.append({
                    'cluster': f'Cluster {cluster}',
                    'feature': feature,
                    'mean_value': cluster_stats.loc[cluster, (feature, 'mean')],
                    'count': cluster_stats.loc[cluster, (feature, 'count')]
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create grouped bar chart
        fig = px.bar(
            stats_df,
            x='feature',
            y='mean_value',
            color='cluster',
            title='Cluster Analysis - Mean Values by Feature',
            labels={'mean_value': 'Mean Value', 'feature': 'Features'},
            barmode='group'
        )
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        
        return fig
    
    def plot_sales_trend(self, transactions_data, date_col='date', amount_col='amount', period='M'):
        """Create sales trend analysis"""
        try:
            # Ensure date column is datetime
            transactions_data[date_col] = pd.to_datetime(transactions_data[date_col], errors='coerce')
            
            # Group by period
            if period == 'M':
                period_name = 'Month'
                transactions_data['period'] = transactions_data[date_col].dt.to_period('M')
            elif period == 'Q':
                period_name = 'Quarter'
                transactions_data['period'] = transactions_data[date_col].dt.to_period('Q')
            elif period == 'Y':
                period_name = 'Year'
                transactions_data['period'] = transactions_data[date_col].dt.to_period('Y')
            else:
                period_name = 'Day'
                transactions_data['period'] = transactions_data[date_col].dt.date
            
            # Calculate sales by period
            sales_by_period = transactions_data.groupby('period')[amount_col].agg(['sum', 'count']).reset_index()
            sales_by_period['period_str'] = sales_by_period['period'].astype(str)
            
            # Create subplot with dual y-axis
            fig = make_subplots(
                specs=[[{"secondary_y": True}]],
                subplot_titles=[f'Sales Trend Analysis - {period_name}ly']
            )
            
            # Add sales amount line
            fig.add_trace(
                go.Scatter(
                    x=sales_by_period['period_str'],
                    y=sales_by_period['sum'],
                    mode='lines+markers',
                    name='Total Sales',
                    line=dict(color='blue')
                ),
                secondary_y=False,
            )
            
            # Add transaction count line
            fig.add_trace(
                go.Scatter(
                    x=sales_by_period['period_str'],
                    y=sales_by_period['count'],
                    mode='lines+markers',
                    name='Transaction Count',
                    line=dict(color='red')
                ),
                secondary_y=True,
            )
            
            # Update axes labels
            fig.update_xaxes(title_text=period_name)
            fig.update_yaxes(title_text="Sales Amount ($)", secondary_y=False)
            fig.update_yaxes(title_text="Number of Transactions", secondary_y=True)
            
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating sales trend plot: {str(e)}")
            return None
    
    def plot_top_products(self, transactions_data, product_data=None, top_n=10):
        """Create top products analysis"""
        try:
            product_col = None
            # Find product column
            for col in ['product_id', 'product', 'item_id', 'item']:
                if col in transactions_data.columns:
                    product_col = col
                    break
            
            if product_col is None:
                st.warning("No product column found in transaction data")
                return None
            
            # Calculate product performance
            product_performance = transactions_data.groupby(product_col).agg({
                'amount': ['sum', 'count', 'mean']
            }).round(2)
            
            product_performance.columns = ['total_revenue', 'total_transactions', 'avg_transaction']
            product_performance = product_performance.reset_index()
            
            # Get top products by revenue
            top_products = product_performance.nlargest(top_n, 'total_revenue')
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Top Products by Revenue', 'Top Products by Transaction Count'],
                vertical_spacing=0.15
            )
            
            # Top products by revenue
            fig.add_trace(
                go.Bar(
                    x=top_products[product_col],
                    y=top_products['total_revenue'],
                    name='Revenue',
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Top products by transaction count
            top_by_count = product_performance.nlargest(top_n, 'total_transactions')
            fig.add_trace(
                go.Bar(
                    x=top_by_count[product_col],
                    y=top_by_count['total_transactions'],
                    name='Transaction Count',
                    marker_color='green'
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(tickangle=-45)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating top products plot: {str(e)}")
            return None
    
    def plot_customer_value_distribution(self, customer_data):
        """Create customer value distribution analysis"""
        try:
            value_cols = ['total_spent', 'purchase_frequency', 'avg_transaction_value']
            available_cols = [col for col in value_cols if col in customer_data.columns]
            
            if not available_cols:
                st.warning("No customer value columns found")
                return None
            
            fig = make_subplots(
                rows=len(available_cols), cols=1,
                subplot_titles=[f'{col.replace("_", " ").title()} Distribution' for col in available_cols],
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(available_cols):
                fig.add_trace(
                    go.Histogram(
                        x=customer_data[col],
                        name=col.replace("_", " ").title(),
                        nbinsx=30
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=300*len(available_cols), showlegend=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating customer value distribution: {str(e)}")
            return None
    
    def plot_churn_risk_segments(self, predictions_data):
        """Create churn risk segments visualization"""
        try:
            # Create risk segments
            predictions_data['risk_segment'] = pd.cut(
                predictions_data['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            segment_counts = predictions_data['risk_segment'].value_counts()
            
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title='Customer Churn Risk Segments',
                labels={'x': 'Risk Level', 'y': 'Number of Customers'},
                color=segment_counts.index,
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange',
                    'High Risk': 'red'
                }
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating churn risk segments: {str(e)}")
            return None
