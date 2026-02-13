import streamlit as st
import pandas as pd
import numpy as np
from utils.visualizations import Visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sales Analysis", page_icon="📈", layout="wide")

st.title("📈 Sales Trend Analysis")
st.markdown("Analyze sales performance, trends, and revenue insights across different time periods.")

# Initialize components if not exists
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = Visualizations()

# Check if data is available
if 'data_processor' not in st.session_state:
    st.error("❌ No data processor available. Please upload your data first.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Go to Data Upload", icon="📁")
    st.stop()

# Check for transaction data
if st.session_state.data_processor.transactions is None:
    st.warning("⚠️ No transaction data available. Sales analysis requires transaction data.")
    st.info("Please upload transaction data in the Data Upload page to enable sales analysis.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Upload Transaction Data", icon="📁")
    st.stop()

transactions = st.session_state.data_processor.transactions

# Sales overview
st.header("💰 Sales Overview")

# Find relevant columns
amount_col = None
date_col = None
product_col = None
customer_col = None

# Find amount column
for col in ['amount', 'total', 'price', 'value', 'revenue']:
    if col in transactions.columns:
        amount_col = col
        break

# Find date column
for col in ['date', 'transaction_date', 'created_at', 'timestamp', 'purchase_date']:
    if col in transactions.columns:
        date_col = col
        break

# Find product column
for col in ['product_id', 'product', 'item_id', 'item', 'product_name']:
    if col in transactions.columns:
        product_col = col
        break

# Find customer column
for col in ['customer_id', 'user_id', 'customer', 'user']:
    if col in transactions.columns:
        customer_col = col
        break

if amount_col is None:
    st.error("❌ No amount/revenue column found in transaction data.")
    st.write("Expected column names: amount, total, price, value, revenue")
    st.stop()

# Ensure date column is datetime
if date_col:
    transactions[date_col] = pd.to_datetime(transactions[date_col], errors='coerce')

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = transactions[amount_col].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")

with col2:
    total_transactions = len(transactions)
    st.metric("Total Transactions", f"{total_transactions:,}")

with col3:
    avg_transaction = transactions[amount_col].mean()
    st.metric("Average Transaction", f"${avg_transaction:.2f}")

with col4:
    if customer_col:
        unique_customers = transactions[customer_col].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")
    else:
        st.metric("Data Points", f"{len(transactions):,}")

# Sales trends
if date_col:
    st.markdown("---")
    st.header("📊 Sales Trends")
    
    # Time period selection
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox(
            "Select Time Period",
            options=['Daily', 'Monthly', 'Quarterly', 'Yearly'],
            index=1
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            options=['Line Chart', 'Bar Chart', 'Area Chart'],
            index=0
        )
    
    # Create time-based aggregation
    transactions_copy = transactions.copy()
    
    if period == 'Daily':
        transactions_copy['period'] = transactions_copy[date_col].dt.date
        period_key = 'D'
    elif period == 'Monthly':
        transactions_copy['period'] = transactions_copy[date_col].dt.to_period('M')
        period_key = 'M'
    elif period == 'Quarterly':
        transactions_copy['period'] = transactions_copy[date_col].dt.to_period('Q')
        period_key = 'Q'
    else:  # Yearly
        transactions_copy['period'] = transactions_copy[date_col].dt.to_period('Y')
        period_key = 'Y'
    
    # Aggregate sales data
    sales_trend = transactions_copy.groupby('period').agg({
        amount_col: ['sum', 'count', 'mean']
    }).round(2)
    
    sales_trend.columns = ['total_revenue', 'transaction_count', 'avg_transaction']
    sales_trend = sales_trend.reset_index()
    sales_trend['period_str'] = sales_trend['period'].astype(str)
    
    # Create trend visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{period} Revenue Trend',
            f'{period} Transaction Count',
            f'{period} Average Transaction Value',
            'Revenue vs Transaction Count'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Revenue trend
    if chart_type == 'Line Chart':
        fig.add_trace(
            go.Scatter(x=sales_trend['period_str'], y=sales_trend['total_revenue'],
                      mode='lines+markers', name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
    elif chart_type == 'Bar Chart':
        fig.add_trace(
            go.Bar(x=sales_trend['period_str'], y=sales_trend['total_revenue'],
                  name='Revenue', marker_color='blue'),
            row=1, col=1
        )
    else:  # Area Chart
        fig.add_trace(
            go.Scatter(x=sales_trend['period_str'], y=sales_trend['total_revenue'],
                      fill='tonexty', name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Transaction count trend
    fig.add_trace(
        go.Scatter(x=sales_trend['period_str'], y=sales_trend['transaction_count'],
                  mode='lines+markers', name='Transactions', line=dict(color='green')),
        row=1, col=2
    )
    
    # Average transaction trend
    fig.add_trace(
        go.Scatter(x=sales_trend['period_str'], y=sales_trend['avg_transaction'],
                  mode='lines+markers', name='Avg Transaction', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Revenue vs Transaction Count
    fig.add_trace(
        go.Scatter(x=sales_trend['period_str'], y=sales_trend['total_revenue'],
                  mode='lines+markers', name='Revenue', line=dict(color='blue')),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=sales_trend['period_str'], y=sales_trend['transaction_count'],
                  mode='lines+markers', name='Transactions', line=dict(color='red'),
                  yaxis='y2'),
        row=2, col=2, secondary_y=True
    )
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth analysis
    if len(sales_trend) > 1:
        st.subheader("📈 Growth Analysis")
        
        sales_trend['revenue_growth'] = sales_trend['total_revenue'].pct_change() * 100
        sales_trend['transaction_growth'] = sales_trend['transaction_count'].pct_change() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_revenue_growth = sales_trend['revenue_growth'].mean()
            st.metric("Avg Revenue Growth", f"{avg_revenue_growth:.1f}%")
        
        with col2:
            avg_transaction_growth = sales_trend['transaction_growth'].mean()
            st.metric("Avg Transaction Growth", f"{avg_transaction_growth:.1f}%")
        
        with col3:
            latest_growth = sales_trend['revenue_growth'].iloc[-1]
            st.metric("Latest Period Growth", f"{latest_growth:.1f}%")
        
        # Growth trend chart
        growth_fig = go.Figure()
        
        growth_fig.add_trace(
            go.Scatter(x=sales_trend['period_str'], y=sales_trend['revenue_growth'],
                      mode='lines+markers', name='Revenue Growth %', line=dict(color='blue'))
        )
        
        growth_fig.add_trace(
            go.Scatter(x=sales_trend['period_str'], y=sales_trend['transaction_growth'],
                      mode='lines+markers', name='Transaction Growth %', line=dict(color='green'))
        )
        
        growth_fig.add_hline(y=0, line_dash="dash", line_color="red")
        growth_fig.update_layout(
            title=f'{period} Growth Rates',
            xaxis_title='Period',
            yaxis_title='Growth Rate (%)',
            height=400
        )
        
        st.plotly_chart(growth_fig, use_container_width=True)

# Product analysis
if product_col:
    st.markdown("---")
    st.header("🛍️ Product Performance")
    
    # Product metrics
    product_performance = transactions.groupby(product_col).agg({
        amount_col: ['sum', 'count', 'mean']
    }).round(2)
    
    product_performance.columns = ['total_revenue', 'total_sales', 'avg_price']
    product_performance = product_performance.reset_index()
    product_performance = product_performance.sort_values('total_revenue', ascending=False)
    
    # Top products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Products by Revenue")
        top_products_revenue = product_performance.head(10)
        
        fig = px.bar(
            top_products_revenue,
            x='total_revenue',
            y=product_col,
            orientation='h',
            title='Top 10 Products by Revenue',
            labels={'total_revenue': 'Total Revenue ($)'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📦 Top Products by Sales Volume")
        top_products_volume = product_performance.nlargest(10, 'total_sales')
        
        fig = px.bar(
            top_products_volume,
            x='total_sales',
            y=product_col,
            orientation='h',
            title='Top 10 Products by Sales Volume',
            labels={'total_sales': 'Total Sales Count'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Product performance table
    st.subheader("📊 Product Performance Summary")
    
    # Add performance metrics
    total_revenue = product_performance['total_revenue'].sum()
    product_performance['revenue_share'] = (product_performance['total_revenue'] / total_revenue * 100).round(2)
    product_performance['cumulative_share'] = product_performance['revenue_share'].cumsum()
    
    # Display top 20 products
    display_products = product_performance.head(20)
    st.dataframe(
        display_products[[product_col, 'total_revenue', 'total_sales', 'avg_price', 'revenue_share']],
        use_container_width=True
    )
    
    # Pareto analysis
    st.subheader("📈 Pareto Analysis (80/20 Rule)")
    
    pareto_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    pareto_fig.add_trace(
        go.Bar(x=list(range(len(product_performance))), y=product_performance['total_revenue'],
              name='Revenue'),
        secondary_y=False,
    )
    
    pareto_fig.add_trace(
        go.Scatter(x=list(range(len(product_performance))), y=product_performance['cumulative_share'],
                  mode='lines+markers', name='Cumulative %', line=dict(color='red')),
        secondary_y=True,
    )
    
    pareto_fig.add_hline(y=80, line_dash="dash", line_color="green", secondary_y=True)
    
    pareto_fig.update_xaxes(title_text="Product Rank")
    pareto_fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    pareto_fig.update_yaxes(title_text="Cumulative Percentage (%)", secondary_y=True)
    pareto_fig.update_layout(title="Product Revenue Pareto Chart", height=500)
    
    st.plotly_chart(pareto_fig, use_container_width=True)
    
    # Find 80% products
    products_80 = len(product_performance[product_performance['cumulative_share'] <= 80])
    total_products = len(product_performance)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Products contributing 80% revenue", products_80)
    with col2:
        percentage_80 = (products_80 / total_products * 100)
        st.metric("Percentage of total products", f"{percentage_80:.1f}%")
    with col3:
        st.metric("Total unique products", total_products)

# Customer analysis
if customer_col:
    st.markdown("---")
    st.header("👥 Customer Analysis")
    
    # Customer metrics
    customer_performance = transactions.groupby(customer_col).agg({
        amount_col: ['sum', 'count', 'mean']
    }).round(2)
    
    customer_performance.columns = ['total_spent', 'total_transactions', 'avg_transaction']
    customer_performance = customer_performance.reset_index()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_customer_value = customer_performance['total_spent'].mean()
        st.metric("Average Customer Value", f"${avg_customer_value:.2f}")
    
    with col2:
        avg_transactions_per_customer = customer_performance['total_transactions'].mean()
        st.metric("Avg Transactions per Customer", f"{avg_transactions_per_customer:.1f}")
    
    with col3:
        repeat_customers = len(customer_performance[customer_performance['total_transactions'] > 1])
        repeat_rate = (repeat_customers / len(customer_performance) * 100)
        st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
    
    # Customer value distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            customer_performance,
            x='total_spent',
            nbins=50,
            title='Customer Value Distribution',
            labels={'total_spent': 'Total Spent ($)', 'count': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            customer_performance,
            x='total_transactions',
            nbins=30,
            title='Customer Transaction Frequency Distribution',
            labels={'total_transactions': 'Number of Transactions', 'count': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top customers
    st.subheader("🌟 Top Customers")
    
    top_customers = customer_performance.nlargest(20, 'total_spent')
    
    fig = px.bar(
        top_customers,
        x=customer_col,
        y='total_spent',
        title='Top 20 Customers by Total Spent',
        labels={'total_spent': 'Total Spent ($)'}
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# Revenue correlation with churn (if churn prediction is available)
if hasattr(st.session_state, 'ml_models') and st.session_state.ml_models.rf_model is not None:
    st.markdown("---")
    st.header("🔗 Revenue Impact of Churn")
    
    # Get churn predictions
    predictions = st.session_state.ml_models.predict_churn_probabilities()
    
    if predictions is not None and customer_col:
        # Merge with customer performance data
        customer_performance_indexed = customer_performance.set_index(customer_col)
        predictions_with_revenue = predictions.copy()
        
        # Map customer performance to predictions
        if len(predictions_with_revenue) <= len(customer_performance_indexed):
            # Assuming predictions index corresponds to customer order
            customer_ids = customer_performance_indexed.index[:len(predictions_with_revenue)]
            predictions_with_revenue['customer_id'] = customer_ids
            predictions_with_revenue = predictions_with_revenue.merge(
                customer_performance_indexed, left_on='customer_id', right_index=True, how='left'
            )
            
            # Revenue at risk analysis
            high_risk_customers = predictions_with_revenue[predictions_with_revenue['churn_probability'] > 0.7]
            
            if len(high_risk_customers) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    revenue_at_risk = high_risk_customers['total_spent'].sum()
                    st.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")
                
                with col2:
                    total_customer_revenue = customer_performance['total_spent'].sum()
                    risk_percentage = (revenue_at_risk / total_customer_revenue * 100)
                    st.metric("Percentage of Total Revenue", f"{risk_percentage:.1f}%")
                
                with col3:
                    avg_risk_customer_value = high_risk_customers['total_spent'].mean()
                    st.metric("Avg High-Risk Customer Value", f"${avg_risk_customer_value:.2f}")
                
                # Churn probability vs customer value
                fig = px.scatter(
                    predictions_with_revenue,
                    x='total_spent',
                    y='churn_probability',
                    color='churn_prediction',
                    title='Customer Value vs Churn Probability',
                    labels={
                        'total_spent': 'Total Spent ($)',
                        'churn_probability': 'Churn Probability',
                        'churn_prediction': 'Churn Prediction'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

# Export sales analysis
st.markdown("---")
st.subheader("💾 Export Sales Analysis")

col1, col2 = st.columns(2)

with col1:
    if date_col and 'sales_trend' in locals():
        csv_trends = sales_trend.to_csv(index=False)
        st.download_button(
            label="📥 Download Sales Trends",
            data=csv_trends,
            file_name="sales_trends.csv",
            mime="text/csv"
        )

with col2:
    if product_col and 'product_performance' in locals():
        csv_products = product_performance.to_csv(index=False)
        st.download_button(
            label="📥 Download Product Performance",
            data=csv_products,
            file_name="product_performance.csv",
            mime="text/csv"
        )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Data_Upload.py", label="📁 Data Upload", icon="📁")
with col2:
    st.page_link("pages/2_Churn_Prediction.py", label="🔮 Churn Prediction", icon="🔮")
with col3:
    st.page_link("pages/3_Customer_Segmentation.py", label="👥 Customer Segmentation", icon="👥")
