import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")

st.title("📈 Sales Forecasting")
st.markdown("Predict future sales trends using Facebook Prophet time series forecasting.")

# Check if data is available
if 'data_processor' not in st.session_state:
    st.error("❌ No data processor available. Please upload your data first.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Go to Data Upload", icon="📁")
    st.stop()

# Check for transaction data
if st.session_state.data_processor.transactions is None:
    st.warning("⚠️ No transaction data available. Sales forecasting requires transaction data.")
    st.info("Please upload transaction data in the Data Upload page to enable sales forecasting.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Upload Transaction Data", icon="📁")
    st.stop()

transactions = st.session_state.data_processor.transactions

# Find relevant columns
amount_col = None
date_col = None

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

if amount_col is None or date_col is None:
    st.error("❌ Required columns not found. Need date and amount columns for forecasting.")
    st.stop()

# Ensure date column is datetime
transactions[date_col] = pd.to_datetime(transactions[date_col], errors='coerce')
transactions = transactions.dropna(subset=[date_col])

# Forecast configuration
st.header("⚙️ Forecasting Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    forecast_periods = st.slider(
        "Forecast Periods (days)",
        min_value=7,
        max_value=365,
        value=90,
        help="Number of days to forecast into the future"
    )

with col2:
    aggregation = st.selectbox(
        "Aggregation Level",
        options=['Daily', 'Weekly', 'Monthly'],
        index=1
    )

with col3:
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Training Prophet model and generating forecast..."):
            try:
                # Prepare data for Prophet
                if aggregation == 'Daily':
                    df_prophet = transactions.groupby(transactions[date_col].dt.date)[amount_col].sum().reset_index()
                    df_prophet.columns = ['ds', 'y']
                    freq = 'D'
                elif aggregation == 'Weekly':
                    df_prophet = transactions.groupby(transactions[date_col].dt.to_period('W'))[amount_col].sum().reset_index()
                    df_prophet[date_col] = df_prophet[date_col].dt.to_timestamp()
                    df_prophet.columns = ['ds', 'y']
                    freq = 'W'
                else:  # Monthly
                    df_prophet = transactions.groupby(transactions[date_col].dt.to_period('M'))[amount_col].sum().reset_index()
                    df_prophet[date_col] = df_prophet[date_col].dt.to_timestamp()
                    df_prophet.columns = ['ds', 'y']
                    freq = 'M'
                
                # Train Prophet model
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=(aggregation == 'Daily'),
                    yearly_seasonality=True
                )
                model.fit(df_prophet)
                
                # Make future dataframe
                future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
                
                # Generate forecast
                forecast = model.predict(future)
                
                # Store results in session state
                st.session_state.forecast_results = {
                    'forecast': forecast,
                    'model': model,
                    'historical': df_prophet,
                    'aggregation': aggregation
                }
                
                st.success("✅ Forecast generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

# Display forecast results
if hasattr(st.session_state, 'forecast_results'):
    results = st.session_state.forecast_results
    forecast = results['forecast']
    historical = results['historical']
    aggregation = results['aggregation']
    
    st.markdown("---")
    st.header("📊 Forecast Results")
    
    # Forecast visualization
    st.subheader("📈 Sales Forecast")
    
    # Create interactive plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['ds'],
        y=historical['y'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Forecast
    forecast_future = forecast[forecast['ds'] > historical['ds'].max()]
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
        y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{aggregation} Sales Forecast',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast statistics
    st.subheader("📊 Forecast Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_historical = historical['y'].mean()
        st.metric("Avg Historical Sales", f"${avg_historical:,.2f}")
    
    with col2:
        avg_forecast = forecast_future['yhat'].mean()
        st.metric("Avg Forecast Sales", f"${avg_forecast:,.2f}")
    
    with col3:
        growth_rate = ((avg_forecast - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
        st.metric("Projected Growth", f"{growth_rate:+.1f}%")
    
    with col4:
        total_forecast = forecast_future['yhat'].sum()
        st.metric("Total Forecast Revenue", f"${total_forecast:,.2f}")
    
    # Trend analysis
    st.subheader("📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trend component
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='green')
        ))
        fig_trend.update_layout(
            title='Sales Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Trend Value',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Yearly seasonality if available
        if 'yearly' in forecast.columns:
            fig_seasonal = go.Figure()
            fig_seasonal.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly'],
                mode='lines',
                name='Yearly Seasonality',
                line=dict(color='purple')
            ))
            fig_seasonal.update_layout(
                title='Yearly Seasonality Pattern',
                xaxis_title='Date',
                yaxis_title='Seasonal Effect',
                height=400
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Forecast table
    st.subheader("📋 Forecast Details")
    
    forecast_display = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    forecast_display['Date'] = forecast_display['Date'].dt.date
    
    # Round values
    for col in ['Forecast', 'Lower Bound', 'Upper Bound']:
        forecast_display[col] = forecast_display[col].round(2)
    
    st.dataframe(forecast_display, use_container_width=True)
    
    # Business insights
    st.subheader("💡 Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Key Findings:**")
        
        # Analyze trend
        trend_direction = "increasing" if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else "decreasing"
        st.write(f"• Sales trend is {trend_direction}")
        
        # Best/worst periods
        best_period = forecast_future.loc[forecast_future['yhat'].idxmax(), 'ds'].strftime('%Y-%m-%d')
        worst_period = forecast_future.loc[forecast_future['yhat'].idxmin(), 'ds'].strftime('%Y-%m-%d')
        
        st.write(f"• Highest forecast: {best_period}")
        st.write(f"• Lowest forecast: {worst_period}")
        
        # Growth projection
        if growth_rate > 0:
            st.write(f"• Projected growth of {growth_rate:.1f}% indicates positive outlook")
        else:
            st.write(f"• Projected decline of {abs(growth_rate):.1f}% suggests need for intervention")
    
    with col2:
        st.write("**Recommendations:**")
        
        if growth_rate > 5:
            st.write("• Capitalize on growth momentum")
            st.write("• Increase inventory to meet demand")
            st.write("• Consider expansion opportunities")
        elif growth_rate > 0:
            st.write("• Maintain current strategies")
            st.write("• Monitor market conditions closely")
            st.write("• Optimize operational efficiency")
        else:
            st.write("• Review and adjust marketing strategies")
            st.write("• Analyze customer retention metrics")
            st.write("• Consider promotional campaigns")
    
    # Export forecast
    st.markdown("---")
    st.subheader("💾 Export Forecast")
    
    export_data = forecast_display.copy()
    csv_data = export_data.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv_data,
        file_name=f"sales_forecast_{aggregation.lower()}.csv",
        mime="text/csv"
    )

# Navigation
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/1_Data_Upload.py", label="📁 Data Upload", icon="📁")
with col2:
    st.page_link("pages/2_Churn_Prediction.py", label="🔮 Churn Prediction", icon="🔮")
with col3:
    st.page_link("pages/3_Customer_Segmentation.py", label="👥 Customer Segmentation", icon="👥")
with col4:
    st.page_link("pages/4_Sales_Analysis.py", label="📈 Sales Analysis", icon="📈")
