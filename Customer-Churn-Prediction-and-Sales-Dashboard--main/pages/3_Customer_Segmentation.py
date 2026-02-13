import streamlit as st
import pandas as pd
import numpy as np
from utils.ml_models import MLModels
from utils.visualizations import Visualizations

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation")
st.markdown("Discover customer segments using K-Means clustering for targeted marketing strategies.")

# Initialize components if not exists
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = MLModels()
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = Visualizations()

# Check if data is available
if 'data_processor' not in st.session_state or st.session_state.data_processor.processed_features is None:
    st.error("❌ No processed data available. Please upload and preprocess your data first.")
    st.page_link("pages/1_Data_Upload.py", label="📁 Go to Data Upload", icon="📁")
    st.stop()

# Clustering method selection
st.header("⚙️ Clustering Configuration")

clustering_method = st.radio(
    "Select Clustering Method",
    options=["K-Means Clustering", "DBSCAN Anomaly Detection"],
    horizontal=True
)

if clustering_method == "K-Means Clustering":
    col1, col2, col3 = st.columns(3)

    with col1:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=4,
            help="Choose the number of customer segments to create"
        )

    with col2:
        st.write("**Clustering Method:**")
        st.info("K-Means Clustering")
        st.write("Groups customers based on similarity in behavior and characteristics")

    with col3:
        if st.button("🎯 Perform Clustering", type="primary"):
            with st.spinner("Performing customer segmentation..."):
                clustering_results = st.session_state.ml_models.perform_customer_segmentation(
                    st.session_state.data_processor.processed_features, 
                    n_clusters=n_clusters
                )
                
                if clustering_results:
                    st.session_state.clustering_results = clustering_results
                    st.session_state.clustering_method = 'kmeans'
                    st.success("✅ Customer segmentation completed!")
                    st.rerun()
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        eps = st.slider(
            "Epsilon (neighborhood distance)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Maximum distance between two samples to be in same neighborhood"
        )
    
    with col2:
        min_samples = st.slider(
            "Minimum Samples",
            min_value=2,
            max_value=20,
            value=5,
            help="Minimum samples in a neighborhood for a point to be core point"
        )
    
    with col3:
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Performing anomaly detection..."):
                anomaly_results = st.session_state.ml_models.perform_anomaly_detection(
                    st.session_state.data_processor.processed_features,
                    eps=eps,
                    min_samples=min_samples
                )
                
                if anomaly_results:
                    st.session_state.clustering_results = anomaly_results
                    st.session_state.clustering_method = 'dbscan'
                    st.success("✅ Anomaly detection completed!")
                    st.rerun()

# Clustering results
if hasattr(st.session_state, 'clustering_results') and st.session_state.clustering_results:
    results = st.session_state.clustering_results
    clustered_data = results['clustered_data']
    method = st.session_state.get('clustering_method', 'kmeans')
    
    st.markdown("---")
    
    # Display results based on method
    if method == 'dbscan':
        st.header("🔍 Anomaly Detection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(clustered_data))
        
        with col2:
            st.metric("Anomalies Detected", results['n_outliers'])
        
        with col3:
            st.metric("Normal Clusters", results['n_clusters'])
        
        with col4:
            st.metric("Anomaly Rate", f"{results['outlier_percentage']:.1f}%")
        
        # Anomaly details
        st.subheader("⚠️ Anomalous Customers")
        
        if results['n_outliers'] > 0:
            outliers = results['outliers']
            
            st.write(f"**{len(outliers)} customers detected as anomalies:**")
            st.write("These customers exhibit unusual behavior patterns that deviate significantly from normal segments.")
            
            # Show outlier statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Outlier Characteristics:**")
                numerical_cols = outliers.select_dtypes(include=[np.number]).columns.tolist()
                numerical_cols = [col for col in numerical_cols if col != 'cluster']
                if numerical_cols:
                    outlier_means = outliers[numerical_cols].mean()
                    normal_means = results['normal_clusters'][numerical_cols].mean()
                    
                    for col in numerical_cols[:5]:  # Show top 5
                        diff_pct = ((outlier_means[col] - normal_means[col]) / normal_means[col] * 100) if normal_means[col] != 0 else 0
                        if abs(diff_pct) > 10:
                            st.write(f"• **{col}**: {diff_pct:+.1f}% vs normal")
            
            with col2:
                st.write("**Sample Anomalies:**")
                sample_outliers = outliers.head(5)
                st.dataframe(sample_outliers, use_container_width=True)
            
            # Visualization
            st.subheader("📊 Anomaly Visualization")
            numerical_cols = clustered_data.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != 'cluster']
            
            if len(numerical_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    feature1 = st.selectbox("Select X-axis feature", options=numerical_cols, index=0, key="anomaly_x")
                
                with col2:
                    feature2 = st.selectbox("Select Y-axis feature", options=numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="anomaly_y")
                
                import plotly.express as px
                clustered_data_viz = clustered_data.copy()
                clustered_data_viz['type'] = clustered_data_viz['cluster'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
                
                fig = px.scatter(
                    clustered_data_viz,
                    x=feature1,
                    y=feature2,
                    color='type',
                    title=f'Anomaly Detection: {feature1} vs {feature2}',
                    color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected with current parameters. Try adjusting epsilon and minimum samples.")
    
    else:
        cluster_stats = results['cluster_stats']
        st.header("📊 Segmentation Results")
    
    # Cluster overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", results['n_clusters'])
    
    with col2:
        st.metric("Total Customers", len(clustered_data))
    
    with col3:
        cluster_sizes = clustered_data['cluster'].value_counts()
        largest_cluster = cluster_sizes.max()
        st.metric("Largest Cluster", largest_cluster)
    
    with col4:
        smallest_cluster = cluster_sizes.min()
        st.metric("Smallest Cluster", smallest_cluster)
    
    # Cluster distribution
    st.subheader("📈 Cluster Distribution")
    
    cluster_counts = clustered_data['cluster'].value_counts().sort_index()
    cluster_percentages = (cluster_counts / len(clustered_data) * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        distribution_data = pd.DataFrame({
            'Cluster': [f'Cluster {i}' for i in cluster_counts.index],
            'Count': cluster_counts.values,
            'Percentage': cluster_percentages.values
        })
        st.dataframe(distribution_data, use_container_width=True)
    
    with col2:
        import plotly.express as px
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='Customer Distribution by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis visualization
    st.subheader("🔍 Cluster Analysis")
    
    # Get numerical columns for visualization
    numerical_cols = clustered_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'cluster']
    
    if len(numerical_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox(
                "Select X-axis feature",
                options=numerical_cols,
                index=0
            )
        
        with col2:
            feature2 = st.selectbox(
                "Select Y-axis feature", 
                options=numerical_cols,
                index=1 if len(numerical_cols) > 1 else 0
            )
        
        if feature1 and feature2:
            cluster_plot = st.session_state.visualizations.plot_cluster_analysis(
                clustered_data, feature1, feature2
            )
            st.plotly_chart(cluster_plot, use_container_width=True)
    
    # Cluster statistics
    st.subheader("📊 Cluster Statistics")
    
    cluster_stats_plot = st.session_state.visualizations.plot_cluster_statistics(cluster_stats)
    st.plotly_chart(cluster_stats_plot, use_container_width=True)
    
    # Detailed cluster profiles
    st.subheader("🎯 Cluster Profiles")
    
    # Create cluster profiles with meaningful descriptions
    for cluster_id in sorted(clustered_data['cluster'].unique()):
        with st.expander(f"📋 Cluster {cluster_id} Profile", expanded=False):
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(clustered_data) * 100)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Cluster Size", cluster_size)
                st.metric("Percentage", f"{cluster_percentage:.1f}%")
            
            with col2:
                st.write("**Key Characteristics:**")
                
                # Get cluster statistics
                cluster_means = cluster_data[numerical_cols].mean()
                overall_means = clustered_data[numerical_cols].mean()
                
                # Identify distinguishing features
                differences = ((cluster_means - overall_means) / overall_means * 100).abs()
                top_differences = differences.nlargest(5)
                
                for feature in top_differences.index:
                    cluster_val = cluster_means[feature]
                    overall_val = overall_means[feature]
                    diff_pct = ((cluster_val - overall_val) / overall_val * 100)
                    
                    if diff_pct > 10:
                        st.write(f"• **{feature}**: {diff_pct:+.1f}% above average")
                    elif diff_pct < -10:
                        st.write(f"• **{feature}**: {diff_pct:+.1f}% below average")
                    else:
                        st.write(f"• **{feature}**: Near average")
            
            # Show sample data
            if st.checkbox(f"Show sample data for Cluster {cluster_id}", key=f"sample_{cluster_id}"):
                sample_size = min(10, len(cluster_data))
                sample_data = cluster_data.sample(n=sample_size, random_state=42)
                st.dataframe(sample_data, use_container_width=True)
    
    # Segmentation insights and recommendations
    st.markdown("---")
    st.header("💡 Segmentation Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Marketing Strategies by Segment")
        
        for cluster_id in sorted(clustered_data['cluster'].unique()):
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            st.write(f"**Cluster {cluster_id}** ({cluster_size} customers)")
            
            # Generate recommendations based on cluster characteristics
            cluster_means = cluster_data[numerical_cols].mean()
            
            # Simple rule-based recommendations
            recommendations = []
            
            if 'purchase_frequency' in cluster_means.index:
                if cluster_means['purchase_frequency'] > clustered_data['purchase_frequency'].median():
                    recommendations.append("High-frequency buyers - Focus on loyalty programs")
                else:
                    recommendations.append("Low-frequency buyers - Use re-engagement campaigns")
            
            if 'total_spent' in cluster_means.index:
                if cluster_means['total_spent'] > clustered_data['total_spent'].median():
                    recommendations.append("High-value customers - Offer premium services")
                else:
                    recommendations.append("Price-sensitive customers - Provide discounts")
            
            if 'engagement_score' in cluster_means.index:
                if cluster_means['engagement_score'] > clustered_data['engagement_score'].median():
                    recommendations.append("Highly engaged - Perfect for new product launches")
                else:
                    recommendations.append("Low engagement - Focus on improving experience")
            
            if not recommendations:
                recommendations = ["Develop targeted marketing based on cluster characteristics"]
            
            for rec in recommendations:
                st.write(f"  • {rec}")
            
            st.write("")
    
    with col2:
        st.subheader("📈 Business Opportunities")
        
        st.write("**Cross-segment Analysis:**")
        
        # Identify opportunities
        if 'total_spent' in numerical_cols:
            high_value_clusters = clustered_data.groupby('cluster')['total_spent'].mean().nlargest(2).index
            st.write(f"• Focus retention efforts on Clusters {', '.join(map(str, high_value_clusters))}")
        
        if 'purchase_frequency' in numerical_cols:
            high_freq_clusters = clustered_data.groupby('cluster')['purchase_frequency'].mean().nlargest(2).index
            st.write(f"• Upsell opportunities in Clusters {', '.join(map(str, high_freq_clusters))}")
        
        if 'engagement_score' in numerical_cols:
            low_eng_clusters = clustered_data.groupby('cluster')['engagement_score'].mean().nsmallest(2).index
            st.write(f"• Re-engagement needed for Clusters {', '.join(map(str, low_eng_clusters))}")
        
        st.write("")
        st.write("**Implementation Steps:**")
        st.write("1. Create segment-specific marketing messages")
        st.write("2. Develop tailored product offerings")
        st.write("3. Set up automated campaign triggers")
        st.write("4. Monitor segment performance metrics")
        st.write("5. Adjust strategies based on results")

# Export clustering results
if hasattr(st.session_state, 'clustering_results'):
    st.markdown("---")
    st.subheader("💾 Export Segmentation Results")
    
    # Prepare download data
    export_data = st.session_state.clustering_results['clustered_data'].copy()
    export_data['customer_index'] = export_data.index
    
    # Create a summary version
    summary_cols = ['customer_index', 'cluster']
    if 'total_spent' in export_data.columns:
        summary_cols.append('total_spent')
    if 'purchase_frequency' in export_data.columns:
        summary_cols.append('purchase_frequency')
    if 'engagement_score' in export_data.columns:
        summary_cols.append('engagement_score')
    
    summary_data = export_data[summary_cols]
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_full = export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Clustering Results",
            data=csv_full,
            file_name="customer_segmentation_full.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_summary = summary_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Segmentation Summary",
            data=csv_summary,
            file_name="customer_segmentation_summary.csv",
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
    st.page_link("pages/4_Sales_Analysis.py", label="📈 Sales Analysis", icon="📈")
