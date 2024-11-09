import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv(r'D:\FALL 2024\DAP391m\chatbotGemini\data\Mental_Health\train.csv')

# Function to display information of each column in the dataset
def show_info_of_a_column(df):
    list_nunique = [df[col].nunique() for col in df.columns]
    list_dtypes = [df[col].dtype for col in df.columns]
    list_null_values = [df[col].isnull().sum() for col in df.columns]
    
    info_df = pd.DataFrame({
        "Data Type": list_dtypes,
        "# Unique Values": list_nunique,
        "# Null Values": list_null_values
    }, index=df.columns)
    return info_df

# Function for visualizing categorical columns
def visualize_categorical(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    column_name = st.selectbox("Select a categorical column", categorical_columns)

    value_counts = df[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, 'count']

    chart_type = st.selectbox("Select chart type", 
                              ["Bar Chart", "Horizontal Bar Chart", "Pie Chart", "Donut Chart", "Sunburst Chart"])

    if chart_type == "Bar Chart":
        fig = px.bar(value_counts, x=column_name, y='count', title=f'Bar Chart of {column_name}', color=column_name)
    elif chart_type == "Horizontal Bar Chart":
        fig = px.bar(value_counts, y=column_name, x='count', orientation='h', title=f'Horizontal Bar Chart of {column_name}', color=column_name)
    elif chart_type == "Pie Chart":
        fig = px.pie(value_counts, names=column_name, values='count', title=f'Pie Chart of {column_name}', color=column_name)
    elif chart_type == "Donut Chart":
        fig = px.pie(value_counts, names=column_name, values='count', hole=0.5, title=f'Donut Chart of {column_name}', color=column_name)
    elif chart_type == "Sunburst Chart":
        fig = px.sunburst(df, path=[column_name], title=f'Sunburst Chart of {column_name}', color=column_name)

    st.plotly_chart(fig)

# Function for visualizing numerical columns
def visualize_numerical(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    column_name = st.selectbox("Select a numerical column", numerical_columns)

    chart_type = st.selectbox("Select chart type", 
                              ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot"])

    if chart_type == "Histogram":
        fig = px.histogram(df, x=column_name, title=f'Histogram of {column_name}', nbins=30)
    elif chart_type == "Box Plot":
        fig = px.box(df, y=column_name, title=f'Box Plot of {column_name}', points="all")
    elif chart_type == "Violin Plot":
        fig = px.violin(df, y=column_name, title=f'Violin Plot of {column_name}', box=True, points="all")
    elif chart_type == "Scatter Plot":
        y_column = st.selectbox("Select Y-axis", numerical_columns, index=0)
        fig = px.scatter(df, x=column_name, y=y_column, title=f'Scatter Plot of {column_name} vs {y_column}')

    st.plotly_chart(fig)

# Main visualization function
def visualization_dashboard(df):
    st.title("Mental Health Data Analysis Dashboard")

    # Display a subset of the DataFrame
    st.subheader("Dataset Overview")
    num_rows = st.slider("Select number of rows to display", 1, len(df), 5)
    st.dataframe(df.head(num_rows))

    # Column Info and Description
    st.subheader("Column Information")
    st.write(show_info_of_a_column(df))

    st.subheader("Data Description")
    include_categorical = st.checkbox("Include categorical columns in description")
    st.write(df.describe(include='all' if include_categorical else None).T)

    # Visualizations
    st.subheader("Categorical Data Visualization")
    visualize_categorical(df)

    st.subheader("Numerical Data Visualization")
    visualize_numerical(df)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmin=-1, zmax=1))
    fig.update_layout(width=700, height=700, title="Correlation Matrix")
    st.plotly_chart(fig)

# Run the visualization dashboard
visualization_dashboard(df)
