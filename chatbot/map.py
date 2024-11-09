import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

# Đọc dữ liệu từ file CSV
file_path = r'D:\FALL 2024\DAP391m\chatbotGemini\data\Covid19\covid_19_data.csv'
df = pd.read_csv(file_path)

# Kiểm tra các cột cần thiết
if 'latitude' in df.columns and 'longitude' in df.columns:
    st.title("Advanced Geospatial Data Visualization Dashboard")

    # Bộ lọc theo cột danh mục (VD: quốc gia hoặc khu vực)
    category_column = st.selectbox("Select Category Column", df.select_dtypes(include=['object']).columns)
    category_value = st.selectbox("Select Category Value", df[category_column].unique())
    filtered_df = df[df[category_column] == category_value]

    # Chọn lớp bản đồ
    map_style = st.selectbox("Select Map Style", ["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB positron", "CartoDB dark_matter"])
    m = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=5, tiles=map_style)

    # Thêm cụm điểm (MarkerCluster) để gom nhóm các điểm gần nhau
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row.get('location', 'Unknown Location')}: {row.get('cases', 'N/A')} cases",
            icon=folium.Icon(color="blue")
        ).add_to(marker_cluster)

    # Thêm CircleMarker với kích thước và màu sắc dựa trên một cột dữ liệu
    size_column = st.selectbox("Choose Column for Circle Size", [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])])
    color = st.color_picker("Choose Circle Color", "#FF5733")
    
    for _, row in filtered_df.iterrows():
        if not pd.isna(row[size_column]):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row[size_column] / 1000,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"{row.get('location', 'Unknown Location')}: {row[size_column]}"
            ).add_to(m)

    # Thêm Heatmap nếu dữ liệu có tính tập trung
    if st.checkbox("Show Heatmap"):
        heat_data = [[row['latitude'], row['longitude'], row[size_column]] for _, row in filtered_df.iterrows() if not pd.isna(row[size_column])]
        HeatMap(heat_data).add_to(m)

    # Tìm kiếm và định vị bản đồ tới một địa điểm cụ thể
    search_location = st.text_input("Search for a location")
    if search_location:
        location_data = filtered_df[filtered_df['location'].str.contains(search_location, case=False, na=False)]
        if not location_data.empty:
            lat, lon = location_data.iloc[0][['latitude', 'longitude']]
            folium.Marker(location=[lat, lon], popup=search_location, icon=folium.Icon(color="green")).add_to(m)
            m.location = [lat, lon]  # Cập nhật vị trí bản đồ

    # Hiển thị bản đồ trên Streamlit
    st_folium(m, width=800, height=600)

    # Tạo các biểu đồ phụ thuộc theo bộ lọc
    st.write(f"Statistics for {category_value}")
    st.write(filtered_df.describe())
else:
    st.error("Dữ liệu không chứa cột latitude và longitude.")
