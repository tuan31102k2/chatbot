import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.markdown import centered_subheader, centered_title

import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

def show_info_of_a_column(df):
	tmp = df.isnull().sum().to_frame().T
	list_nunique = []
	list_dtypes = []
	list_null_values = []
	# Get list dtypes
	for col in df.columns:
		list_dtypes.append(df[col].dtype)
		list_nunique.append(df[col].nunique())
		list_null_values.append(df.isnull().sum()[col])
	tmp_df = pd.DataFrame({"# Null values": list_null_values, "Data Type": list_dtypes, "# unique values": list_nunique}, index=df.columns)
	return tmp_df
	
def visualize_categorical(df):
	categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
	column_name = st.selectbox("Select a column", categorical_columns)
    # Đếm tần suất các giá trị trong cột phân loại
	value_counts = df[column_name].value_counts().reset_index()
	value_counts.columns = [column_name, 'count']

    # Lựa chọn kiểu visualization
	chart_type = st.selectbox("Type of Chart", 
                              ["Bar Chart", "Horizontal Bar Chart", "Pie Chart", "Donut Chart", "Sunburst Chart"])

    # Biểu đồ thanh
	if chart_type == "Bar Chart":
		fig = px.bar(value_counts, x=column_name, y='count', title=f'Bar Chart of {column_name}',
                     labels={column_name: column_name, 'count': 'Count'}, color=column_name)
    
    # Biểu đồ thanh ngang
	elif chart_type == "Horizontal Bar Chart":
		fig = px.bar(value_counts, y=column_name, x='count', orientation='h', title=f'Horizontal Bar Chart of {column_name}',
                     labels={column_name: column_name, 'count': 'Count'}, color=column_name)
    
    # Biểu đồ tròn
	elif chart_type == "Pie Chart":
		fig = px.pie(value_counts, names=column_name, values='count', title=f'Pie Chart of {column_name}', color=column_name)
    
    # Biểu đồ donut
	elif chart_type == "Donut Chart":
		fig = px.pie(value_counts, names=column_name, values='count', hole=0.5, title=f'Donut Chart of {column_name}', color=column_name)
    
    # Biểu đồ sunburst
	elif chart_type == "Sunburst Chart":
		fig = px.sunburst(df, path=[column_name], values=None, title=f'Sunburst Chart of {column_name}', color=column_name)
    
    # Hiển thị biểu đồ
	st.plotly_chart(fig)

def visualize_numerical(df):
	numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
	column_name = st.selectbox("Select a column", numerical_columns)
	# Lựa chọn kiểu visualization
	chart_type = st.selectbox("Type of Chart", 
                              ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot"])

    # Biểu đồ phân phối (histogram)
	if chart_type == "Histogram":
		fig = px.histogram(df, x=column_name, title=f'Histogram of {column_name}',
                           labels={column_name: column_name}, nbins=30)
    
    # Biểu đồ hộp (box plot)
	elif chart_type == "Box Plot":
		fig = px.box(df, y=column_name, title=f'Box Plot of {column_name}', 
                     labels={column_name: column_name}, points="all")
    
    # Biểu đồ violin
	elif chart_type == "Violin Plot":
		fig = px.violin(df, y=column_name, title=f'Violin Plot of {column_name}', 
                        labels={column_name: column_name}, box=True, points="all")
    
    # Biểu đồ scatter (scatter plot) 
	elif chart_type == "Scatter Plot":
		# yêu cầu chọn thêm cột số khác để làm trục Y
		y_column = st.selectbox("Select Y-axis", numerical_columns, index=0)
		fig = px.scatter(df, x=column_name, y=y_column, title=f'Scatter Plot of {column_name} vs {y_column}',
                         labels={column_name: column_name, y_column: y_column})
    
    # Hiển thị biểu đồ
	st.plotly_chart(fig)
	
def visualization():
	# Đọc dữ liệu
	df = pd.read_csv(r'D:\FALL 2024\DAP391m\chatbotGemini\data\MentalHealthSurvey.csv')

	# Tạo tiêu đề cho Dashboard
	centered_title("Mental Health Student Analysis Dashboard")
	# Hiển thị bảng dữ liệu ban đầu
	
	centered_subheader("Data Frame")
	num_rows = st.slider(label="Number of rows", min_value=0, max_value=len(df))
	st.write(df.head(num_rows))
	
	col1v1, col2v1 = st.columns(spec=[0.3, 0.46], gap='medium')
	with col1v1:
		centered_subheader("Information of each column")
		st.table(show_info_of_a_column(df))
	
	with col2v1:
		centered_subheader("Data Frame Describe")
		is_include_all = st.checkbox("Including categorical columns")
		st.table(df.describe(include=('all' if is_include_all == True else None)).T)
	
	col1, col2 = st.columns(spec=[0.9, 0.9], gap="large")
	with col1:
		centered_subheader("Visualization of Categorical Column")
		visualize_categorical(df)
	with col2:
		centered_subheader("Visualization of Numerical Column")
		visualize_numerical(df)
	

	centered_subheader("Bivariate Analysis")
	cols = st.columns(2)
	# Danh sách các biến về sức khỏe tâm lý
	mental_health_vars = ['depression', 'anxiety', 'isolation', 'future_insecurity']
    # Vòng lặp qua các biến và tạo biểu đồ
	for i, var in enumerate(mental_health_vars):
        # Chọn cột hiển thị
		with cols[i % 2]:
			fig = px.histogram(df, x='university', color=var, 
                               barmode='group', title=f'University vs {var.replace("_", " ").title()}',
                               labels={'university': 'University', 'count': 'Count'}, 
                               color_discrete_sequence=px.colors.qualitative.Set1)
			st.plotly_chart(fig)
	
	col1v2, col2v2 = st.columns(2)
	
	with col1v2:
		# Biểu đồ bar về mức độ trầm cảm theo giới tính
		centered_subheader("Mức độ trầm cảm theo giới tính")
		fig_bar = px.bar(df, x='gender', y='depression', title="Mức độ trầm cảm theo giới tính", 
                 labels={'gender': 'Giới tính', 'depression': 'Mức độ trầm cảm'},
                 color='gender', barmode='group', height=400)
		st.plotly_chart(fig_bar)
		
	with col2v2:
		centered_subheader('Mức độ lo âu theo năm học')
		fig_box = px.box(df, x='academic_year', y='anxiety', title='Mức độ lo âu theo năm học',
                 labels={'academic_year': 'Năm học', 'anxiety': 'Mức độ lo âu'}, 
                 color='academic_year', height=400)
		st.plotly_chart(fig_box)
		
	corr_matrix = df.corr(numeric_only=True)

	centered_subheader("Correlation Matrix of Numerical Columns")
	# Tạo heatmap với Plotly
	fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,  # Ma trận tương quan
        x=corr_matrix.columns,  # Tên cột
        y=corr_matrix.columns,  # Tên hàng
        colorscale='RdBu',  # Bảng màu
        zmin=-1, zmax=1,  # Giá trị min/max cho màu sắc
        colorbar=dict(title="Correlation")  # Thanh màu chú thích
    	)
	)

	# Thêm tiêu đề và tùy chỉnh layout
	fig.update_layout(
		title="Correlation Matrix",
		xaxis_nticks=len(corr_matrix.columns),  # Số lượng ticks theo số cột
		yaxis_nticks=len(corr_matrix.columns),
		width=800, height=700,
	)

	# Hiển thị với Streamlit
	st.plotly_chart(fig)
		
def visualize_covid_19v2():
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
		map_style = st.selectbox("Select Map Style", ["OpenStreetMap", "Stamen Toner", "CartoDB positron", "CartoDB dark_matter"])
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

def visualize_covid_19():
    # Đọc dữ liệu từ file CSV
    file_path = r'D:\FALL 2024\DAP391m\chatbotGemini\data\Covid19\covid_19_data.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("File CSV không tìm thấy. Vui lòng kiểm tra đường dẫn file.")
        return

    # Kiểm tra các cột cần thiết
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.title("Advanced Geospatial Data Visualization Dashboard")

        # Bộ lọc theo cột danh mục (VD: quốc gia hoặc khu vực)
        category_column = st.selectbox("Select Category Column", df.select_dtypes(include=['object']).columns)
        if category_column:
            category_value = st.selectbox("Select Category Value", df[category_column].dropna().unique())
            filtered_df = df[df[category_column] == category_value]
        else:
            st.error("Vui lòng chọn một cột danh mục hợp lệ.")
            return

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