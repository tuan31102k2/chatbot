import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.markdown import centered_subheader, centered_title


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
                              ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Distribution of the column by Target"])

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
	
def compare_with_target(df):
	column_name = st.selectbox("Select a column", df.columns)
	fig = px.histogram(df, x='Depression', color=column_name, 
                               barmode='group', title=f'Depression vs {column_name.replace("_", " ").title()}',
                               labels={'Depression': 'Depression', 'count': 'Count'}, 
                               color_discrete_sequence=px.colors.qualitative.Set1)
	st.plotly_chart(fig)
	
def compare_with_numerical_with_target(df):
	numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
	column_name = st.selectbox("Select a numerical column", numerical_columns)
	fig = px.violin(df, x = 'Depression', y=column_name, title=f'Distribution of {column_name} by Target', 
                        labels={column_name: column_name}, box=True, points="all")
	st.plotly_chart(fig)

def visualizationv2():
	# Đọc dữ liệu
	df = pd.read_csv(r'D:\FALL 2024\DAP391m\chatbotGemini\data\Mental_Health\final.csv')

	# df.drop(['id'], axis=1, inplace=True)

	# Tạo tiêu đề cho Dashboard
	centered_title("Mental Health Dashboard")
	# Hiển thị bảng dữ liệu ban đầu
	
	centered_subheader("Data Frame")
	num_rows = st.slider(label="Number of rows", min_value=0, max_value=len(df))
	st.write(df.head(num_rows))
	df = df[:num_rows]
	
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
	
	col1v2, col2v2 = st.columns(spec=[0.45, 0.45], gap="large")
	with col1v2:
		centered_subheader("Compare with Target Column")
		
		compare_with_target(df)

	with col2v2:
		centered_subheader("Compare with Numerical Column with Target Column")
		
		compare_with_numerical_with_target(df)
		
	centered_subheader("Correlation Matrix of Numerical Columns")
	corr_matrix = df.corr(numeric_only=True)

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
	
	# predict_model(df)

def predict_model(df):
	centered_title("Model Interpretation")
	# Baseline
	centered_subheader("Missing Value")
	
			
	centered_subheader("Data Normalization")
	normal_list = ["Min max scale", "Standard Scale", "Normalization Scale"]
	chosen = st.selectbox("Select a technique that you want?", normal_list)
	if chosen == "Min max scale":
		pass
	elif chosen == 'Standard Scale':
		pass
	elif chosen == 'Normalization Scale':
		pass
	
	centered_subheader("Extract Feature")
	categorical_col = df.select_dtypes(include=['object']).columns.to_list()
	list_extract = {
				"OneHotEncoder": OneHotEncoder(),
				"OrdinalEncoder": OrdinalEncoder(), 
				"LabelEncode": LabelEncoder()
				}
	extraction_option = st.selectbox("Select a extraction technique", [None] + list(list_extract.keys()))
	# if extraction_option != 'None'
	if extraction_option == "OneHotEncoder":
			pass
	elif extraction_option == "OrdinalEncoder":
			pass
	elif extraction_option == "LabelEncoder":
			pass
	if extraction_option is not None:
		with st.spinner("Converting categorical to numerical..."):
			time.sleep(5)	
			st.success("Convert Successful!!!")
		
	
	models = {
			"Linear Regression" : ...,
			"Logistic Regression": ...,
		   	"Random Forest": ...,
			"XG Boost": ...,
			"Decision Tree": ..., 
			}
	