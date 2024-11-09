from utils.markdown import centered_subheader, centered_title
import streamlit as st
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

# model
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

x_train, x_test, y_train, y_test = None, None, None, None
def get_data():
	FILE_PATH = r'D:\FALL 2024\DAP391m\chatbotGemini\data\Mental_Health\final.csv'
	data = pd.read_csv(FILE_PATH)
	return data

def split_data(df):
	centered_subheader("Train test split")
	test_size_percent = st.slider(label="Select test size (%)", min_value=0, max_value=100)
	# Convert into float number
	test_size = test_size_percent / 100
	isShuffle = st.checkbox("Shuffle the train set")
	if test_size != 0:
		train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=isShuffle)
		with st.spinner("Splitting..."):
			time.sleep(3)
			st.success("Train test split successful!!!")
		train_col, test_col = st.columns(2)
		with train_col:
			centered_subheader(f"Training Set {100 - test_size_percent}%")
			st.write(train)
		with test_col:
			centered_subheader(f"Testing Set {test_size_percent}%")
			st.write(test)
			
def solving_missing_method(df, column_name, kind):
	if kind is not None:
		if kind == 'Drop column':
			df = df.drop([column_name], axis=1)
		else:
			if column_name == "Profession":
				mode_val = df['Profession'].mode()[0]
				df[column_name] = df[column_name].fillna(value = mode_val)
			else:
				if kind == 'Fill by mean value':
					mean_val = df[column_name].mean()
					df[column_name] = df[column_name].fillna(value=mean_val)
				elif kind == "Linear Interpolation":
					df[column_name] = df[column_name].interpolate(method='linear')
	return df
	
def fill_missing_value(df):
	st.subheader("Missing Values")
	# numerical_missing = {"Column": [], "Percentage of missing value":[]}
	# categorical_missing = {"Column": [], "Percentage of missing value":[]}
	missing_cols = {"Column": [], "Percentage of missing value":[], "Data Type": []}
	numerical_method = {"Academic Pressure": None,
					 	"Work Pressure": None,
						"CGPA": None,
						"Study Satisfaction": None,
						"Job Satisfaction": None}
	
	categorical_method = {"Profession": None}
	
	for col in df.columns:
		missing_val = df[col].isnull().sum()
		missing_val_percent = (missing_val / len(df)) * 100
		if missing_val > 0:
			missing_cols["Column"].append(col)
			missing_cols['Percentage of missing value'].append(f"{missing_val_percent:.4f}%")
			missing_cols["Data Type"].append(df[col].dtype)

	st.write(pd.DataFrame(missing_cols))
	numerical_options = ["None", "Drop column", "Fill by mean value", "Linear Interpolation"]
	# Academic Pressure 
	numerical_method["Academic Pressure"] = st.selectbox("Academic Pressure", numerical_options)
		
	# Work Pressure
	numerical_method["Work Pressure"] = st.selectbox("Work Pressure", numerical_options)
		
	# CGPA
	numerical_method["CGPA"] = st.selectbox("CGPA", numerical_options)
		
	# Study Satisfaction
	numerical_method["Study Satisfaction"] = st.selectbox("Study Satisfaction", numerical_options)

	# Job Satisfaction
	numerical_method["Job Satisfaction"] = st.selectbox("Job Satisfaction", numerical_options)
		
	for col, method in numerical_method.items():
		df = solving_missing_method(df, col, method)	

	categorical_options = ["None", "Drop column", "Fill by mode value"]
	# Profession
	categorical_method["Profession"] = st.selectbox("Profession", categorical_options)

	for col, method in categorical_method.items():
		df = solving_missing_method(df, col, method)	
		
	return df

def preprocess_data(df):
	col1, col2 = st.columns(2)
	with col1:
		centered_subheader("Data Scaling")
		list_method_normalization = ["None", 'Simple Feature Scaling', 'Min-max Scaling', 'Z-score']
		method_normalization = st.selectbox("Select a method", list_method_normalization)
		
		for col in df.select_dtypes(include=['float64', 'int64']).columns.to_list():
			if method_normalization == 'Simple Feature Scaling':
				max_val = df[col].max()
				df[col] = df[col] / max_val
			elif method_normalization == 'Min-max Scaling':
				max_val = df[col].max()
				min_val = df[col].min()
				df[col] = (df[col] - min_val) / (max_val - min_val)
			elif method_normalization == 'Z-score':
				std_scl = StandardScaler()
				df[col] = std_scl.fit_transform(df[[col]])
		
	with col2:
		centered_subheader("Turning Categorical into Numerical")
		list_method_convert = ["None", 'One Hot Encoding', "Label Encoding"]
		method_converting = st.selectbox("Select a method", list_method_convert)
		if method_converting == 'Label Encoding':
			le = LabelEncoder()
			for col in df.select_dtypes(include='object').columns.to_list():
				df[col] = le.fit_transform(df[col])
		elif method_converting == 'One Hot Encoding':
			pass	
	
	return df

def split_data(df):
	centered_subheader("Train test split")
	test_size_percent = st.slider(label="Select test size (%)", min_value=0, max_value=100)
	# Convert into float number
	test_size = test_size_percent / 100
	isShuffle = st.checkbox("Shuffle the train set")
	# x_train, y_train = None, None
	isSplit = st.button("Split")
	if isSplit:
		train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=isShuffle)
		with st.spinner("Splitting..."):
			time.sleep(3)
			st.success("Train test split successful!!!")
		train_col, test_col = st.columns(2)
		with train_col:
			centered_subheader(f"Training Set {100 - test_size_percent}%")
			st.write(train)
		with test_col:
			centered_subheader(f"Testing Set {test_size_percent}%")
			st.write(test)
		x_train, y_train = train.drop(['Depression'], axis=1), train['Depression']
		x_test, y_test = train.drop(['Depression'], axis=1), test['Depression']
		return x_train, x_test, y_train, y_test
	
def predict(x_train, x_test, y_train, y_test):
	list_models = {
					"Logistic Regression": LogisticRegression(max_iter=1000),
					"Random Forest": RandomForestClassifier(),
					"Naive Bayes": GaussianNB(),
					# "Perceptron": Perceptron(max_iter=1000),
					"Decision Tree": DecisionTreeClassifier(),
					"KNN": KNeighborsClassifier(),
					"Gradient Boosting": GradientBoostingClassifier(),
					"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
					"CatBoost": CatBoostClassifier(verbose=0),
					"LGBM": LGBMClassifier(),
					"AdaBoost": AdaBoostClassifier()}
	st.write(x_train, y_train)
	centered_subheader("Prediction")
	model_name = st.selectbox("Select a model", list_models.keys())
	isTrain = st.button("Fit Data")
	if isTrain:
		model = list_models[model_name]
		model.fit(x_train, y_train)
		with st.spinner("Training ..."):
			time.sleep(5)
		y_pred = model.predict(x_test)	
			
def visualize_model():
	centered_title("Model Interpretation")
	df = get_data()
	df = fill_missing_value(df)
	solving_missing_data = st.button("Apply")
	if solving_missing_data:
		with st.spinner("Solving missing data ..."):
			time.sleep(3)
			st.success("Successfully")
		centered_subheader("After solving missing value")
		st.write(df)
	df = preprocess_data(df)
	preprocess_data_button = st.button("Preprocess")
	if preprocess_data_button:
		with st.spinner("Preprocessing data ..."):
			time.sleep(3)
			st.success("Successfully")
		centered_subheader("After preprocessing data")
		st.write(df)
	
	# x_train, x_test, y_train, y_test = split_data(df)
	# applied = st.button("Apply")
	# if applied:
	# 	predict(x_train, x_test, y_train, y_test)	
	centered_subheader("Train test split")
	test_size_percent = st.slider(label="Select test size (%)", min_value=0, max_value=100)
	# Convert into float number
	test_size = test_size_percent / 100
	isShuffle = st.checkbox("Shuffle the train set")
	# x_train, y_train = None, None
	isSplit = st.button("Split")
	if isSplit:
		train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=isShuffle)
		with st.spinner("Splitting..."):
			time.sleep(3)
			st.success("Train test split successful!!!")
		train_col, test_col = st.columns(2)
		with train_col:
			centered_subheader(f"Training Set {100 - test_size_percent}%")
			st.write(train)
		with test_col:
			centered_subheader(f"Testing Set {test_size_percent}%")
			st.write(test)
		x_train, y_train = train.drop(['Depression'], axis=1), train['Depression']
		x_test, y_test = train.drop(['Depression'], axis=1), test['Depression']
		list_models = {
						"Logistic Regression": LogisticRegression(max_iter=1000),
						"Random Forest": RandomForestClassifier(),
						"Naive Bayes": GaussianNB(),
						# "Perceptron": Perceptron(max_iter=1000),
						"Decision Tree": DecisionTreeClassifier(),
						"KNN": KNeighborsClassifier(),
						"Gradient Boosting": GradientBoostingClassifier(),
						"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
						"CatBoost": CatBoostClassifier(verbose=0),
						"LGBM": LGBMClassifier(),
						"AdaBoost": AdaBoostClassifier()}
		centered_subheader("Prediction")
		model_name = st.selectbox("Select a model", list_models.keys())
		isTrain = st.button("Fit Data")
		if isTrain:
			model = list_models[model_name]
			model.fit(x_train, y_train)
			with st.spinner("Training ..."):
				time.sleep(5)
			y_pred = model.predict(x_test)


