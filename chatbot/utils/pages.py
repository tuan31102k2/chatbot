from utils.markdown import centered_subheader, centered_title, add_background
from utils.authentication import verify_user, register_user
from utils.chatbot import query_gemini_api
from utils.analysisv2 import visualizationv2
from utils.model import visualize_model

from utils.analysis import visualization, visualize_covid_19
import streamlit as st
from config import *

# Hàm hiển thị trang đăng ký
def register_page():
    centered_subheader("Account Register")

    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        if new_user == '' or new_password == '':
            st.warning("The account cannot be empty!")
        else:
            if register_user(new_user, new_password):
                st.success("Register Succesfully!")
            else:
                st.warning("Account is already exists!")

# Hàm hiển thị trang đăng nhập
def login_page():
    centered_subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True  # Đánh dấu trạng thái đăng nhập thành công
            st.session_state.username = username
            st.success(f"Login Successfully!")  # Hiển thị thông báo đăng nhập thành công
            st.experimental_rerun()
        else:
            st.warning('Username or password is invalid, please try again!')
            
# Hàm hiển thị chatbot sau khi đăng nhập thành công
def chatbot_page():
    st.subheader(f"Welcome to {st.session_state.username}")
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]['role'] != 'assistant':
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_gemini_api(prompt)
                st.write(response)
            
            message = {'role': "assistant", 'content': response}
            st.session_state.messages.append(message)
   
def analysis_mental_health_Student_page():
    visualization()
    
def analysis_mental_health_page():
    visualizationv2()
    
def model_page():
    visualize_model()

def covid19_map_page():
    visualize_covid_19()
    
def set_layout():
    # Khởi tạo session state nếu chưa có
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    st.set_page_config(layout="wide")
    add_background()
    # Add logo FPT Edu
    st.image(LOGO_IMG_PATH, width=200)
    
    # Kiểm tra trạng thái để hiển thị page tương ứng
    if st.session_state.logged_in:
        st.sidebar.title("Menu")
        menu = ["Chatbot", "Analysis Student Mental Health", "Mental Health", "Prediction Model", "Covid-19 Map","Exit"]
        choice = st.sidebar.selectbox("Select Page", menu)

        if choice == "Chatbot":
            chatbot_page()  # Hiển thị chatbot
        elif choice == "Analysis Student Mental Health":
            analysis_mental_health_Student_page()
        elif choice == 'Mental Health':
            analysis_mental_health_page()
        elif choice == "Prediction Model":
            model_page()
        elif choice == 'Covid-19 Map':
            covid19_map_page()
        elif choice == "Exit":
            if st.button("Quit"):
                st.session_state.logged_in = False  # Đặt lại trạng thái đăng nhập
                st.session_state.username = ""

                if 'chat_history' in st.session_state:
                    st.session_state['chat_history'] = []
                st.experimental_rerun()
    else:
        st.sidebar.title("Labs (DAP391m)")
        menu = ["Login", "Sign Up"]
        choice = st.sidebar.radio("Select Option", menu)
        if choice == "Login":
            login_page()  # Hiển thị trang đăng nhập
        elif choice == "Sign Up":
            register_page()  # Hiển thị trang đăng ký