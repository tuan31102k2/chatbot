# utils/chatbot.py
import requests
from config import GEMINI_API_URL, GEMINI_API_KEY

"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import google.generativeai as genai

def query_gemini_api(prompt):
    genai.configure(api_key=GEMINI_API_KEY)

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(prompt)

    return response.text