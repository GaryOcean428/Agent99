"""
Agent99 - AI Assistant Streamlit Application

This module implements a Streamlit-based chat interface for the Agent99 AI assistant.
It uses various components to analyze user input, generate responses, and manage
conversation memory.
"""

import streamlit as st
from input_analyzer import InputAnalyzer
from response_generator import ResponseGenerator
from memory_manager import MemoryManager
from groq_model_manager import GroqModelManager
from config import Config

# Initialize components
config = Config()
model_manager = GroqModelManager()
memory_manager = MemoryManager()
input_analyzer = InputAnalyzer(config)
response_generator = ResponseGenerator(config, memory_manager)

st.set_page_config(page_title="Agent99", page_icon="🕵️")

st.title("Agent99 - AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    response = response_generator.generate(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
