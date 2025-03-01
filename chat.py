import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="GKTrainings..Chatbot")
st.title("💬 Chatbot with Session Memory (Groq)")

# --- Function to Validate Groq API Key ---
def validate_groq_api(api_key):
    try:
        test_chatbot = ChatGroq(model="llama3-8b-8192", api_key=api_key)
        test_chatbot.invoke("Test")
        return True, "✅ Groq API Key is valid."
    except Exception as e:
        return False, f"❌ Invalid Groq API Key: {str(e)}"

# --- UI: API Key Input ---
if "groq_api_key" not in st.session_state:
    groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")
    if st.button("Validate API Key"):
        if groq_api_key:
            is_valid, message = validate_groq_api(groq_api_key)
            if is_valid:
                st.success(message)
                st.session_state["groq_api_key"] = groq_api_key
                st.rerun()
            else:
                st.error(message)
        else:
            st.error("❌ Please enter your Groq API Key.")
    st.stop()

# --- Initialize Chatbot ---
chatbot = ChatGroq(model="llama3-70b-8192", api_key=st.session_state["groq_api_key"])

# --- Initialize Chat Session ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "user_session_001"  # Unique session identifier

if "chatbot_memory" not in st.session_state:
    st.session_state["chatbot_memory"] = ChatMessageHistory()  # Persistent session memory

# --- Function to Get Session History ---
def get_session_history(session_id):
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

# Wrap chatbot with session-based memory
chatbot_with_memory = RunnableWithMessageHistory(
    chatbot, 
    get_session_history=lambda _: get_session_history(st.session_state["session_id"])
)

# --- Load and Display Chat History ---
for message in st.session_state["chatbot_memory"].messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# --- User Chat Input ---
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.session_state["chatbot_memory"].add_message(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("🤖 Thinking..."):
        response = chatbot_with_memory.invoke(
            [HumanMessage(content=user_input)],
            config={"configurable": {"session_id": st.session_state["session_id"]}},  # Pass session_id
        )
        bot_response = response.content

    # Store and display bot response
    st.session_state["chatbot_memory"].add_message(AIMessage(content=bot_response))
    with st.chat_message("assistant"):
        st.markdown(bot_response)