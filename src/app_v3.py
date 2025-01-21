import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from streamlit_extras.colored_header import colored_header
from streamlit_modal import Modal

# Load environment variables --local machine
#load_dotenv()

# Initialize API clients -- LOCAL MACHINE
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))


#For Git/Server deployment

# Initialize API clients
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether the password entered is correct."""
        if st.session_state["password"] == st.secrets["ACCESS_PASSWORD"]:
            st.session_state["password_correct"] = True

#FOR LOCAL MACHINE

# def check_password():
#     """Returns `True` if the user has entered the correct password."""

#     def password_entered():
#         """Checks whether the password entered is correct."""
#         if st.session_state["password"] == os.getenv("ACCESS_PASSWORD"):
#             st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("""
            <style>
            .auth-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("## 🔐 Access Required")
            st.text_input("Enter Access Password:", type="password", key="password")
            st.button("Submit", on_click=password_entered, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("## 🔐 Access Required")
            st.text_input("Enter Access Password:", type="password", key="password")
            st.button("Submit", on_click=password_entered, type="primary")
            st.error("😕 Incorrect password")
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    else:
        # Password correct
        return True

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "gemini": [],
            "openai": [],
            "mistral": []
        }
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000

def check_creator_question(question):
    """Check if the question is asking about the creator/developer"""
    creator_keywords = [
        "who created", "who made", "who developed", "who built",
        "who is the creator", "who is the developer", "who's the creator",
        "who designed", "creator of", "developer of", "built by"
    ]
    return any(keyword in question.lower() for keyword in creator_keywords)

def get_gemini_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This Comparison Tool is created by Basant Singh a Product Manager at Whizlabs."
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        messages[-1]["content"] if messages else "",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    )
    return response.text

def get_openai_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is created by Basant Singh - Product Manager, Whizlabs."
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def get_mistral_response(messages, temperature, max_tokens):
    if check_creator_question(messages[-1]["content"]):
        return "This LLM Comparison Tool is developed by Basant Singh a Product Manager at Whizlabs."
    
    response = mistral_client.chat(
        model="mistral-tiny",
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(
        page_title="LLM Comparison Tool",
        page_icon="🤖",
        layout="wide"
    )
    
    if check_password():
        initialize_session_state()
        
        # Custom CSS for modern UI
        st.markdown("""
            <style>
            .stApp {
                background-color: #f5f5f5;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e3f2fd;
            }
            .assistant-message {
                background-color: white;
            }
            .model-header {
                font-size: 1.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            /* Add styles for vertical dividers */
            .vertical-divider {
                border-left: 3px solid #FFD700;  /* Yellow color */
                height: 50px;  /* Fixed height */
                margin: 0 auto;
                margin-top: 10px;
            }
            
            .divider-column {
                display: flex;
                justify-content: center;
                padding: 0;
            }

            /* Basic input styling */
            .stTextInput input {
                border-radius: 20px !important;
                border: 1px solid #ddd !important;
                background-color: white !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                height: 45px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("# 🤖 LLM Comparison Tool - ChatBottle Royale👑")
        st.markdown("### 🔄 A tool by Basant to compare responses from different AI models in real-time!")

        # Settings in sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            
            # Temperature slider with emoji
            st.markdown("#### 🌡️ Creativity / Temperature")
            st.session_state.temperature = st.slider(
                "Control randomness",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher value results in higher level of creativity."
            )
            
            # Max tokens slider with emoji
            st.markdown("#### 📝 Max Tokens")
            st.session_state.max_tokens = st.slider(
                "Control response length",
                min_value=100,
                max_value=1000,
                value=300,
                step=100,
                help="Maximum number of tokens in the response"
            )
            
            # Add a divider
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.info("Try different temperatures and token limits to see how they affect the responses!")

        # Create columns with dividers
        col1, div1, col2, div2, col3 = st.columns([0.3, 0.01, 0.3, 0.01, 0.3])

        with col1:
            st.markdown('<div class="model-header">🧠 Google Gemini</div>', unsafe_allow_html=True)
            for message in st.session_state.messages["gemini"]:
                div_class = "user-message" if message["role"] == "user" else "assistant-message"
                emoji = "👤" if message["role"] == "user" else "🤖"
                st.markdown(f"""
                    <div class="chat-message {div_class}">
                        <b>{emoji} {message["role"].title()}:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        with div1:
            st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="model-header">🎯 OpenAI GPT-3.5</div>', unsafe_allow_html=True)
            for message in st.session_state.messages["openai"]:
                div_class = "user-message" if message["role"] == "user" else "assistant-message"
                emoji = "👤" if message["role"] == "user" else "🤖"
                st.markdown(f"""
                    <div class="chat-message {div_class}">
                        <b>{emoji} {message["role"].title()}:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        with div2:
            st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="model-header">⚡ Mistral AI</div>', unsafe_allow_html=True)
            for message in st.session_state.messages["mistral"]:
                div_class = "user-message" if message["role"] == "user" else "assistant-message"
                emoji = "👤" if message["role"] == "user" else "🤖"
                st.markdown(f"""
                    <div class="chat-message {div_class}">
                        <b>{emoji} {message["role"].title()}:</b> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        # User input section
        st.markdown("### 💬 Your Message")
        user_input = st.text_input("", key="user_input", label_visibility="collapsed")

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🚀 Send"):
                if user_input:
                    # Add user message to all conversations
                    for model in st.session_state.messages:
                        st.session_state.messages[model].append({
                            "role": "user",
                            "content": user_input
                        })

                    # Get responses from each model
                    try:
                        gemini_response = get_gemini_response(
                            st.session_state.messages["gemini"],
                            st.session_state.temperature,
                            st.session_state.max_tokens
                        )
                        st.session_state.messages["gemini"].append({
                            "role": "assistant",
                            "content": gemini_response
                        })
                    except Exception as e:
                        st.error(f"🚫 Gemini Error: {str(e)}")

                    try:
                        openai_response = get_openai_response(
                            st.session_state.messages["openai"],
                            st.session_state.temperature,
                            st.session_state.max_tokens
                        )
                        st.session_state.messages["openai"].append({
                            "role": "assistant",
                            "content": openai_response
                        })
                    except Exception as e:
                        st.error(f"🚫 OpenAI Error: {str(e)}")

                    try:
                        mistral_response = get_mistral_response(
                            st.session_state.messages["mistral"],
                            st.session_state.temperature,
                            st.session_state.max_tokens
                        )
                        st.session_state.messages["mistral"].append({
                            "role": "assistant",
                            "content": mistral_response
                        })
                    except Exception as e:
                        st.error(f"🚫 Mistral Error: {str(e)}")

                    st.rerun()

        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = {
                    "gemini": [],
                    "openai": [],
                    "mistral": []
                }
                st.rerun()

if __name__ == "__main__":
    main() 
