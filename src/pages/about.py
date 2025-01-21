import streamlit as st
from streamlit_extras.colored_header import colored_header

st.set_page_config(
    page_title="About - LLM Comparison Tool",
    page_icon="ℹ️",
    layout="wide"
)

colored_header(
    label="About the LLM Comparison Tool",
    description="Learn more about this application",
    color_name="blue-70"
)

st.markdown("""
## Features

This tool allows you to compare responses from three different Language Models:

1. **Google Gemini Pro** - Google's latest experimental LLM
2. **OpenAI GPT-3.5** - OpenAI's powerful language model
3. **Mistral AI** - An emerging open-source alternative

### Key Features:
- Real-time comparison of responses
- Adjustable temperature setting
- Persistent chat history during session
- Modern, clean interface
- Error handling for API failures

### How to Use
1. Enter your API keys in the `.env` file
2. Adjust the temperature using the slider in the sidebar
3. Type your message and click Send
4. Compare the responses from each model

### Technical Details
- Built with Streamlit
- Uses official APIs for each LLM
- Implements modern UI/UX practices
- Maintains conversation context
""")

st.sidebar.title("About")
st.sidebar.info("This is a demonstration of different LLM capabilities.") 