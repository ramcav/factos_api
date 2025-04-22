import streamlit as st
import requests
import os
import json

# Configure the API URL
API_URL = os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(
    page_title="Fake News Detection Demo",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for models and selected model
if 'models' not in st.session_state:
    try:
        response = requests.get(f"{API_URL}/api/models")
        st.session_state.models = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.session_state.models = []

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

st.title("🔍 Fake News Detection Demo")
st.markdown("""
This application demonstrates different machine learning models for fake news detection.
Enter your text and select a model to test if the content might be fake news!
""")

def get_prediction(model_id, text):
    """Get prediction from the API for the given text and model"""
    if model_id is None:
        st.info("No model selected. The default model (Naive Bayes) will be used.")
        model_id = "3"  # Assuming 1 is the ID for Naive Bayes model
    try:
        response = requests.post(
            f"{API_URL}/api/models/{model_id}/predict",
            json={"text": text}
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting prediction: {str(e)}")
        return None

# Sidebar with model selection
st.sidebar.title("Model Selection")
st.sidebar.markdown("Choose a model for fake news detection:")

# Use models from session state
if st.session_state.models:
    model_names = {str(model["id"]): model["name"] for model in st.session_state.models}

    # Update selected model in session state when selection changes
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_names.keys()),
        format_func=lambda x: model_names[x],
        key='selected_model'  # This automatically updates st.session_state.selected_model
    )

    # Display model information
    selected_model_info = next(
        (model for model in st.session_state.models if str(model["id"]) == selected_model),
        None
    )

    if selected_model_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Details")
        st.sidebar.markdown(f"**Description:** {selected_model_info['description']}")
        st.sidebar.markdown(f"**Accuracy:** {selected_model_info['accuracy']:.2%}")
        st.sidebar.markdown(f"**Type:** {selected_model_info['type'].title()}")

    # Main content area
    text_input = st.text_area(
        "Enter text to analyze",
        height=200,
        placeholder="Paste your news article or text here..."
    )

    if st.button("Analyze Text"):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                result = get_prediction(st.session_state.selected_model, text_input)
                print("RESULT", result)
                if result and "error" not in result:
                    # Create columns for displaying results
                    col1, col2, col3 = st.columns(3)

                    # Display prediction with appropriate color
                    prediction_color = "green" if result["prediction"] == 1 else "red"
                    col1.metric(
                        "Prediction",
                        result["prediction_label"],
                        delta=None
                    )
                    st.markdown(
                        f"<h2 style='text-align: center; color: {prediction_color};'>{result['prediction_label']}</h2>",
                        unsafe_allow_html=True
                    )

                    # Display confidence score
                    confidence_percentage = f"{result['confidence']:.2%}"
                    col2.metric(
                        "Confidence Score",
                        confidence_percentage,
                        delta=None
                    )

                    # Display model used
                    col3.metric(
                        "Model Used",
                        result["model_name"],
                        delta=None
                    )

                    st.markdown("---")
                    st.markdown("### Analysis Details")
                    st.json(result)
                else:
                    st.error("Error getting prediction. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.error("Could not fetch models from the API. Please make sure the API server is running.")
