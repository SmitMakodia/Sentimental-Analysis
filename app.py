import streamlit as st
import joblib
import torch
import os
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import streamlit.components.v1 as components

st.set_page_config(page_title="OmniSent | AI Sentiment Engine", layout="wide")

MODEL_PATH_TRANSFORMER = os.path.join("models", "transformer")
MODEL_PATH_BASELINE = os.path.join("models", "baseline", "baseline_model.pkl")

@st.cache_resource
def load_transformer():
    print("Loading Transformer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_TRANSFORMER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_TRANSFORMER)
        explainer = SequenceClassificationExplainer(model, tokenizer)
        return tokenizer, model, explainer
    except Exception as e:
        st.error(f"Error loading Transformer: {e}")
        return None, None, None

@st.cache_resource
def load_baseline():
    print("Loading Baseline...")
    try:
        return joblib.load(MODEL_PATH_BASELINE)
    except Exception as e:
        st.error(f"Error loading Baseline: {e}")
        return None

tokenizer, model, explainer = load_transformer()
baseline_model = load_baseline()

def predict_transformer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

def predict_baseline(text):
    probs = baseline_model.predict_proba([text])[0]
    return probs

st.title("OmniSent: Domain-Agnostic Sentiment Engine")
st.markdown("""
This system uses a fine-tuned **DistilRoBERTa Transformer** to detect sentiment in complex text. 
It outperforms standard Logistic Regression by understanding context (e.g., sarcasm, slang).
""")

st.sidebar.header("Configuration")
model_choice = st.sidebar.radio("Select Model:", ["Transformer (Recommended)", "Baseline (Logistic Regression)"])

text_input = st.text_area("Enter text to analyze:", height=150, placeholder="Type a review or tweet here...")

if st.button("Analyze Sentiment", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        if model_choice == "Transformer (Recommended)":
            probs = predict_transformer(text_input)
            neg_score, pos_score = probs[0], probs[1]
            word_attributions = explainer(text_input)
        else:
            probs = predict_baseline(text_input)
            neg_score, pos_score = probs[0], probs[1]
            word_attributions = None

        label = "POSITIVE" if pos_score > neg_score else "NEGATIVE"
        confidence = pos_score if label == "POSITIVE" else neg_score
        color = "green" if label == "POSITIVE" else "red"

        with col1:
            st.subheader("Result")
            st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': "Confidence (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "white"}],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Interpretation")
            
            if model_choice == "Transformer (Recommended)":
                st.info("Highlights: **Green** words pushed the model to Positive, **Red** words pushed it to Negative.")
                viz_obj = explainer.visualize()
                
                if hasattr(viz_obj, 'data'):
                    raw_html = viz_obj.data
                else:
                    raw_html = str(viz_obj)

                styled_html = f"""
                <html>
                    <head>
                        <style>
                            body {{
                                background-color: #ffffff !important;
                                color: #000000 !important;
                                font-family: sans-serif;
                            }}
                            .viz-container {{
                                padding: 20px;
                                border-radius: 10px;
                                border: 1px solid #e0e0e0;
                                background-color: #ffffff;
                                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                            }}
                            span {{
                                border-radius: 6px !important;
                                padding: 3px 6px !important;
                                margin: 0 2px !important;
                                display: inline-block !important;
                                font-weight: 500 !important;
                                border: 1px solid rgba(0,0,0,0.1);
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="viz-container">
                            {raw_html}
                        </div>
                    </body>
                </html>
                """
                
                components.html(styled_html, height=400, scrolling=True)

            else:
                st.warning("Feature Importance visualization is not available for the Baseline model in this view.")
                st.write(f"**Raw Probabilities:**")
                st.write(f"Negative: {neg_score:.4f}")
                st.write(f"Positive: {pos_score:.4f}")

st.markdown("---")
st.caption(f"Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")




# always use venv 👍🏻