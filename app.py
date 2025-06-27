import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
# IMPORTANT: Replace this with your Hugging Face repo ID
HF_REPO_ID = "Lech-Iyoko/Symptom-to-Condition_Classifier" 
LGBM_MODEL_FILENAME = "lgbm_disease_classifier.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# --- Caching Wrapper for Model Loading ---
@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        lgbm_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LGBM_MODEL_FILENAME)
        label_encoder_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LABEL_ENCODER_FILENAME)
        lgbm_model = joblib.load(lgbm_model_path)
        label_encoder = joblib.load(label_encoder_path)
        return tokenizer, bert_model, lgbm_model, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# --- Feature Engineering Functions ---
def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def generate_embedding(text, tokenizer, bert_model):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        bert_model.eval()
        model_output = bert_model(**encoded_input)
    embedding = mean_pool(model_output, encoded_input['attention_mask'])
    return embedding.cpu().numpy()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Symptom to Condition Classifier", layout="wide")
st.title("Symptom to Condition Classifier 🤖")
st.markdown("This tool uses a machine learning model to predict a possible medical condition based on symptoms. **This is a portfolio project and is not a substitute for professional medical advice.** Always consult a healthcare provider for any medical concerns.")

with st.container():
    st.write("---")
    tokenizer, bert_model, lgbm_model, label_encoder = load_models()

if tokenizer is not None:
    st.subheader("Enter Your Symptoms")
    user_input = st.text_area("Please describe your symptoms in detail:", height=150, placeholder="e.g., I have a sharp pain in my chest, difficulty breathing, and a persistent cough.")

    if st.button("Classify Condition"):
        if user_input:
            with st.spinner("Analyzing symptoms..."):
                embedding = generate_embedding(user_input, tokenizer, bert_model)
                prediction_id = lgbm_model.predict(embedding)
                predicted_condition = label_encoder.inverse_transform(prediction_id)[0]
                probabilities = lgbm_model.predict_proba(embedding)[0]
                top_3_indices = probabilities.argsort()[-3:][::-1]
                top_3_conditions = label_encoder.inverse_transform(top_3_indices)
                top_3_probs = probabilities[top_3_indices]

            st.subheader("Analysis Results")
            st.success(f"**Most Likely Condition:** {predicted_condition.title()}")
            st.write("**Top 3 Possibilities:**")
            for condition, prob in zip(top_3_conditions, top_3_probs):
                st.write(f"- **{condition.title()}:** `{prob:.1%}` confidence")
        else:
            st.error("Please enter symptoms in the text box.")
else:
    st.error("Could not load models. Please check the repository ID and ensure all files are present on the Hugging Face Hub.")