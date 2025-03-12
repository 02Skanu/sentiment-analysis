import streamlit as st
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    model_path = "/Users/seokwanwoo/Desktop/ML_DL/sentiment-bert-model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("감성 분류 모델 with SHAP")
st.write("문장을 입력하면, 해당 문장의 감성을 예측하고 설명을 제공합니다.")

input_text = st.text_input("문장을 입력하세요:")

def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits.softmax(dim=-1)
    return probabilities.cpu().numpy()

def model_forward(input_texts):
    if isinstance(input_texts, list):
        input_texts = [text if isinstance(text, str) else "" for text in input_texts]
    else:
        input_texts = [input_texts]
    encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded_inputs = {key: tensor.to(device) for key, tensor in encoded_inputs.items()}
    with torch.no_grad():
        output = model(**encoded_inputs)
        return output.logits.cpu().numpy()

if st.button("감성 분석 및 해석"):  
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = outputs.logits.softmax(dim=-1).cpu().numpy()
    pred_label = np.argmax(probabilities)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    st.write("예측된 감성:", label_map[pred_label])
    explainer = shap.Explainer(model_forward, tokenizer)
    shap_values = explainer([input_text])
    fig, ax = plt.subplots()
    shap.plots.text(shap_values, ax=ax)
    st.pyplot(fig)
