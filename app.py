import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import pandas as pd
import torch
import faiss
import numpy as np

# Load generator model and tokenizer for BART
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

# Load embedding model and tokenizer for FAISS indexing
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load the medical dataset
@st.cache_data
def load_dataset():
    data = pd.read_csv("medical_dataset.csv")  # Replace with your dataset path
    data["text"] = (
        "Disease Name: " + data["Disease Name"].fillna("") +
        "\nGene(s) Involved: " + data["Gene(s) Involved"].fillna("") +
        "\nInheritance Pattern: " + data["Inheritance Pattern"].fillna("") +
        "\nSymptoms: " + data["Symptoms"].fillna("") +
        "\nSeverity Level: " + data["Severity Level"].fillna("") +
        "\nRisk Assessment: " + data["Risk Assessment"].fillna("") +
        "\nTreatment Options: " + data["Treatment Options"].fillna("") +
        "\nSuggested Medical Tests: " + data["Suggested Medical Tests"].fillna("") +
        "\nMinimum Values for Medical Tests: " + data["Minimum Values for Medical Tests"].fillna("") +
        "\nEmergency Treatment: " + data["Emergency Treatment"].fillna("")
    )
    return data

data = load_dataset()
texts = data["text"].tolist()

# Generate embeddings for all texts
@st.cache_data
def embed_texts(texts):
    embeddings = []
    for text in texts:
        inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings.append(embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

embeddings = embed_texts(texts)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to create an embedding for a query
def embed_query(query):
    inputs = embed_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to retrieve similar entries and generate a response
def retrieve_and_generate(query, top_k=3):
    query_embedding = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [texts[i] for i in indices[0]]
    context = "\n\n".join(retrieved_texts) + "\n\nQuestion: " + query

    inputs = generator_tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(**inputs, max_length=150, num_return_sequences=1)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI for input and output
st.title("Medical Query Assistant")
st.write("Enter patient information and medical details to receive insights.")

# Collect user input
patient_name = st.text_input("Patient Name:")
patient_sex = st.selectbox("Patient Sex:", options=["Male", "Female", "Other"])
patient_age = st.number_input("Patient Age:", min_value=0, max_value=120, step=1)

disease = st.text_input("Enter known disease (if applicable):")
symptoms = st.text_area("Describe symptoms (separate multiple symptoms with commas):")
medical_test = st.text_area("Enter details of any relevant medical tests:")

if st.button("Get Medical Insight"):
    # Construct the query
    query = f"Patient Name: {patient_name}\nSex: {patient_sex}\nAge: {patient_age} years\n"
    if disease:
        query += f"Disease Name: {disease}\n"
    if symptoms:
        query += f"Symptoms: {symptoms}\n"
    if medical_test:
        query += f"Medical Test: {medical_test}\n"

    # Generate and display the response
    response = retrieve_and_generate(query)
    st.write("### Generated Medical Insight:")
    st.write(response)
