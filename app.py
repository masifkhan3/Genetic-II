# Import required libraries
import streamlit as st
import pandas as pd
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# Load your dataset
@st.cache_data
def load_data():
    data = pd.read_csv("/content/genetic-Final.csv")
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
    return data["text"].tolist()

texts = load_data()

# Load models and tokenizer
@st.cache_resource
def load_models():
    embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
    return embedding_tokenizer, embedding_model, generator_tokenizer, generator_model

embedding_tokenizer, embedding_model, generator_tokenizer, generator_model = load_models()

# Initialize FAISS index
@st.cache_resource
def initialize_faiss_index(texts):
    # Function to generate embeddings
    def embed_text(text):
        inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    # Generate embeddings for all texts
    embeddings = [embed_text(text) for text in texts]

    # Initialize FAISS index
    embedding_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))
    return index

index = initialize_faiss_index(texts)

# Function to retrieve similar entries and generate a response
def retrieve_and_generate(query, top_k=3):
    # Generate query embedding
    query_embedding = embed_text(query).reshape(1, -1)

    # Retrieve top-k similar texts
    distances, indices = index.search(query_embedding, top_k)
    retrieved_texts = [texts[i] for i in indices[0]]

    # Prepare input for generation by combining retrieved texts
    context = "\n\n".join(retrieved_texts) + "\n\nQuestion: " + query

    # Generate answer
    inputs = generator_tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(**inputs, max_length=150, num_return_sequences=1)

    # Decode the output
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit Interface
st.title("Medical Query Assistant")
st.write("Enter details about diseases, symptoms, or medical tests for information retrieval.")

# Input fields
disease = st.text_input("Enter the disease name:")
symptoms = st.text_area("Enter symptoms:")
medical_test = st.text_input("Enter medical test details:")

# Process input query
query = ""
if disease:
    query += f"Disease Name: {disease}\n"
if symptoms:
    query += f"Symptoms: {symptoms}\n"
if medical_test:
    query += f"Medical Test: {medical_test}\n"

# Button to submit query and generate response
if st.button("Generate Response"):
    if query.strip():
        with st.spinner("Generating response..."):
            response = retrieve_and_generate(query)
            st.subheader("Generated Response")
            st.write(response)
    else:
        st.warning("Please enter at least one detail.")
