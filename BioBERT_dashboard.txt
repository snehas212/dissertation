import streamlit as st
import joblib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

# Load the pre-trained model and tokenizer for BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model_ner = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer)

# Load the sentiment analysis model and vectorizer
sentiment_model = joblib.load('random_forest_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Drug Review Sentiment Analysis Dashboard')

# Text area for user input
review_input = st.text_area("Enter your drug review here:", 'Type your review...')

def extract_drug_names(text):
    # Processing text and extracting entities using a NER pipeline with BioBERT
    ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    results = ner_pipeline(text)
    # Check all entity labels and words detected
    all_entities = [(ent['word'], ent['entity']) for ent in results]
    print("All entities detected:", all_entities)
    # Filtering results to get drug names
    drug_names = [ent['word'] for ent in results if 'DRUG' in ent['entity']]
    return drug_names

if st.button('Analyze Review'):
    # Use BioBERT to find drug names
    # results = ner_pipeline(review_input)
    # drug_names = [ent['word'] for ent in results if ent['entity'] == 'B-DRUG' or ent['entity'] == 'I-DRUG']
    drug_names = extract_drug_names(review_input)
    if drug_names:
        st.write("Identified Drug Name(s):", ', '.join(set(drug_names)))  # Using set to remove duplicates
    else:
        st.write("No drug names identified.")
    
    # Transform the user input using the same TF-IDF vectorizer used during training
    transformed_input = tfidf_vectorizer.transform([review_input])
    
    # Predict sentiment and probabilities
    sentiment_prediction = sentiment_model.predict(transformed_input)
    sentiment_probability = sentiment_model.predict_proba(transformed_input)
    
    # Display sentiment
    sentiment_label = 'Positive' if sentiment_prediction[0] == 1 else 'Negative'
    st.write("Sentiment Prediction:", sentiment_label)
    
    # Plot the sentiment probabilities
    fig, ax = plt.subplots()
    classes = ['Negative', 'Positive']
    ax.bar(classes, sentiment_probability[0], color=['red', 'green'])
    ax.set_ylabel('Probability')
    ax.set_title('Sentiment Probability')
    st.pyplot(fig)
