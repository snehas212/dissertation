import streamlit as st
import spacy
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained model and SpaCy model
nlp = spacy.load('en_core_web_sm')
model = joblib.load('random_forest_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') 
st.title('Drug Review Sentiment Analysis Dashboard')

# Text area for user input
review_input = st.text_area("Enter your drug review here:", 'Type your review...')

if st.button('Analyze Review'):
    # Use SpaCy to find drug names
    # doc = nlp(review_input)
    # drug_names = [ent.text for ent in doc.ents if ent.label_ == 'CHEM']
    # if drug_names:
    #     st.write("Identified Drug Name(s):", ', '.join(drug_names))
    # else:
    #     st.write("No drug names identified.")
    
    # Transform the user input using the same TF-IDF vectorizer used during training
    transformed_input = tfidf_vectorizer.transform([review_input])
    
    # Predict sentiment and probabilities
    sentiment_prediction = model.predict(transformed_input)
    sentiment_probability = model.predict_proba(transformed_input)
    
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

