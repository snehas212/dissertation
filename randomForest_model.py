import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 

# Load datasets
def load_data(train_filepath, test_filepath):
    train_data = pd.read_csv(train_filepath, delimiter='\t')
    test_data = pd.read_csv(test_filepath, delimiter='\t')
    return train_data, test_data

# Preprocess data using SpaCy for Named Entity Recognition to extract drug names
def preprocess_data(data):
    nlp = spacy.load('en_core_web_sm')
    # Extracting drug names and storing them
    data['drug_names'] = data['review'].apply(lambda x: [ent.text for ent in nlp(x).ents if ent.label_ == 'CHEM'])
    # Create sentiment column from ratings
    data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 6 else 0)
    return data

# Feature extraction
def feature_extraction(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(data['review'])
    return features, tfidf

# Build and train the Random Forest model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)

# Load datasets
train = '/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv'
test = '/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv'

# Main function to load data, preprocess, train, evaluate and save the model
def main():
    train_data, test_data = load_data(train,test)
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    X_train, tfidf = feature_extraction(train_data)
    X_test = tfidf.transform(test_data['review'])  # Transform test data with the same tfidf vectorizer

    y_train = train_data['sentiment']
    y_test = test_data['sentiment']

    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    save_model(model, 'random_forest_drug_sentiment.pkl')

if __name__ == '__main__':
    main()
