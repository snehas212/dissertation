import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib 

# Set up logging
logging.basicConfig(filename='final_model_comparison.log', level=logging.INFO)

# Load datasets
train = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTrain_raw.tsv', delimiter='\t')
test = pd.read_csv('/home/sneha/Sentiment Analysis/Datasets/drugs_review/drugsComTest_raw.tsv', delimiter='\t')

print("Train dataset shape: ", train.shape)
print("Test dataset shape: ", test.shape)
data = pd.concat([train, test], ignore_index=True)
data.info()
print(data.describe())

# Save a plot of the rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(data['rating'])
plt.title('Rating Distribution')
plt.savefig('rating_distribution.png')
plt.show()

# Checking missing values
missing_values = data.isnull().sum()
print(missing_values)

# Drug names word cloud
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=400).generate(' '.join(data['drugName']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Drug Names')
plt.savefig('drug_names_wordcloud.png')
plt.show()

# Sentiment based on ratings
data['sentiment'] = np.where(data['rating'] > 5, 'Positive', 'Negative')
plt.figure(figsize=(10, 6))
sns.countplot(data['sentiment'])
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.show()

# Convert text to vectors
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
features = tfidf.fit_transform(data['review'].fillna(' ')).toarray()
labels = data['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model training and evaluation
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'ANN': MLPClassifier(max_iter=300),
    'LGBM': LGBMClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    logging.info(f'{name} Model: \nAccuracy: {accuracy}\nReport: {report}\n')
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} Classification Report:\n{report}')

        # Save the Random Forest model
    if name == 'Random Forest':
        joblib.dump(model, 'random_forest_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')  # Also save the vectorizer
        print("Random Forest model and TF-IDF vectorizer models are saved.")

