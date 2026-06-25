
import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# Download NLTK Stopwords
# =====================================================

nltk.download('stopwords')

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("First 5 Rows:")
print(df.head())

print("\nShape:", df.shape)

# =====================================================
# Convert Labels
# ham = 0
# spam = 1
# =====================================================

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nConverted Labels:")
print(df.head())

# =====================================================
# Spam vs Ham Count
# =====================================================

print("\nSpam vs Ham Count:")
print(df['label'].value_counts())

import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar')

plt.title("Spam vs Ham Messages")
plt.xlabel("Label (0=Ham, 1=Spam)")
plt.ylabel("Count")
plt.show()

# =====================================================
# NLP Text Preprocessing
# =====================================================

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenization
    words = text.split()

    # Stopword Removal + Stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =====================================================
# Apply Preprocessing
# =====================================================

df['message'] = df['message'].apply(preprocess)

print("\nProcessed Messages:")
print(df.head())

# =====================================================
# TF-IDF Feature Extraction
# =====================================================

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['message']).toarray()

y = df['label']

print("\nFeature Matrix Shape:")
print(X.shape)

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# =====================================================
# Train Model
# =====================================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# =====================================================
# Predictions
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# Accuracy
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# Test Custom Email
# =====================================================

email = """
Congratulations!

You have won a FREE iPhone.

Click here now to claim your reward.
"""

email = preprocess(email)

email_vector = tfidf.transform([email])

prediction = model.predict(email_vector)

print("\nCustom Email Test:")

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")

# =====================================================
# Save Model
# =====================================================

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully")

print("Files Created:")
print("spam_model.pkl")
print("vectorizer.pkl")

