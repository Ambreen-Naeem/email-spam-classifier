from flask import Flask, render_template, request
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    email = request.form["email"]

    processed_email = preprocess(email)

    vector = tfidf.transform([processed_email])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "🚨 Spam Email"
    else:
        result = "✅ Not Spam"

    return render_template(
        "index.html",
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True)

