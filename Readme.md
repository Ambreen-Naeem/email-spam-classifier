# Email Spam Classifier using NLP and Machine Learning

## Project Overview

This project classifies emails as Spam or Not Spam using Natural Language Processing (NLP) and Machine Learning techniques.

The system preprocesses email text, extracts features using TF-IDF Vectorization, and uses a Multinomial Naive Bayes classifier to detect spam emails accurately.

---

## Features

* Email text preprocessing
* Stopword removal
* Stemming using Porter Stemmer
* TF-IDF feature extraction
* Spam/Not Spam prediction
* Web interface using Flask
* User-friendly design

---

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NLTK
* TF-IDF Vectorizer
* Multinomial Naive Bayes

---

## Dataset

SMS Spam Collection Dataset from Kaggle

Total Records: 5572

* Ham Messages: 4825
* Spam Messages: 747

---

## Model Performance

Accuracy: 96.86%

Confusion Matrix:

[[965 0]
[35 115]]

---

## Project Structure

EmailSpamClassifier/

├── app.py

├── main.py

├── spam.csv

├── spam_model.pkl

├── vectorizer.pkl

├── requirements.txt

├── templates/

│ └── index.html

├── static/

│ └── style.css

└── screenshots/

---

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run Flask App

python app.py

3. Open browser

http://127.0.0.1:5000

---

## Future Improvements

* Deep Learning (LSTM/BERT)
* Confidence Score
* Spam Probability Visualization
* Email Phishing Detection
* Cloud Deployment

---
## Screenshots

### Home Page
![Home Page](screenshots/Home Page.png)

### Spam Detection
![Spam Detection](screenshots/Spam Email.png)

### Not Spam Detection
![Not Spam](screenshots/Not Spam.png)

---
## Author

Ambreen Naeem

Computer Science Student
