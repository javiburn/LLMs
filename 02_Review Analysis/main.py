from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer()
df = pd.read_csv("movie_reviews.csv")
tokenizer = tf_text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(df['review'])

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Ejemplos de predicciones
ejemplos = ["This film is great!", "I did not like this movie at all."]
ejemplos_features = vectorizer.transform(ejemplos)
predicciones = model.predict(ejemplos_features)
print(predicciones)