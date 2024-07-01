import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("movie_reviews.csv")

review = df['review'].tolist()
sentiment = df['sentiment'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(review, sentiment, test_size=0.2, random_state=42)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(review, return_tensors='tf', padding=True, truncation=True, max_length=128)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    inputs['input_ids'], sentiment, test_size=0.2, random_state=42)

# Compilar y entrenar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_inputs,
    np.array(train_labels),
    validation_data=(val_inputs, np.array(val_labels)),
    epochs=3,
    batch_size=32
)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(val_inputs, np.array(val_labels), verbose=2)
print(f'Test accuracy: {test_acc}')