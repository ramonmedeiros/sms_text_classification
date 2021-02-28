import tensorflow as tf
import pandas as pd
from tensorflow import keras

import tensorflow_datasets as tfds
import tensorflow_text as text

import numpy as np
import matplotlib.pyplot as plt

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000

# read data and set titles
train_dataset = pd.read_csv(train_file_path, sep='\t', names=['label', 'text'])
test_dataset = pd.read_csv(test_file_path, sep='\t', names=['label', 'text'])

# map ham spam
train_dataset['label'] = train_dataset['label'].map({"ham": 1.0, 'spam': 0.0})
test_dataset['label'] = test_dataset['label'].map({"ham": 1.0, 'spam': 0.0})

import ipdb;ipdb.set_trace()
train_dataset['label'] = np.asarray(train_dataset['label']).astype(np.float32)
test_dataset['label'] = np.asarray(test_dataset['label']).astype(np.float32)

# tokenize text and generate dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

for text, label in train_dataset.take(1):
  print('texts: ', text.numpy())
  print('label: ', label.numpy())

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

import ipdb;ipdb.set_trace()
train_dataset.element_spec

for text in train_dataset.take(1):
  print('texts: ', text.numpy()[:3])


encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]


# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):



  return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)


# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()

