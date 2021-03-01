import tensorflow as tf
import pandas as pd
from tensorflow import keras

import numpy as np

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000

# read data and set titles
train_dataset = pd.read_csv(train_file_path, sep='\t', names=['label', 'text'])
test_dataset = pd.read_csv(test_file_path, sep='\t', names=['label', 'text'])

# map ham spam and covert values to float
train_dataset['label'] = train_dataset['label'].map({"ham": 1.0, 'spam': 0.0})
test_dataset['label'] = test_dataset['label'].map({"ham": 1.0, 'spam': 0.0})
train_dataset['label'] = np.asarray(train_dataset['label']).astype(np.float32)
test_dataset['label'] = np.asarray(test_dataset['label']).astype(np.float32)

# create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset['text'], train_dataset['label']))
test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset['text'], test_dataset['label']))

# examine data
#for text, label in train_dataset.take(1):
#  print('texts: ', text.numpy())
#  print('label: ', label.numpy())

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# examine data
#for text, label in train_dataset.take(1):
#  print('texts: ', text.numpy()[:3])
#  print('label: ', label.numpy()[:3])

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)

encoder.adapt(train_dataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())
#print(vocab[:20])

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

assert [layer.supports_masking for layer in model.layers] == [False, True, True, True, True]

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

# save model
model.save('model')

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    predictions = model.predict(np.array([pred_text]))[0][0]

    if round(predictions) >= 1:
        return [predictions, "ham"]
    return [predictions, "spam"]


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

