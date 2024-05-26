from nltk_utils import Helper
import random
from gensim import models
import json
import pickle
import numpy as np
import tensorflow as tf
from string import punctuation
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

df = pd.read_csv('./assets/data.csv')
# df.head()

slang_df = pd.read_csv('./assets/slangs.csv')

helper = Helper()
patterns = df.pattern.values
tags = df.tag.values
clean_patterns = []
for text in patterns:
    clean_patterns.append(helper.remove_punctuations(text))


removed_slangs = []
for text in clean_patterns:
    removed_slangs.append(helper.slang_cleaning(text, slang_df))

removed_stopwords = []
for text in removed_slangs:
    removed_stopwords.append(helper.stopword_removal(text))

tokenized_text = []
for text in removed_stopwords:
    tokenized_text.append(helper.tokenize(text))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(tokenized_text)
train = tokenizer.texts_to_sequences(tokenized_text)
word_index = tokenizer.word_index
f = open('assets/tokenizer.pickle', 'wb')
pickle.dump(tokenizer, f)
f.close()
# hyper params
max_len=20
output_dim=300
train = pad_sequences(train, maxlen=20)

from sklearn.preprocessing import LabelEncoder
cls = []
kategori_list = pd.unique(df.tag.values)
kategori_list = kategori_list.tolist()
for i in range(len(tags)):
    one_hot = np.zeros((len(kategori_list),), dtype=int)
    idx = kategori_list.index(tags[i])
    one_hot[idx] = 1
    cls.append(one_hot)
cls = np.array(cls)
encoder = LabelEncoder()
# temp_tag = pd.unique(tags).tolist()
labels = encoder.fit_transform(kategori_list)

tags = dict()
for i in range(len(kategori_list)) :
    tags[i] = kategori_list[i]
print(tags)

f = open('assets/tags.pickle', 'wb')
pickle.dump(tags, f)
f.close()

word2vec_path = './idwiki_word2vec_300_new.txt'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path)
train_embedding_weights = np.zeros((len(word_index)+1, output_dim))
for word,index in word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(output_dim)
print(train_embedding_weights.shape)
from sklearn.model_selection import train_test_split
num_words = len(word_index)+1

X_train, X_test, y_train, y_test = train_test_split(train, cls, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 
                              output_dim,
                              weights=[train_embedding_weights],
                              input_length=max_len),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])
# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
from keras.callbacks import ModelCheckpoint, EarlyStopping
batch_size = 32
num_epochs = 200

model.fit(
    X_train,
    y_train, 
    epochs=num_epochs,  
    shuffle=True,
    validation_data=(X_test, y_test),
    batch_size=batch_size
    )
model.save("chatbot_model.h5")
test_loss, test_acc = model.evaluate(X_test, y_test)

from sklearn.metrics import f1_score,confusion_matrix, recall_score, precision_score
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Convert y_test from one-hot encoding to class labels if necessary
y_test_classes = y_test.argmax(axis=1)
# matrix = confusion_matrix(y_test_classes, y_pred_classes)
# print("Matrix ", matrix)
# Calculate F1 score
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_classes,average='weighted',zero_division=0)
precision = precision_score(y_test_classes, y_pred_classes,average='weighted')
print("Accuracy:", test_acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)