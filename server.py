from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk_utils import Helper
import pandas as pd
import time
app = Flask(__name__)
helper = Helper()
model = tf.keras.models.load_model('chatbot_model.h5')
slang_df = pd.read_csv('./assets/slangs.csv')
tokenizer = pickle.load(open("assets/tokenizer.pickle", 'rb'))
labels = pickle.load(open("assets/labels.pickle", 'rb'))

def generate_response(text):
    text = helper.slang_cleaning(text, slang_df)
    text = helper.stopword_removal(text)
    tokenize_text = helper.tokenize(text)
    tokenize_text = helper.remove_punctuations(tokenize_text)
    sequences = tokenizer.texts_to_sequences([tokenize_text])
    vec = pad_sequences(sequences, maxlen=20)
    pred = model.predict(vec)
    y_pred = np.argmax(pred, axis=-1)
    conf = np.max(pred)
    with open('respose_flow.txt', 'a') as file:
        file.write(f'\n{text}, {tokenize_text}, {sequences}, {vec}, {y_pred}, ({labels[y_pred[0]]},{conf})')
    return labels[y_pred[0]], conf


@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('text', '')
    start_time = time.time()
    response, confidence = generate_response(input_text)
    with open('chat_time.txt', 'a') as file:
        file.write(f'\n{input_text},{response},{confidence},{start_time},{time.time()}')
    return jsonify({'intent': response,"confidence": float(confidence)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
