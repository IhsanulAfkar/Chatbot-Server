from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk_utils import Helper
import pandas as pd
app = Flask(__name__)
helper = Helper()
model = tf.keras.models.load_model('chatbot_model.h5')
# model = tf.keras.models.load_model('chatbot_model_py')
slang_df = pd.read_csv('./assets/slangs.csv')
tokenizer = pickle.load(open("assets/tokenizer.pickle", 'rb'))
tags = pickle.load(open("assets/tags.pickle", 'rb'))

def generate_response(text):
    text = helper.remove_punctuations(text)
    text = helper.slang_cleaning(text, slang_df)
    text = helper.stopword_removal(text)
    # tokenize
    # tokenized_text = helper.tokenize(text)
    sequences = tokenizer.texts_to_sequences([text])
    vec = pad_sequences(sequences, maxlen=20)
    pred = model.predict(vec)
    y_pred = np.argmax(pred, axis=-1)
    conf = np.max(pred)
    # return y_pred
    return tags[y_pred[0]], conf


@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('text', '')
    print(input_text)
    response, confidence = generate_response(input_text)

    return jsonify({'intent': response,"confidence": float(confidence)})

if __name__ == '__main__':
    # os.getenv("HOST","0.0.0.0")
    app.run(debug=True, port=5000, host="0.0.0.0")
