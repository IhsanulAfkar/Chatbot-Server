import pandas as pd
from nltk_utils import Helper
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
tokenizer = pickle.load(open("assets/tokenizer.pickle", 'rb'))
helper = Helper()
slang_df = pd.read_csv('./assets/slangs.csv')
chat = ", topup diamond 344 min?"
chat = helper.slang_cleaning(chat, slang_df)
nlp_tokenize = helper.tokenize(chat)
print(nlp_tokenize)
nlp_tokenize = helper.remove_punctuations(nlp_tokenize)
print(nlp_tokenize)
keras_tokenize = tokenizer.texts_to_sequences([nlp_tokenize])
print(keras_tokenize)
padded_chat = pad_sequences(keras_tokenize, maxlen=20)
print(padded_chat)