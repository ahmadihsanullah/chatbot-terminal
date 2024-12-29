# import library
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

def preprocess(chat):
    # konversi ke non kapital
    chat = chat.lower()
    # hilangkan tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

class JSONParser:
    def __init__(self):
        self.data = None
        self.df = None

    def parse(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def get_dataframe(self):
        text_input = []
        intents = []
        
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                text_input.append(pattern)
                intents.append(intent['tag'])
        
        self.df = pd.DataFrame({
            'text_input': text_input,
            'intents': intents
        })
        return self.df

    def get_response(self, tag):
        for intent in self.data['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
        return "Maaf, saya tidak mengerti."

def bot_response(chat, pipeline, jp):
    chat = preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    # Lower threshold and more flexible matching
    if max_prob < 0.05:
        return "Maaf kak, aku ga ngerti :(" , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag), pred_tag

# load data
path = "data/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# praproses data
df['text_input_prep'] = df.text_input.apply(preprocess)

# pemodelan
pipeline = make_pipeline(
    TfidfVectorizer(stop_words=None),  # Use TF-IDF instead of Count Vectorizer
    MultinomialNB(alpha=0.1)  # Add slight smoothing
)

# train
print("[INFO] Training Data ...")
pipeline.fit(df.text_input_prep, df.intents)

# save model
with open("model_chatbot.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

# interaction with bot
print("[INFO] Anda Sudah Terhubung dengan Bot Kami")
while True:
    chat = input("Anda >> ")
    res, tag = bot_response(chat, pipeline, jp)
    print(f"Bot >> {res}")
    if tag == 'bye':
        break