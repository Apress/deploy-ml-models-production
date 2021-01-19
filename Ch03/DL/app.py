# load packages
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load tokenizer for preprocessing
with open('tokenizer.pickle', 'rb') as tk:
    tokenizer = pickle.load(tk)

# Load pre-trained model into memory
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)

# Load weights into new model
lstm_model.load_weights("model.h5")

def sentiment_prediction(review):
    sentiment=[]
    # Convert to array
    input_review = [review]
    input_review = [x.lower() for x in input_review]
    input_review = [re.sub('[^a-zA-z0-9\s]','',x) for x in input_review]
 

    # Convert into list with word ids
    input_feature = tokenizer.texts_to_sequences(input_review)
    input_feature = pad_sequences(input_feature,1473, padding='pre')
    sentiment = lstm_model.predict(input_feature)[0]
    
    if(np.argmax(sentiment) == 0):
        pred="Negative"
    else:
        pred= "Positive"
    
    return pred


def run():
    streamlit.title("Sentiment Analysis - LSTM Model")
    html_temp="""
    
    """
 
    streamlit.markdown(html_temp)
    review=streamlit.text_input("Enter the Review ")
    prediction=""
    
    if streamlit.button("Predict Sentiment"):
        prediction=sentiment_prediction(review)
    streamlit.success("The sentiment predicted by Model : {}".format(prediction))
    
if __name__=='__main__':
    run()


    
