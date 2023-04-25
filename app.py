from flask import Flask,request,jsonify
import numpy as np
import new
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from new import transform_text

nltk.download('punkt')
nltk.download('stopwords')
# model1 = pickle.load(open('model1.pkl','rb'))
tfidf = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"


@app.route('/sms-spam',methods=['POST'])
def smsspam():
    input_sms=request.form.get('input_sms')
    transformed_sms = new.transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        answer= "spam"
    else:
        answer= "not spam"
    
    return jsonify({'answer':str(answer)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
