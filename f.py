import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

m = st.markdown("""
<style>
 .stApp {
        background-color: black;
        color: white;
    }
    h1 {
        color: white;
    }
     .stTextArea label {
        color: white;
    }
    div.stButton > button:first-child {
        background-color: rgb(0, 204, 255);
        color: white;
    }
    textarea, .stTextInput input {
        background-color: #333;
        color: white;
    }
</style>""", unsafe_allow_html=True)



st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    
    transformed_sms = transform_text(input_sms)
    
    vector_input = tfidf.transform([transformed_sms])
  
    result = model.predict(vector_input)[0]
   
    if result == 1:
        st.markdown("<h2 style='color: red;'>Spam</h2>", unsafe_allow_html=True)
    else:
         st.markdown("<h2 style='color: green;'>Not Spam</h2>", unsafe_allow_html=True)