import streamlit as st
import joblib
import sys
import os

from src.preprocess import preprocess_text

sys.path.append(os.path.join(os.path.dirname(__file__),"src"))

base = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base, 'models', 'best_model.pkl')
vectorizer_path = os.path.join(base,'models', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


st.set_page_config(
    page_title = "Fake News Detection",
    layout = "centered"
)

st.title("Fake News Detection")
st.write("Enter Full Article About more than 500 words  **(Article between Aug - Dec 2017)")
st.write("Enter the news article to predict if it's fake or true:")


news = st.text_area("News Article", height = 250)

if st.button("predict"):

    if news.strip() == "":

        st.warning("please enter some news article to predict !!!")
    
    else:

        processed_news = preprocess_text(news)
        news_vectorized = vectorizer.transform([processed_news])
        prediction = model.predict(news_vectorized)[0]

        if prediction == 0:
            st.error("The news article is predicted to be Fake News/Article")
        else:
            st.success("The news article is predicted to be True News/Article")
