import os
import joblib


from preprocess import preprocess_text

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base, 'models')

model = joblib.load(os.path.join(model_path, 'best_model.pkl'))
vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))


def predict_news(text):

    preprocessed_news = preprocess_text(text)
    news_vector = vectorizer.transform([preprocessed_news])
    prediction = model.predict(news_vector)

    if prediction == 0:
        return "Fake News/Article"
    else:
        return "True News/Article"
    

if __name__ == "__main__":
    sample_txt = input("Enter a news article to predict if it's fake or true: ")
    res = predict_news(sample_txt)
    print("The news article is predicted to be {}".format(res))