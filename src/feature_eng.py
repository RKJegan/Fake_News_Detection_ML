from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base, 'models')

from preprocess import preprocess_text

def create_features(data):

    data['content'] = data['content'].apply(preprocess_text)  # Preprocess the content column

    vectorizer = TfidfVectorizer(max_features = 5000)  # Limit to top 5000 features
    
    x = vectorizer.fit_transform(data['content'])
    y = data['label']



    joblib.dump(vectorizer, os.path.join(model_path, 'tfidf_vectorizer.pkl'))  # Save the vectorizer for future use

    return x, y, vectorizer



if __name__ == "__main__":

    from data_loader import load_data

    data = load_data()

    x,y,vectorizer = create_features(data)

    print("Feature matrix shape:", x.shape)
    print("Labels shape:", y.shape)
