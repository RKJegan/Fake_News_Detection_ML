import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from data_loader import load_data
from feature_eng import create_features

def train_models():

    data = load_data()
    x, y, vectorizer = create_features(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y)

    models = {

    "Logistic Regession" : LogisticRegression(max_iter = 1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(random_state = 42),
    "Linear SVC": LinearSVC(random_state = 42)
    }


    trained_models = {}

    for name, model in models.items():

        print("=" * 50)
        print(name)

        model.fit(x_train,y_train)

        trained_models[name] = model

    return trained_models, x_test, y_test
    

    
if __name__ == "__main__":

    trained_models, x_test, y_test = train_models()
    print()

    for model in trained_models.keys():

        print(model, "Trained Successfully!")