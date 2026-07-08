import joblib
import pandas as pd
import os

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report)

from model_training import train_models

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base, 'models')
result_path = os.path.join(base, 'models_result')

os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

def evaluate_models():

    trained_models, x_test, y_test = train_models()

    evaluation_results = []

    
    best_model = None
    best_model_name = ""
    best_f1 = -1
    best_roc_auc = -1
    best_accuracy = -1



    for name, model in trained_models.items():

        prediction = model.predict(x_test)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        

        

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(x_test)
            roc_auc = roc_auc_score(y_test, y_scores)
        else:
            roc_auc = None


        evaluation_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        

        if f1 > best_f1:
            best_f1 = f1
            best_roc_auc = roc_auc
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
        
        elif f1 == best_f1:

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
            
            elif roc_auc == best_roc_auc:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
    
    results_df = pd.DataFrame(evaluation_results)

    results_df = results_df.sort_values(by=["F1 Score", "ROC AUC", "Accuracy"], ascending=False)

    results_df.to_csv(os.path.join(result_path, "evaluation_results.csv"), index=False)
    print(results_df)


    best_pred = best_model.predict(x_test)

    classification_rep = classification_report(y_test, best_pred)
    matrix = confusion_matrix(y_test, best_pred)
    
    print("confusion matrix and classification report for the best model:")
    print(matrix)
    print(classification_rep)

    with open(os.path.join(result_path, "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(classification_rep)

    with open(os.path.join(result_path, "confusion_matrix.txt"), "w", encoding="utf-8") as file:
        file.write("Confusion Matrix\n\n")
        file.write(str(matrix))

    joblib.dump(best_model, os.path.join(model_path, "best_model.pkl"),compress = 3)  # Save the best model for future use

    print("\n" + "=" * 50)
    print(f"Best Model: {best_model_name}")
    print(f"F1 Score: {best_f1}")
    print(f"ROC AUC : {best_roc_auc}")
    print(f"Accuracy : {best_accuracy}")

    return "Best Model Saved Successfully!"


if __name__ == "__main__":
    evaluate_models()