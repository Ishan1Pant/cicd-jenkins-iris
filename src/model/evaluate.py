from sklearn.metrics import accuracy_score,classification_report
import pickle
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_preprocessing import load_and_preprocess_data 


def evaluate_model():
    X_train,X_test,y_train,y_test=load_and_preprocess_data()
    with open ("../../models/classifier.pkl","rb") as cls:
        model=pickle.load(cls)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_pred,y_test)
    class_rep=classification_report(y_pred,y_test)

    print(f"Accuracy Score : {accuracy}")
    print("Classification Report :")
    print(class_rep)

if __name__=="__main__":
    evaluate_model()