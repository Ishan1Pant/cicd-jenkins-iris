from sklearn.svm import SVC 
import pickle
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_preprocessing import load_and_preprocess_data 

def train_model():
    X_train,X_test,y_train,y_test=load_and_preprocess_data()
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train,y_train)
    with open ("../../models/classifier.pkl","wb") as model_file:
        pickle.dump(model,model_file)
    
    print("Model trained Successfully")

if __name__=="__main__":
    train_model()