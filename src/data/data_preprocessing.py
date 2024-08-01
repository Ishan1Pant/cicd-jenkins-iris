from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import pickle 

def load_and_preprocess_data():
    data=load_iris()
    X=data.data 
    y=data.target 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6) 
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test) 
    with open("../../models/preprocessor.pkl","wb") as preprocessor:
        pickle.dump(scaler,preprocessor)
    return X_train_scaled,X_test_scaled,y_train,y_test 

if __name__=="__main__":
    load_and_preprocess_data()
    print("Preprocessing Successful")