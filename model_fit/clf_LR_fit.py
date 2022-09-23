import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 


from utils import X_train, y_train, age_mean

from sklearn.linear_model import LogisticRegression

import pickle


def feature_engineering(X, age_mean):
    """adds extra features to a DataFrame"""
    # filling missing values
    X["Age"].fillna(age_mean, inplace=True)
    # one-hot encoding
    X["female"] = (X["Sex"]=="female").astype(int)
    
    # interaction terms
    X["female_pclass"] = X["female"]       * X["Pclass"]
    X["male_pclass"]   = (1 - X["female"]) * X["Pclass"]
    X.drop(["Sex", "Pclass"], inplace = True, axis = 1)
    return X


m = LogisticRegression(max_iter=1000)


if __name__ == "__main__":
   X_train_fe = feature_engineering(X_train, age_mean)
   clf_LR_fit = m.fit(X_train_fe, y_train)
   print("model succesfully fit") 
   with open("../artefacts/clf_fitted.bin", "wb") as file_out:
        pickle.dump(clf_LR_fit, file_out)
