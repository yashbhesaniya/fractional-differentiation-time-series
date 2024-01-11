import pytest

import pandas as pd
import numpy as np
import os 
import sys
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.realpath('./ML-in-Finance'))

from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.data_preparation.asset_class import Asset
from src.finance_ml.data_preparation.portfolio_class import Portfolio
from src.finance_ml.feature_importance.ensemble import mdi,mda,shap_func

def test_ensemble():

    '''
    Test feature importance module.
    '''

    # Generating a random matrix
    X = pd.DataFrame(data = np.random.normal(size=(1000,10)),columns=range(0,10))

    Y = pd.DataFrame(np.random.randint(2, size=(1000,1)),columns=['CLASSE'])

    #Create a sklearn model
    clf = RandomForestClassifier()

    clf.fit(X,Y['CLASSE'])

    impurity = mdi(clf,X.columns)
    accuracy = mda(clf,X,Y['CLASSE'])
    shap_f = shap_func(clf,X,list(X.columns))


    #Test types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert isinstance(impurity, pd.DataFrame)
    assert isinstance(accuracy, pd.DataFrame)
    assert isinstance(shap_f, pd.DataFrame)


    
