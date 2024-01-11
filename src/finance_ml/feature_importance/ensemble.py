import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss,accuracy_score
import shap


def mdi(model, feature_names):
    '''
    Calculate feature importances based on mean impurity reduction.

    Arguments:

        model: The fitted model object.
        feature_names: A index of feature names.

    Output:

        mdi: DataFrame containing the mean of feature importances normalized by the sum of mean importances.

    Description:

        This function takes a fitted model object and a list of feature names as input. It calculates the feature importances based on the mean impurity reduction. 
    '''


    importance_mdi = model.feature_importances_

    indices_mdi = importance_mdi.argsort()[::-1]
    
    return pd.DataFrame({'FEATURE':feature_names[indices_mdi],'IMPORTANCE':importance_mdi[indices_mdi]})
        
def mda(model, X, y):

    '''
    Calculate feature importances based on mean impurity reduction.

    Arguments:

        model: The fitted model object.
        X: Variables used for training
        y: Target Variable

    Output:

        mda: DataFrame containing the mean of feature importances normalized by the sum of mean importances.

    Description:

        This function takes a fitted model object and a list of feature names as input. It calculates the feature importances based on the mean decrease accuracy. 
    '''

    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=0)

    importance_mda = perm_importance.importances_mean

    indices_mda = importance_mda.argsort()[::-1]

    return pd.DataFrame({'FEATURE':X.columns[indices_mda],'IMPORTANCE':importance_mda[indices_mda]})

def shap_func(model, X, feature_names):

    '''
    Calculate feature importances based on shap absolute values.

    Arguments:

        model: The fitted model object.
        X: Variables used for training
        feature_names: A index of feature names.

    Output:

        shap: DataFrame containing the mean of feature importances normalized by the sum of mean importances.

    Description:

        This function takes a fitted model object and a list of feature names as input. It calculates the feature importances based on the shap metric. 
    '''

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X[feature_names])[1]

    shap_mean = abs(shap_values).mean(axis=0)

    final=pd.DataFrame(data=[shap_mean,feature_names],
                        index=['IMPORTANCE','FEATURE']).T

    return final

