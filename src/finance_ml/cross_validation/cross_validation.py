'''
    Feito em 06/2023

    Grupo
        Beatriz Loureiro
        Giovanni Dalvi
        Richard Sousa
'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, r2_score, accuracy_score

class CrossValidation():
    # k_folds: number of divisions of data
    # overlap: decimal number of drop before and after test set
    # embargo_rate: decimal number of additional drop after test set
    def __init__(self, k_folds = 5, overlap = 0.01, embargo_rate = 0):
        self.k_folds = k_folds 
        self.embargo_rate = embargo_rate
        self.overlap = overlap
        self.metrics_error = {
            'mean_squared_error': mean_squared_error, 
            'max_error': max_error, 
            'mean_absolute_error': mean_absolute_error,
            'r2_score': r2_score,
            'accuracy': accuracy_score
            }
        
    # X = features
    # Y = target
    # metrics_error: choose metrics error function available on the dict "metrics_error"
    # model: choose an available model on the sklearn lib
    # if discrete model, use accuracy_score as your error_metric
    def cross_validation_score(self, X, Y, model, error_metric = 'mean_squared_error'):
        # if 'mean_squared_error' == 
        if self.overlap < 0 or self.embargo_rate < 0 or  error_metric not in self.metrics_error.keys(): 
            #if the user use overlap or embargo_rate less than 0, then the program shouldn't work.
            raise Exception("Wrong parameters!" ) 
        
        #choose metric error
        error_metric_function = self.metrics_error[error_metric]

        # We will compute the total size of data:
        size = len(X.index)

        # We will calculate the error for every fold:
        Total_error = []
        # We will set the size of the embargo in relation of total data:
        embargo = int(size * self.embargo_rate)
        overlap = int(size * self.overlap)

        # We will define the size of each fold. The last fold will be a bit bigger if the size of data isn't divisive by the
        # number of folds.
        fold = size // self.k_folds
        last_fold = fold + size % self.k_folds
        
        # Now we will divide the data between train and test, off course they will be disjunt sets, we will apply the embargo after
        # the test data, in order to prevent linkage of information from test to train.
        for j in range(self.k_folds-1):
            X_test = X.iloc[j*fold:(j+1)*fold]
            Y_test = Y.iloc[j*fold:(j+1)*fold]

            # This "if" will be use for the first k_fold:
            if j ==0:
                X_train = X.iloc[(j+1)*fold + overlap + embargo:]
                Y_train = Y.iloc[(j+1)*fold + overlap + embargo:]

            #This part will deal with all the other parts of the data, until the time we use the last fold as the test data
            else:
                X_train = pd.concat([X.iloc[:j*fold - overlap],X.iloc[(j+1)*fold + overlap + embargo:]])
                Y_train = pd.concat([Y.iloc[:j*fold - overlap],Y.iloc[(j+1)*fold + overlap + embargo:]])

            # We will train the model with the train data:
            model.fit(X_train,Y_train)

            # We will predict the output with our model:
            prediction = model.predict(X_test)

            # We will vectorize to do quick calculations:
            vector_pred = np.array(prediction)
            vector_Y_test = np.array(Y_test)

            # We will use for real output the mean square error:
            error = error_metric_function(vector_pred, vector_Y_test)
            Total_error.append(error)

        # Some carefull will be take with the last data, we will use another size of fold, as explained, as it is the last part of
        # the data, we will not care about embargo, we just care about the overlap.
        X_test, Y_test = X.iloc[-last_fold :], Y.iloc[-last_fold:]
        X_train, Y_train = X.iloc[:-last_fold - overlap], Y.iloc[:-last_fold - overlap]

        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        vector_pred = np.array(prediction)

        vector_Y_test = np.array(Y_test)
        error = (1/len(Y_test))*sum((vector_pred - vector_Y_test)**2)
        Total_error.append(error)

        return Total_error