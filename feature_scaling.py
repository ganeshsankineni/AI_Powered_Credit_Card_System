import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sklearn
import sys
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger=setup_logging('feature_scaling')
import sklearn
import sys
from sklearn.preprocessing import StandardScaler
from all_models import common
from all_models import HyperParameter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
def fs(x_train,y_train,x_test,y_test):
    try:
        logger.info(f"Training data  independent : {x_train.shape}")
        logger.info(f"Training data  dependent : {y_train.shape}")
        logger.info(f"Testing data  independent : {x_test.shape}")
        logger.info(f"Testing data  dependent : {y_test.shape}")
        logger.info(f"before : {x_train.head(1)}")

        sc=StandardScaler()
        sc.fit(x_train)
        x_train_sc=sc.transform(x_train)
        x_test_sc = sc.transform(x_test)

        with open('standscaler.pkl', 'wb') as f:
            pickle.dump(sc, f)
        with open('columns.pkl', 'wb') as f:
            pickle.dump(x_train.columns, f)


        logger.info(f" after : {x_train_sc}")
        common(x_train_sc,y_train,x_test_sc,y_test)
        HyperParameter(x_train_sc, y_train, x_test_sc, y_test)

        reg = LogisticRegression(class_weight='balanced', max_iter=1000)
        reg.fit(x_train_sc, y_train)
        pred = reg.predict(x_test_sc)
        proba = reg.predict_proba(x_test_sc)

        logger.info(f"test accuracy : {accuracy_score(y_test, pred)}")
        logger.info(f"\n{confusion_matrix(y_test, pred)}")
        logger.info(f"\n{classification_report(y_test, pred)}")

        with open('MODEL.pkl', 'wb') as f:
            pickle.dump(reg, f)
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")