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
logger=setup_logging('cat_number')
import sklearn
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def c_t_n(x_train_cat,x_test_cat):
    try:
        logger.info(f'before nominal x_train_cat:{x_train_cat.drop(['Contract'],axis=1).shape}:\n:{x_train_cat.columns}')
        logger.info(f'before nominal x_test_cat:{x_test_cat.drop(['Contract'],axis=1).shape}:\n:{x_test_cat.columns}')
        oh=OneHotEncoder(drop='first')
        oh.fit(x_train_cat.drop(['Contract'],axis=1))
        values_train=oh.transform(x_train_cat.drop(['Contract'],axis=1)).toarray()
        values_test = oh.transform(x_test_cat.drop(['Contract'],axis=1)).toarray()
        t_train = pd.DataFrame(values_train)
        t_train.columns = oh.get_feature_names_out()
        t_test = pd.DataFrame(values_test)
        t_test.columns = oh.get_feature_names_out()
        x_train_cat.reset_index(drop=True, inplace=True)
        x_test_cat.reset_index(drop=True, inplace=True)
        t_train.reset_index(drop=True, inplace=True)
        t_test.reset_index(drop=True, inplace=True)
        x_train_cat = pd.concat([x_train_cat, t_train], axis=1)
        x_test_cat = pd.concat([x_test_cat, t_test], axis=1)
        x_train_cat=x_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner'],axis=1)
        x_test_cat = x_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                        'TechSupport', 'StreamingTV', 'StreamingMovies',
                                        'PaperlessBilling', 'PaymentMethod', 'Telecom_Partner'], axis=1)
        logger.info(f'after nominal x_train_cat:{x_train_cat.drop(['Contract'],axis=1).shape}:\n:{x_train_cat.columns}')
        logger.info(f'after nominal x_test_cat:{x_test_cat.drop(['Contract'],axis=1).shape}:\n:{x_test_cat.columns}')
        logger.info("====================ordinal=================================================")
        logger.info(
            f'before ordinal x_train_cat:{x_train_cat.shape}:\n:{x_train_cat.columns}')
        logger.info(f'before ordinal x_test_cat:{x_test_cat.shape}:\n:{x_test_cat.columns}')
        od= OrdinalEncoder()
        od.fit(x_train_cat[['Contract']])
        ord_train = od.transform(x_train_cat[['Contract']])
        ord_test = od.transform(x_test_cat[['Contract']])
        t1o_train = pd.DataFrame(ord_train)
        t2o_test = pd.DataFrame(ord_test)
        t1o_train.columns = ['Contract_re']
        t2o_test.columns = ['Contract_re']
        t1o_train.reset_index(drop=True, inplace=True)
        t2o_test.reset_index(drop=True, inplace=True)
        x_train_cat = pd.concat([x_train_cat, t1o_train], axis=1)
        x_test_cat = pd.concat([x_test_cat, t2o_test], axis=1)
        x_train_cat = x_train_cat.drop(['Contract'], axis=1)
        x_test_cat = x_test_cat.drop(['Contract'], axis=1)
        #x_train_cat = x_train_cat.drop(['Contract'], axis=1)
        #x_test_cat = x_test_cat.drop(['Contract'], axis=1)
        logger.info(
            f'after ordinal x_train_cat:{x_train_cat.shape}:\n:{x_train_cat.columns}')
        logger.info(f'after ordinal x_test_cat:{x_test_cat.shape}:\n:{x_test_cat.columns}')
        logger.info(f'null values in x_train_cat :{x_train_cat.isnull().sum()}')
        logger.info(f'null values in x_train_cat :{x_train_cat.isnull().sum()}')
        return x_train_cat, x_test_cat
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")




