import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger=setup_logging('missing_value_median')


def handling_missing_values(x_train,x_test):
    try:
        logger.info(f'before handling null values x_train names and shape :{x_train.shape}:{x_train.columns}:{x_train.isnull().sum()}')
        logger.info(f'before handling null values x_test names and shape :{x_test.shape}:{x_test.columns}:{x_test.isnull().sum()}')
        for i in x_train.columns:
            if x_train[i].isnull().sum() > 0:
                x_train[i+"_replaced"]=x_train[i].copy()
                x_test[i + "_replaced"] = x_test[i].copy()
                x_train[i+"_replaced"]=x_train[i+"_replaced"].fillna(x_train[i].median())
                x_test[i + "_replaced"]=x_test[i + "_replaced"].fillna(x_train[i].median())
                x_train=x_train.drop([i],axis=1)
                x_test=x_test.drop([i],axis=1)
        logger.info(f'after handling null values x_train names and shape :{x_train.shape}:{x_train.columns}:{x_train.isnull().sum()}')
        logger.info(f'before handling null values x_test names and shape :{x_test.shape}:{x_test.columns}:{x_test.isnull().sum()}')
        return x_train, x_test
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")

