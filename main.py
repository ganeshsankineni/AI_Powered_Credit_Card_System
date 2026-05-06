import sys
from statistics import kde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import logging
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger = setup_logging("main")
from sklearn.model_selection import train_test_split
from missing_value_median import handling_missing_values
from var_tran import vt_outliers
from feature import f_s
from cat_number import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs
class CHRUN:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f"Data set size: {self.df.shape}")
            self.df['Telecom_Partner'] = np.where(
                self.df['PaymentMethod'] == 'Mailed check', 'Airtel',
                np.where(
                    self.df['PaymentMethod'] == 'Bank transfer (automatic)', 'VI-!dea',
                    np.where(
                        self.df['PaymentMethod'] == 'Credit card (automatic)', 'BSNL',
                        'Jio'
                    )
                )
            )
            logger.info("Telecom_Partner column created successfully")
            logger.info(f'after adding new column:{self.df.shape}')
            logger.info('before replacing empty spaces')
            logger.info(self.df.isnull().sum())
            self.df = self.df.drop(['customerID'] ,axis=1)
            for i in self.df.columns:
                if self.df[i].dtype == 'object' or  self.df[i].dtype.name == 'string':
                    self.df[i] = self.df[i].str.strip()
                    self.df[i] = self.df[i].replace('', np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            logger.info('after  replacing empty spaces ')
            logger.info(self.df.isnull().sum())
            self.y = self.df['Churn'].map({'Yes':1, 'No':0})
            self.x = self.df.drop(columns=['Churn'])

            self.x_train, self.x_test,self.y_train,self.y_test=train_test_split(self.x, self.y,test_size=0.2,random_state=42)

            logger.info(f'train data size: {len(self.x_train)}:{len(self.x_train)} \n total test data {self.x_train.shape}')
            logger.info(f'test data size: {len(self.x_test)}:{len(self.x_test)}  \n total test data {self.x_test.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def missing_values(self):
        try:
            logger.info(
                f'before handling null values x_train names and shape :{self.x_train.shape}:{self.x_train.columns}:{self.x_train.isnull().sum()}')
            logger.info(
                f'before handling null values x_test names and shape :{self.x_test.shape}:{self.x_test.columns}:{self.x_test.isnull().sum()}')
            self.x_train, self.x_test = handling_missing_values(self.x_train, self.x_test)
            logger.info(
                f'after handling null values x_train names and shape :{self.x_train.shape}:{self.x_train.columns}:{self.x_train.isnull().sum()}')
            logger.info(
                f'after handling null values x_test names and shape :{self.x_test.shape}:{self.x_test.columns}:{self.x_test.isnull().sum()}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def data_separation(self):
        try:
            self.x_train_num_cols=self.x_train.select_dtypes(exclude ='object')
            self.x_test_num_cols = self.x_test.select_dtypes(exclude = 'object')
            self.x_train_cat_cols = self.x_train.select_dtypes(include = 'object')
            self.x_test_cat_cols = self.x_test.select_dtypes(include = 'object')
            logger.info(f'x_train_num_cols:{self.x_train_num_cols.columns}:{self.x_train_num_cols.shape}')
            logger.info(f'x_test_num_cols:{self.x_test_num_cols.columns}:{self.x_test_num_cols.shape}')
            logger.info(f'x_train_cat_cols:{self.x_train_cat_cols.columns}:{self.x_train_cat_cols.shape}')
            logger.info(f'x_test_cat_cols:{self.x_test_cat_cols.columns}:{self.x_test_cat_cols.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def variable_transfermation(self):
        try:
            logger.info(f'before train column names : {self.x_train_num_cols.columns}')
            logger.info(f'before column names : {self.x_test_num_cols.columns}')
            self.x_train_num_cols,self.x_test_num_cols=vt_outliers(self.x_train_num_cols,self.x_test_num_cols)
            logger.info(f'after train column names : {self.x_train_num_cols.columns}')
            logger.info(f'after column names : {self.x_test_num_cols.columns}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def feature_selection(self):
        try:
            self.x_train_num_cols, self.x_test_num_cols = f_s(self.x_train_num_cols,self.x_test_num_cols,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def cat_num(self):
        try:
            self.x_train_cat_cols,self.x_test_cat_cols=c_t_n(self.x_train_cat_cols,self.x_test_cat_cols)
            self.x_train_num_cols.reset_index(drop=True, inplace=True)
            self.x_train_cat_cols.reset_index(drop=True, inplace=True)
            self.x_test_num_cols.reset_index(drop=True, inplace=True)
            self.x_test_cat_cols.reset_index(drop=True, inplace=True)
            self.training_data=pd.concat([self.x_train_num_cols,self.x_train_cat_cols],axis=1)
            self.testing_data = pd.concat([self.x_test_num_cols, self.x_test_cat_cols], axis=1)
            logger.info(f'final training data : {self.training_data.shape}')
            logger.info(f'nulls final training data : {self.training_data.isnull().sum()}')
            logger.info(f'final testing data : {self.testing_data.shape}')
            logger.info(f'nulls final testing data : {self.testing_data.isnull().sum()}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
    def data_balancing(self):
        try:
            logger.info(f" Before Number of Rows for Good Customer {1} : {sum(self.y_train == 1)}")
            logger.info(f"Before Number of Rows for Bad Customer {0} : {sum(self.y_train == 0)}")
            logger.info(f"Before balancing training_data : {self.training_data.shape}")
            sm=SMOTE(random_state = 42)
            self.training_data_bal,self.y_train_bal =sm.fit_resample(self.training_data,self.y_train)
            logger.info(f" After Number of Rows for Good Customer {1} : {sum(self.y_train_bal == 1)}")
            logger.info(f" After Number of Rows for Bad Customer {0} : {sum(self.y_train_bal == 0)}")
            logger.info(f"After balancing training_data : {self.training_data_bal.shape}")
            fs(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.error(f"Error: {e}")
            logger.error(f"Error in line no: {er_line.tb_lineno}")
if __name__ == '__main__':
    try:
        obj = CHRUN('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.missing_values()
        obj.data_separation()
        obj.variable_transfermation()
        obj.feature_selection()
        obj.cat_num()
        obj.data_balancing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")