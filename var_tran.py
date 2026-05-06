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
logger=setup_logging('var_tran')
from scipy.stats import yeojohnson
from scipy.stats import boxcox
def vt_outliers(x_train_num,x_test_num):
    try:
        logger.info(f'before train column names : {x_train_num.columns}')
        logger.info(f'before column names : {x_test_num.columns}')
        #MonthlyCharges
        x_train_num['MonthlyCharges'+'_yeo'],lam_value = yeojohnson(x_train_num['MonthlyCharges'])
        x_test_num['MonthlyCharges'+'_yeo']=yeojohnson(x_test_num['MonthlyCharges'], lmbda=lam_value)
        x_train_num=x_train_num.drop(['MonthlyCharges'],axis=1)
        x_test_num=x_test_num.drop(['MonthlyCharges'],axis=1)
        # TotalCharges_replaced
        x_train_num['TotalCharges_replaced' + '_boxcox'], lam_value = boxcox(x_train_num['TotalCharges_replaced'])
        x_test_num['TotalCharges_replaced' + '_boxcox'] = boxcox(x_test_num['TotalCharges_replaced'],lmbda=lam_value)
        x_train_num = x_train_num.drop(['TotalCharges_replaced'], axis=1)
        x_test_num = x_test_num.drop(['TotalCharges_replaced'], axis=1)
        # tenure
        x_train_num['tenure' + '_sqrt'] = np.sqrt(x_train_num['tenure'])
        x_test_num['tenure' + '_sqrt'] = np.sqrt(x_test_num['tenure'])
        x_train_num = x_train_num.drop(['tenure'], axis=1)
        x_test_num = x_test_num.drop(['tenure'], axis=1)
        # trimming
        iqr = x_train_num['MonthlyCharges'+ '_yeo'].quantile(0.75) - x_train_num['MonthlyCharges'+ '_yeo'].quantile(0.25)
        upper_limit = x_train_num['MonthlyCharges'+ '_yeo'].quantile(0.75) + (1.5 * iqr)
        lower_limit = x_train_num['MonthlyCharges'+ '_yeo'].quantile(0.25) - (1.5 * iqr)
        x_train_num['MonthlyCharges'+ '_trim'] = np.where(x_train_num['MonthlyCharges'+ '_yeo'] > upper_limit, upper_limit,
                                            np.where(x_train_num['MonthlyCharges'+ '_yeo'] < lower_limit, lower_limit,
                                                     x_train_num['MonthlyCharges'+ '_yeo']))
        x_test_num['MonthlyCharges'+ '_trim'] = np.where(x_test_num['MonthlyCharges'+ '_yeo'] > upper_limit, upper_limit,
                                           np.where(x_test_num['MonthlyCharges'+ '_yeo'] < lower_limit, lower_limit,
                                                    x_test_num['MonthlyCharges'+ '_yeo']))
        x_train_num = x_train_num.drop(['MonthlyCharges'+ '_yeo'], axis=1)
        x_test_num = x_test_num.drop(['MonthlyCharges'+ '_yeo'], axis=1)

        # tenure
        iqr = x_train_num['TotalCharges_replaced' + '_boxcox'].quantile(0.75) - x_train_num['TotalCharges_replaced' + '_boxcox'].quantile(0.25)
        upper_limit = x_train_num['TotalCharges_replaced' + '_boxcox'].quantile(0.75) + (1.5 * iqr)
        lower_limit = x_train_num['TotalCharges_replaced' + '_boxcox'].quantile(0.25) - (1.5 * iqr)
        x_train_num['TotalCharges_replaced' + '_trim'] = np.where(x_train_num['TotalCharges_replaced' + '_boxcox'] > upper_limit,
                                               upper_limit,
                                               np.where(x_train_num['TotalCharges_replaced' + '_boxcox'] < lower_limit,
                                                        lower_limit,
                                                        x_train_num['TotalCharges_replaced' + '_boxcox']))
        x_test_num['TotalCharges_replaced' + '_trim'] = np.where(x_test_num['TotalCharges_replaced' + '_boxcox'] > upper_limit, upper_limit,
                                              np.where(x_test_num['TotalCharges_replaced' + '_boxcox'] < lower_limit,
                                                       lower_limit,
                                                       x_test_num['TotalCharges_replaced' + '_boxcox']))
        x_train_num = x_train_num.drop(['TotalCharges_replaced' + '_boxcox'], axis=1)
        x_test_num = x_test_num.drop(['TotalCharges_replaced' + '_boxcox'], axis=1)

        # TotalCharges
        iqr = x_train_num['tenure' + '_sqrt'].quantile(0.75) - x_train_num['tenure' + '_sqrt'].quantile(0.25)
        upper_limit = x_train_num['tenure' + '_sqrt'].quantile(0.75) + (1.5 * iqr)
        lower_limit = x_train_num['tenure' + '_sqrt'].quantile(0.25) - (1.5 * iqr)
        x_train_num['tenure' + '_trim'] = np.where(x_train_num['tenure' + '_sqrt'] > upper_limit,
                                                   upper_limit,
                                                   np.where(x_train_num['tenure' + '_sqrt'] < lower_limit,
                                                            lower_limit,
                                                            x_train_num['tenure' + '_sqrt']))
        x_test_num['tenure' + '_trim'] = np.where(x_test_num['tenure' + '_sqrt'] > upper_limit, upper_limit,
                                                  np.where(x_test_num['tenure' + '_sqrt'] < lower_limit,
                                                           lower_limit,
                                                           x_test_num['tenure' + '_sqrt']))
        x_train_num = x_train_num.drop(['tenure' + '_sqrt'], axis=1)
        x_test_num = x_test_num.drop(['tenure' + '_sqrt'], axis=1)

        logger.info(f'after train column names : {x_train_num.columns}')
        logger.info(f'after column names : {x_test_num.columns}')
        return x_train_num, x_test_num

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.error(f"Error: {e}")
        logger.error(f"Error in line no: {er_line.tb_lineno}")