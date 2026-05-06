import numpy as np
import pandas as pd
import sys

class CHRUN:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            print("Data loaded successfully")
            dd = self.df
            for i in dd:
                if dd[i].dtype == 'object':
                    dd[i] = dd[i].str.strip()
                    dd[i] = dd[i].replace('', np.nan)

            dd['TotalCharges'] = pd.to_numeric(dd['TotalCharges'], errors='coerce')

            print(dd.isnull().sum())

        except Exception as e:
            er_type, er_obj, er_tb = sys.exc_info()
            print(f"Error: {e}")
            print(f"Error in line no: {er_tb.tb_lineno}")
if __name__ == '__main__':
    try:
        obj = CHRUN('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except Exception as e:
        er_type, er_obj, er_tb = sys.exc_info()
        print(f"Error: {e}")
        print(f"Error in line no: {er_tb.tb_lineno}")