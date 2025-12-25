import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
#from missing_val_handle import VALUE_MISSING
#from variable_transformation_outlierhandle import VAR_TRANS_OUT_HANDLE
# from data_balance import BALANCING_DATA
from model_training import TRAINING


class HEART_DISEASE_PREDICTION:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'Data we have :\n{self.df}')
            logger.info(f'shape:\n{self.df.shape}')
            logger.info(f'null values:\n{self.df.isnull().sum()}')
            for i in self.df.columns:
                logger.info(f'{self.df[i].dtype}')
            self.X=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            logger.info(f'X shape:\n{self.X.shape}')
            logger.info(f'y shape:\n{self.y.shape}')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            # logger.info(f'X_train shape:\n{self.X_train.shape}')
            # logger.info(f'y_train shape:\n{self.y_train.shape}')
            # logger.info(f'X_test shape:\n{self.X_test.shape}')
            # logger.info(f'y_test shape:\n{self.y_test.shape}')
            # logger.info(f'X_train null:\n{self.X_train.isnull().sum()}')
            # logger.info(f'y_train null:\n{self.y_train.isnull().sum()}')
            # logger.info(f'X_test null:\n{self.X_test.isnull().sum()}')
            # logger.info(f'y_test null:\n{self.y_test.isnull().sum()}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def missing_values(self):
        try:
            if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                self.X_train,self.X_test=VALUE_MISSING.random_sample(self.X_train,self.X_test)
            else:
                logger.info(f'There are no missing values in X_train and X_test')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def var_trans_out_handling(self):
        try:
            #logger.info(f'{self.X_train.info()}')
            for i in self.X_train.columns:
                logger.info(f'{self.X_train[i].dtype}')
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            self.X_train,self.X_test=VAR_TRANS_OUT_HANDLE.variable_transform_outliers(self.X_train,self.X_test)
            logger.info(f'===========================================================')
            logger.info(f'columns of X_train_num:{self.X_train.columns}')
            logger.info(f'columns of X_test_num:{self.X_test.columns}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    # def data_balancing(self):
    #     try:
    #         logger.info(f"Before SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
    #         self.X_train,self.y_train=BALANCING_DATA.balance_data(self.X_train,self.y_train)
    #         logger.info(f"After SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
    #         logger.info(f"Balanced X_train shape: {self.X_train.shape}")
    #         logger.info(f"Balanced y_train shape: {self.y_train.shape}")
    #
    #     except Exception as e:
    #         error_type, error_msg, error_line = sys.exc_info()
    #         logger.error(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    """
        Here there are no catogorical columns so cat to num conversion no need
    """

    """
        Since there are only 13 independent columns all are important so feature selection not need
    """

    def data_scaling(self):
        try:
            logger.info(f'{self.X_train.shape}')
            logger.info(f'{self.X_test.shape}')
            logger.info(f'Before \n:{self.X_train}')
            logger.info(f'Before \n:{self.X_test}')
            scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

            sc = StandardScaler()
            sc.fit(self.X_train[scale_cols])

            self.X_train[scale_cols] = sc.transform(self.X_train[scale_cols])
            self.X_test[scale_cols] = sc.transform(self.X_test[scale_cols])

            # Save scaler for inference
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(sc, f)

            logger.info(f'{self.X_train.shape}')
            logger.info(f'{self.X_test.shape}')
            logger.info(f'Before \n:{self.X_train}')
            logger.info(f'Before \n:{self.X_test}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def algo(self):
        try:
            TRAINING.all_models(self.X_train,self.X_test,self.y_train,self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == '__main__':
    try:
        obj=HEART_DISEASE_PREDICTION('"C:\\Users\\pooji\\Downloads\\MRI-main\\MRI-main"')
        obj.missing_values()
        #obj.var_trans_out_handling()
        # obj.data_balancing()
        #obj.data_scaling()
        obj.algo()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')