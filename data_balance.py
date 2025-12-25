import sys
# from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('data_balance')


class BALANCING_DATA:
    def balance_data(X_train, y_train):
        try:
            # smote=SMOTE(random_state=42)
            # X_train,y_train=smote.fit_resample(X_train,y_train)
            return X_train,y_train
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in line {error_line.tb_lineno}: {error_msg}')