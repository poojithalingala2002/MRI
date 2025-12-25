import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,roc_curve,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('model_training')

class TRAINING:
    def all_models(X_train,X_test,y_train,y_test):
        try:
            classifiers = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
                "Random Forest": RandomForestClassifier(criterion='entropy', n_estimators=10),
                "AdaBoost": AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=10),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=10),
                "XGBoost": XGBClassifier()
            }

            plt.figure(figsize=(8, 6))

            best_auc = 0
            best_model = None
            best_model_name = None

            for name, model in classifiers.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Log metrics
                logger.info(f"==================={name}=======================")
                logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                # Get probability scores for ROC
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except:
                    y_prob = model.decision_function(X_test)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                logger.info(f"AUC Score ({name}): {roc_auc}")
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")

                #track best model
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_model = model
                    best_model_name = name

            # Random guess lineq
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves - All Models")
            plt.legend(loc="lower right")
            plt.show()

            logger.info("==============================================")
            logger.info(f"Best Model Based on AUC: {best_model_name}")
            logger.info(f"Best AUC Score: {best_auc}")
            logger.info("==============================================")

            # Save the best model
            with open('best_model.pkl', 'wb') as f1:
                pickle.dump(best_model, f1)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')