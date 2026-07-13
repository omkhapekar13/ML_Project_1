import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from  sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree":{
                    'criterion':['squared_error','absolute_error','poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': [None, 'sqrt', 'log2']
                },
                "Random Forest":{
                    #'n_estimators':[8,16,32,64,128,256],
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                },
                "Adaboost Regressor":{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.001, 0.01, 0.1, 1]
                },
                "Gradient Boosting":{
                    "loss": ["squared_error", "absolute_error", "huber"],
                    # "learning_rate": [0.01, 0.05, 0.1],
                    # "n_estimators": [100, 200],
                    ##"max_depth": [3, 5]
                },
                "XGBRegressor":{
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'max_depth': [3, 5, 7],
                    # 'min_child_weight': [1, 3, 5],
                    # 'subsample': [0.8, 1.0],
                    # 'colsample_bytree': [0.8, 1.0],
                    # 'gamma': [0, 0.1, 0.3],
                    # 'reg_alpha': [0, 0.1, 1],
                    # 'reg_lambda': [1, 5, 10]
                },
                "CatBoosting Regressor":{
                    'iterations': [200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'depth': [4, 6, 8, 10],
                    # 'l2_leaf_reg': [1, 3, 5, 7],
                    # 'border_count': [32, 64, 128]
                },
                "K-Neighbors Regressor":{
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'leaf_size': [20, 30, 40],
                    # 'p': [1, 2]
                },
                "Linear Regression":{
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            }

            model_report: dict = evaluate_model(x_train,x_test,y_train,y_test,models,params)

            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training & testing datasets : {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square =  r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)