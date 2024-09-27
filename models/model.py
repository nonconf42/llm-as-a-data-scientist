import sklearn
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV


problem_type_score = {
    'classification' : 'accuracy',
    'regression' : 'neg_root_mean_squared_error'
}
class Model:
    def __init__(self, model_name, problem_type):
        self.model_name = model_name
        self.problem_type = problem_type
        if model_name == 'XGB' and problem_type == 'classification':
            self.model = xgb.XGBClassifier(
                learning_rate = 0.02,
                n_estimators= 2000,
                max_depth= 4,
                min_child_weight= 2,
                gamma=0.9,                   
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread= -1,
                scale_pos_weight=1)
            
        elif model_name == 'XGB' and problem_type == 'regression':
            self.model = xgb.XGBRegressor(
                learning_rate = 0.02,
                n_estimators = 2000,
                max_depth = 5
            )
            
    def fit(self, data, subset='all'):
        if self.model_name == 'XGB':
            parameters = {
                'max_depth': [3, 5, 7, 9], 
                'n_estimators': [5, 10, 15, 20, 25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1]
            }

        self.grid_search = GridSearchCV(
            self.model, 
            parameters, 
            cv=5,
            scoring=problem_type_score[self.problem_type],
        )
        if subset == 'all':
            self.grid_search.fit(data.train_input_clean, data.train_labels)
            self.model = self.grid_search.best_estimator_
            self.score = self.grid_search.best_score_
            
        elif subset == 'top':
            self.grid_search.fit(data.train_input_selected, data.train_labels)
            self.model = self.grid_search.best_estimator_
            self.score = self.grid_search.best_score_


        