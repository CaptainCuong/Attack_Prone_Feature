import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

file = pd.read_csv('data.csv',sep=',')
file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_BERT']+file['ASR_DeepWordBug'])/4
print(file['ASR'])
file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug','Dataset'],inplace=True)
x_train, x_val, y_train, y_val = train_test_split(file.drop(columns='ASR'), np.array(file['ASR']), test_size = 0.4, random_state = 0)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.2, random_state = 0)
# print(x_train)
# print(y_train)
# Set fixed K-folds
k_fold = KFold(n_splits=5, shuffle=True, random_state=79)

grid = False

if grid:
    # Hyperparameter grid
    param_grid = { 
                  'feature_fraction': np.arange(0.1,1,0.1),
                  'subsample': np.arange(0.70,0.8,0.01),
                  'learning_rate': [0.01],
                  'max_depth': [3,5,7,9],  
                  'num_leaves' : np.arange(10,40,5),
                  'n_estimators' : [3000,4000,5000,7000],
                  'max_bin': [30,40,50,60],
                  "lambda_l1":np.arange(0.0, 1.1, 0.2),
                  "lambda_l2":np.arange(0.0, 1.1, 0.2),
                  "min_gain_to_split": np.arange(0, 36, 2),
                  }

    # Parameter Tuning
    print('Parameter tuning...')
    lgb_est = lgb.LGBMRegressor(random_state = 79, objective = "regression", 
                                metric = "rmse",boosting_type ='gbdt')
    lgb_model = RandomizedSearchCV(lgb_est,param_grid, random_state = 79, n_jobs=-1, 
                                   cv = k_fold, scoring = "neg_mean_squared_error")
    lgb_model.fit(x_train, y_train, 
                eval_set = [(x_train, y_train),(x_val, y_val)],
                eval_metric = ['rmse'],
                eval_names = ["Train","Validation"],
                callbacks=[lgb.early_stopping(10)]
                )

    final_model = lgb_model.best_estimator_
    plt.rcParams['figure.figsize'] = [25, 10]
    lgb.plot_importance(final_model, max_num_features=10, xlabel = 'Value')
else:
    # final_model = lgb.LGBMRegressor(feature_fraction=0.30000000000000004, lambda_l1=0.0,
    #                           lambda_l2=0.2, learning_rate=0.01, max_bin=50, max_depth=9,
    #                           metric='rmse', min_gain_to_split=30, n_estimators=5000,
    #                           num_leaves=25, objective='regression', random_state=79,
    #                           subsample=0.8)
    final_model = lgb.LGBMRegressor(learning_rate=0.05, max_bin=400,
                              metric='rmse', n_estimators=5000,
                              objective='regression', random_state=79,
                              )
    final_model.fit(x_train, y_train,
        eval_set = [(x_val, y_val)],
        eval_metric = ['rmse'],
        callbacks=[lgb.early_stopping(10)])
    predicted_y = final_model.predict(x_test)
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print(summary)
    print(mean_squared_error(y_test, predicted_y,squared=False))
    print(r2_score(y_test, predicted_y))
    x_test = x_test.assign(Groundtruth_ASR=y_test,Predicted_ASR=predicted_y)
    x_test.to_csv('result_summary.csv')
print('Finish')