import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
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
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,explained_variance_score,\
                            mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import itertools
from sklearn.inspection import permutation_importance
from PyALE import ale
import statsmodels.api as sm

datasets = ['amazon_review_full', # 18
           'amazon_review_polarity','dbpedia', # 56, 71
           'yahoo_answers','ag_news', # 11, 49
           'yelp_review_full','yelp_review_polarity'] # 2, 69
attackers = ['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug']

file = pd.read_csv('data_test.csv',sep=',')
file = file[file.notnull().all(1)]
file = file[(file!='Nan').all(1)]
file.drop(columns=['Index','Number of labels'],inplace=True)
file = file.astype({'ASR_BERT': 'float64','ASR_DeepWordBug': 'float64'})
file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_BERT']+file['ASR_DeepWordBug'])/4
# file = file.sample(frac=1)

print('Extrapolation test')
print('-'*100)
for dataset in datasets:
    print('*-'*50)
    print(f'Test on masking dataset {dataset}')
    train = file[file['Dataset']!=dataset].drop(columns='Dataset')
    train.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug'],inplace=True)
    test = file[file['Dataset']==dataset].drop(columns='Dataset')
    test.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug'],inplace=True)
    
    # linear regression
    x = sm.add_constant(train.drop(columns='ASR'))
    y = train['ASR']
    model = sm.OLS(y, x).fit()
    # print(model.summary())

    # LGBMR
    x_train, x_val, y_train, y_val = train_test_split(train.drop(columns='ASR'), np.array(train['ASR']), test_size = 0.2, random_state = 0)
    x_test, y_test = test.drop(columns='ASR'), np.array(test['ASR'])
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
    print(f'Result when masking {dataset}')
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    x_test = x_test.assign(Groundtruth_ASR=y_test,Predicted_ASR=predicted_y)
    x_test.to_csv(f'result_extrapolate_summary_{dataset}.csv')
    print('-'*100)
raise
print('Interpolation test')
file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_BERT','ASR_DeepWordBug','Dataset'],inplace=True)

x_train, x_val, y_train, y_val = train_test_split(file.drop(columns='ASR'), np.array(file['ASR']), test_size = 0.4, random_state = 0)
linear_X_train, linear_y_train, linear_X_test, linear_y_test = x_train.to_numpy(), y_train, x_val.to_numpy(), y_val
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.2, random_state = 0)
X = x_train.to_numpy()

x = sm.add_constant(file.drop(columns='ASR'))
y = file['ASR']
model = sm.OLS(y, x).fit()
print(model.summary())
raise

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
    print('Interpolation test')
    linear_model = LinearRegression(fit_intercept=True).fit(linear_X_train,linear_y_train)
    linear_predict = linear_model.predict(linear_X_test)
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(linear_predict.shape[0]):
        summary['Predicted'].append(linear_predict[i])
        summary['Groundtruth'].append(linear_y_test[i])
    summary = pd.DataFrame(summary)
    print(summary)
    print('RMSE: ',mean_squared_error(linear_y_test, linear_predict,squared=False))
    print('R2: ',r2_score(linear_y_test, linear_predict))
    print('MAE: ',mean_absolute_error(linear_y_test, linear_predict))
    print('Explained_variance_score: ',explained_variance_score(linear_y_test, linear_predict))
    print('MAPE: ',mean_absolute_percentage_error(linear_y_test, linear_predict))
    print('-'*50)

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
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*50)
    x_test = x_test.assign(Groundtruth_ASR=y_test,Predicted_ASR=predicted_y)
    x_test.to_csv('result_summary_interpolate.csv')

    r = permutation_importance(final_model, x_val, y_val,
                                n_repeats=30,
                                random_state=0)
    print(r.importances_mean)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{file.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                 f" +/- {r.importances_std[i]:.3f}")

# Accumulated Local Effects (ALE)
discrete_fts = ['Number of unique tokens',
            'Minimum number of tokens', 'Maximum number of tokens', 'Number of cluster',
            'Number of classes']
continuous_fts = ['Average number of tokens', 'Mean distance', 'Fisher ratio', 
                  'CalHara Index', 'DaBou Index', 'Pearson Med', 'Kurtosis', 
                  'Misclassification rate']

pylab.rcParams['font.size'] = 17
img_by_img=True
plot=False
if img_by_img and plot:
    for i, ft in enumerate(discrete_fts):
        fig = plt.figure(figsize=(10,7))
        axis = fig.add_subplot()
        ale_eff = ale(
            X=x_val, model=final_model, feature=[ft], grid_size=50, 
            feature_type='discrete' if ft in discrete_fts else 'continuous',
            include_CI=False, fig=fig, ax=axis
        )
        if ft in ['Number of unique tokens','Maximum number of tokens']:
            xticks = axis.get_xticks()
            axis.set_xticks(xticks[::5]) # set new tick positions
            axis.tick_params(axis='x', rotation=30) # set tick rotation
            axis.margins(x=0) # set tight margins
        print(f'{ft} :')
        eff = list(ale_eff['eff'])
        print(max(eff)-min(eff))
        fig.tight_layout()
        fig.savefig(f'image/interprete/{ft}.png')
        fig.show()
        plt.show()

    for i, ft in enumerate(continuous_fts):
        fig = plt.figure(figsize=(10,7))
        axis = fig.add_subplot()
        ale_eff = ale(
            X=x_val, model=final_model, feature=[ft], grid_size=50, 
            feature_type='discrete' if ft in discrete_fts else 'continuous',
            include_CI=False, fig=fig, ax=axis
        )
        print(f'{ft} :')
        eff = list(ale_eff['eff'])
        print(max(eff)-min(eff))
        fig.tight_layout()
        fig.savefig(f'image/interprete/{ft}.png')
        fig.show()
        plt.show()

elif plot:
    fig = plt.figure(figsize=(15,8))
    n_col = 3
    n_row = (len(discrete_fts)-1)//n_col+1
    axes = [fig.add_subplot(n_row,n_col,row*n_col+col+1) for row in range(n_row) for col in range(n_col)]
    for i, ft in enumerate(discrete_fts):
        ale_eff = ale(
            X=x_val, model=final_model, feature=[ft], grid_size=50, 
            feature_type='discrete' if ft in discrete_fts else 'continuous',
            include_CI=False, fig=fig, ax=axes[i]
        )
        if ft in ['Number of unique tokens','Maximum number of tokens']:
            xticks = axes[i].get_xticks()
            axes[i].set_xticks(xticks[::2]) # set new tick positions
            axes[i].tick_params(axis='x', rotation=30) # set tick rotation
            axes[i].margins(x=0) # set tight margins
    fig.tight_layout()
    fig.show()
    plt.show()


    fig = plt.figure(figsize=(15,10))
    n_col = 3
    n_row = (len(continuous_fts)-1)//n_col+1
    axes = [fig.add_subplot(n_row,n_col,row*n_col+col+1) for row in range(n_row) for col in range(n_col)]
    for i, ft in enumerate(continuous_fts):
        ale_eff = ale(
            X=x_val, model=final_model, feature=[ft], grid_size=50, 
            feature_type='discrete' if ft in discrete_fts else 'continuous',
            include_CI=False, fig=fig, ax=axes[i]
        )
    fig.tight_layout()
    fig.show()
    plt.show()


# fig = plt.figure(figsize=(15,10))
# db_ctns_fts = [(ft1,ft2) for ft1, ft2 in list(itertools.product(continuous_fts,continuous_fts)) if ft1!=ft2][:12]
# n_col = 3
# n_row = (len(db_ctns_fts)-1)//n_col+1
# axes = [fig.add_subplot(n_row,n_col,row*n_col+col+1) for row in range(n_row) for col in range(n_col)]
# for i, (ft1, ft2) in enumerate(db_ctns_fts):
#     if ft1 == ft2:
#         continue
#     ale_eff = ale(
#         X=x_test, model=final_model, feature=[ft1,ft2], grid_size=50, 
#         feature_type='discrete',
#         include_CI=False, fig=fig, ax=axes[i]
#     )
# fig.tight_layout()
# fig.show()
# plt.show()

# x_test=(x_test-x_test.min())/(x_test.max()-x_test.min())
# db_ctns_fts = [(ft1,ft2) for ft1, ft2 in list(itertools.product(continuous_fts,continuous_fts)) if ft1!=ft2][:12]
# for i, (ft1, ft2) in enumerate(db_ctns_fts):
#     if ft1 == ft2:
#         continue
#     fig = plt.figure(figsize=(15,10))
#     axis = fig.add_subplot()
#     ale_eff = ale(
#         X=x_test, model=final_model, feature=[ft1,ft2], grid_size=50, 
#         feature_type='discrete',
#         include_CI=False, fig=fig, ax=axis
#     )
#     fig.tight_layout()
#     fig.show()
#     plt.show()
print('Finish')