import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.model_selection import StratifiedKFold,KFold, train_test_split
import warnings
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
from scipy.stats import norm, skew, rankdata
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization

import os
print(os.listdir("../input"))

os.environ['NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiI3NTNlNWFjZC0xNjY1LTQ2OTQtODNkMC1kMzk2N2UxMzRmODkifQ=="

import neptune
import gc

neptune.init(project_qualified_name='jionie/santander')

random_state = 42
np.random.seed(random_state)

#Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

private_lb = pd.read_csv("../input/Private_LB.csv")
public_lb = pd.read_csv("../input/Public_LB.csv")
synthetic = pd.read_csv("../input/synthetic_samples_indexes.csv")

private_lb = private_lb.rename(index=str, columns={"Private_LB": "index"})
public_lb = public_lb.rename(index=str, columns={"Public_LB": "index"})
synthetic = synthetic.rename(index=str, columns={"synthetic_samples_indexes": "index"})

true = public_lb.append(private_lb, ignore_index=True)

test_true = test.iloc[true["index"], :]
test_synthetic = test.iloc[synthetic["index"], :]

del private_lb, public_lb, synthetic, true
gc.collect()

feature = [c for c in train.columns if c not in ['ID_code', 'target']]
important = [c for c in train.columns if c not in ['ID_code', 'target']]


def processing(df, cols, add_0, add_1, no_add_1=True):
    
    total_cols = cols.copy()
    total_cols.append('ID_code')
    if (no_add_1):
        tmp = pd.concat([df[total_cols], add_0[total_cols]], axis=0)
    else:
        tmp = pd.concat([df[total_cols], add_0[total_cols], add_1[total_cols]], axis=0)
    size = tmp.shape[0]

    for var in cols:
        print(var)
        tmp[var+'_count'] = tmp.groupby([var])['ID_code'].transform('count')
        df[var+'_count'] = tmp.iloc[:df.shape[0]][var+'_count']
        max_ = df[var+'_count'].max()
        min_ = df[var+'_count'].min()
        
        df[var+'_count'] = df[var+'_count'].apply(lambda x:min((abs(x-max_)/(max_-min_)),\
                                                  (abs(x-min_)/(max_-min_))))
    return df

test_true = processing(test_true, important, train, test_synthetic, True)
test_synthetic = processing(test_synthetic, important, train, test_true, True)
test = pd.concat([test_true, test_synthetic], axis=0).sort_index()

test['id'] = test['ID_code'].apply(lambda x : int(x[5:]))
test = test.sort_values(by='id').reset_index(drop=True)

test = test.drop('id', axis=1)
target = train['target']
test = test.drop("ID_code", axis=1)

ori_features = [c for c in train.columns if c not in ['target']]
    
def resort_train(df):
    df['id'] = df['ID_code'].apply(lambda x : int(x[6:]))
    df = df.sort_values(by='id').reset_index(drop=True)
    df = df.drop(['ID_code', 'id'], axis=1)
    
    return df

#important = ['var_12', 'var_108', 'var_126']
X_train, X_valid, y_train, y_valid = train_test_split(train[ori_features], train[['ID_code', 'target']], \
                                                  test_size=0.2, random_state=42)

predictions = np.zeros(y_valid.shape[0])

X_train = processing(X_train, important, test_true, X_valid, False)
X_valid = processing(X_valid, important, test_true, X_train, False)

X_train = resort_train(X_train)
y_train = resort_train(y_train)
X_valid = resort_train(X_valid)
y_valid = resort_train(y_valid)

features = [c for c in X_train.columns if c not in ['ID_code', 'target']]

trn_data = lgb.Dataset(X_train[features], label=y_train)
val_data = lgb.Dataset(X_valid[features], label=y_valid)

def neptune_monitor(prefix):
    def callback(env):
        for name, loss_name, loss_value, _ in env.evaluation_result_list:
            channel_name = '{}{}_{}'.format(prefix, name, loss_name)
            neptune.send_metric(channel_name, x=env.iteration, y=loss_value)
    return callback


def plot_prediction_distribution(y_true, y_pred, ax):
    df = pd.DataFrame({'prediction': y_pred, 'ground_truth': y_true})
    
    sns.distplot(df[df['ground_truth'] == 0]['prediction'], label='negative', ax=ax)
    sns.distplot(df[df['ground_truth'] == 1]['prediction'], label='positive', ax=ax)

    ax.legend(prop={'size': 16}, title = 'Labels')



print('Light GBM Model')

def lgbcv(lambda_l1, lambda_l2): 

#def lgbcv(feature_fraction, silent=True, seed=1234):
    
    NAME = 'baseline'

    param = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "tree_learner": "serial",
        "boost_from_average": "false",
        'device': 'gpu',
        'gpu_platform_id': 1,
        'gpu_device_id': 1,
        "verbosity" : 1,
        "max_depth" : -1,


        "learning_rate" : 0.001,
        'min_split_gain': 0,
        "num_leaves" : 8,
        
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 10,

        "lambda_l1" : lambda_l1,
        "lambda_l2" : lambda_l2,

        "bagging_freq": 1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.8,
        
        #"drop_rate" : 0.4,
        
    }

    train_params = {
        'num_boosting_rounds': 5000,
        'early_stopping_rounds' : 1000,
        'verbose_eval': 5000
    }

    params = {**param, **train_params}

    with neptune.create_experiment(name=NAME,
                                   params=params):

        monitor = neptune_monitor(prefix='train')

        clf = lgb.train(param, trn_data, train_params['num_boosting_rounds'], \
                        valid_sets = [trn_data, val_data], verbose_eval = train_params['verbose_eval'], \
                        early_stopping_rounds=train_params['early_stopping_rounds'], callbacks=[monitor])

        predictions = clf.predict(X_valid, num_iteration=clf.best_iteration)

        print("CV score: {:<8.5f}".format(roc_auc_score(y_valid['target'], predictions)))
        neptune.send_metric('roc_auc', roc_auc_score(y_valid['target'], predictions))
        
        loss = roc_auc_score(y_valid['target'], predictions)
        
        return loss

param_list = list()
score_list = list()

lgbBO = BayesianOptimization(lgbcv, {'lambda_l1': (0, 50),'lambda_l2': (0, 50)})

#lgbBO = BayesianOptimization(lgbcv, {'feature_fraction': (0.011, 0.8)})


lgbBO.maximize(init_points = 10, n_iter = 25, xi=0.06)
print('-' * 53)
print('Final Results')
print(lgbBO.max)

