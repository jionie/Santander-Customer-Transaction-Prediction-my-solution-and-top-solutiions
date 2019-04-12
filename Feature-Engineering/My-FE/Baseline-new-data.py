import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.model_selection import StratifiedKFold,KFold
import warnings
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
from scipy.stats import norm, skew, rankdata
from sklearn.preprocessing import LabelEncoder

import os
print(os.listdir("../input"))

os.environ['NEPTUNE_API_TOKEN'] = "your api key"

import neptune
import gc

neptune.init(project_qualified_name='your project')

random_state = 42
np.random.seed(random_state)


#Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

private_lb = pd.read_csv("../input/Private_LB.csv")
public_lb = pd.read_csv("../input/Public_LB.csv")
synthetic = pd.read_csv("../input/synthetic_samples_indexes.csv")

private_lb = private_lb.rename(index=str, columns={"Private_LB": "index"})
public_lb = public_lb.rename(index=str, columns={"Public_LB": "index"})
synthetic = synthetic.rename(index=str, columns={"synthetic_samples_indexes": "index"})

true = public_lb.append(private_lb, ignore_index=True)


#train, test = reverse(train, test)

test_true = test.iloc[true["index"], :]
test_synthetic = test.iloc[synthetic["index"], :]

del private_lb, public_lb, synthetic, true
gc.collect()


feature = [c for c in train.columns if c not in ['ID_code', 'target']]


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


important = [c for c in train.columns if c not in ['ID_code', 'target']]

test_true = processing(test_true, important, train, test_synthetic, True)
test_synthetic = processing(test_synthetic, important, train, test_true, True)
test = pd.concat([test_true, test_synthetic], axis=0).sort_index()

test['id'] = test['ID_code'].apply(lambda x : int(x[5:]))
test = test.sort_values(by='id').reset_index(drop=True)

#del test_synthetic
gc.collect()


test = test.drop('id', axis=1)
target = train['target']
test = test.drop("ID_code", axis=1)

NAME = 'baseline'

param = {
    'bagging_freq':1,
    'bagging_fraction': 0.8,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.8,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 8,
    'num_threads': 8,
    #"lambda_l1" : 18.1,
    #"lambda_l2" : 7.696,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'device': 'gpu',
    'gpu_platform_id': 1,
    'gpu_device_id': 1,
    'verbosity': -1
}

train_params = {
    'num_boosting_rounds': 1000000,
    'early_stopping_rounds' : 4000,
    'verbose_eval': 5000
}

params = {**param, **train_params}


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

with neptune.create_experiment(name=NAME,
                               params=params):
    
    num_folds = 7

    folds = KFold(n_splits=num_folds, shuffle=True, random_state=323)
    oof = np.zeros(len(train))
    getVal = np.zeros(len(train))
    predictions = np.zeros(len(target))
    feature_importance_df = pd.DataFrame()

    print('Light GBM Model')
    
    ori_features = [c for c in train.columns if c not in ['target']]
    
    def resort_train(df):
        df['id'] = df['ID_code'].apply(lambda x : int(x[6:]))
        df = df.sort_values(by='id').reset_index(drop=True)
        df = df.drop(['ID_code', 'id'], axis=1)
        
        return df

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

        X_train, y_train = train.iloc[trn_idx][ori_features], train.iloc[trn_idx][['ID_code', 'target']]
        X_valid, y_valid = train.iloc[val_idx][ori_features], train.iloc[val_idx][['ID_code', 'target']]
        
        
        X_train = processing(X_train, important, test_true, X_valid, False)
        X_valid = processing(X_valid, important, test_true, X_train, False)
        
        X_train = resort_train(X_train)
        y_train = resort_train(y_train)
        X_valid = resort_train(X_valid)
        y_valid = resort_train(y_valid)
        
        features = [c for c in X_train.columns if c not in ['ID_code', 'target']]

        print("Fold idx:{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        
        monitor = neptune_monitor(prefix='fold{}_'.format(fold_))
        
        clf = lgb.train(param, trn_data, train_params['num_boosting_rounds'], \
                        valid_sets = [trn_data, val_data], verbose_eval = train_params['verbose_eval'], \
                        early_stopping_rounds=train_params['early_stopping_rounds'], \
                        callbacks=[monitor])
        
        oof[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
        getVal[val_idx]+= clf.predict(X_valid, num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    neptune.send_metric('roc_auc', roc_auc_score(target, oof))


num_sub = 7
print('Saving the Submission File')
test = pd.read_csv('../input/test.csv')
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission_new_new_{}.csv'.format(num_sub), index=False)
getValue = pd.DataFrame(getVal)
getValue.to_csv("Validation_kfold.csv")
