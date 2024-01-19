import pandas as pd
import xgboost as xgb
import warnings
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

#load data
sample = pd.read_csv('../data/sample_submission.csv')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#drop some vars
drop_list = ['PassengerId', 'Name', 'Cabin']
train = train.drop(drop_list, axis=1)
train = train.dropna(how='any')

onehot_list = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Transported']

# onehot-encoderのインスタンス化
enc = OneHotEncoder(sparse=False)

for column in onehot_list:
    # OneHotエンコーディングを適用
    transformed = enc.fit_transform(train[[column]])
    
    # エンコーディングされたデータをDataFrameに変換
    transformed_df = pd.DataFrame(transformed, columns=[f"{column}_{cat}" for cat in enc.categories_[0]], index=train.index)  # インデックスを指定
    
    # 元のデータから対象の列を削除
    train = train.drop(column, axis=1)
    
    # エンコーディングされたデータを元のDataFrameに結合
    train = pd.concat([train, transformed_df], axis=1)

train = train.drop(['HomePlanet_Mars', 
                    'Destination_TRAPPIST-1e', 
                    'CryoSleep_False', 
                    'VIP_False', 
                    'Transported_False'], axis=1)

#説明変数と被説明変数に分割
x = train.drop(['Transported_True'], axis=1)
y = train['Transported_True']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

train_all = pd.concat([y_train, X_train], axis=1)


#cross varidation
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'max_depth':3, 'eta':0.1}
cross_val = xgb.cv(
    params, dtrain, num_boost_round=1000, early_stopping_rounds=50
)

best_n_boost_round = cross_val.shape[0]


#ベイズ最適化による最適ハイパーパラメータ自動詮索

#pre setting of categorical parameters
try_grow_policy = 'depthwise'
try_objective = 'reg:squarederror'
try_booster = 'gbtree'
try_tree_method = 'auto'
try_sampling_method = 'uniform'
try_importance_type = 'gain'
try_device = 'cpu'
try_multi_strategy = 'diagonal'
try_eval_metric = 'rmse'

#evaluation function
def xgboost_eval(try_max_depth,
                 try_learning_rate, 
                 try_n_estimators, 
                 try_gamma, 
                 try_min_child_weight, 
                 try_subsample, 
                 try_colsample_bytree, 
                 try_reg_alpha, 
                 try_reg_lambda):
    # convert to int since these are not continuous variables
    try_max_depth = int(try_max_depth)
    try_n_estimators = int(try_n_estimators)

    #parameter settings
    #最大値を設定する系はナシ
    #データセットの前処理に関わる変数もナシ。
    model = xgb.XGBClassifier(
        max_depth=try_max_depth,
        learning_rate=try_learning_rate,
        n_estimators=try_n_estimators,
        gamma=try_gamma,
        min_child_weight=try_min_child_weight,
        subsample=try_subsample,
        colsample_bytree=try_colsample_bytree,
        reg_alpha=try_reg_alpha,
        reg_lambda=try_reg_lambda,
        try_grow_policy = try_grow_policy,
        try_objctive = try_objective,
        try_booster = try_booster,
        try_tree_method = try_tree_method,
        try_importance_type = try_importance_type,
        try_device = try_device,
        try_multi_strategy = try_multi_strategy,
        try_eval_metric = try_eval_metric
    )
    
    
    # model training
    model.fit(X_train, y_train)
    # calculate model score
    score = model.score(X_test, y_test)
    #start logging (nested)
    with mlflow.start_run(run_name = 'XGBoost',
                          experiment_id= experiment,
                          nested = True):
        #logging settings
        mlflow.log_params({
            'max_depth': try_max_depth,
            'learning_rate': try_learning_rate,
            'n_estimators': try_n_estimators,
            'gamma': try_gamma,
            'min_child_weight': try_min_child_weight,
            'subsample': try_subsample,
            'colsample_bytree': try_colsample_bytree,
            'reg_alpha': try_reg_alpha,
            'reg_lambda': try_reg_lambda,
            'grow_policy': try_grow_policy,
            'objective': try_objective,
            'booster': try_booster,
            'tree_method': try_tree_method,
            'sampling_method': try_sampling_method,
            'importance_type': try_importance_type,
            'device': try_device,
            'multi_strategy': try_multi_strategy,
            'eval_metric': try_eval_metric,
            'score': score
            })
        mlflow.xgboost.log_model(model,'mdoel')
        
    return score

# set search bounds of each parameter
pbounds = {
    'try_max_depth': (3, 50),
    'try_learning_rate': (0.01, 0.5),
    'try_n_estimators': (100, 1000),
    'try_gamma': (0, 5),
    'try_min_child_weight': (1, 10),
    'try_subsample': (0.5, 1.0),
    'try_colsample_bytree': (0.5, 1.0),
    'try_reg_alpha': (0, 1),
    'try_reg_lambda': (0, 1)
}

#create an experiment
experiment = mlflow.create_experiment('spaceship_titanic_bayes_opt')

#start run experiment
with mlflow.start_run(run_name='XGboost',
                      experiment_id=experiment):
    
    #instansation of optimizer
    optimizer = BayesianOptimization(
        f=xgboost_eval,
        pbounds=pbounds,
        random_state=1
    )
    
    #calculation
    optimizer.maximize(init_points=5, n_iter=95)
        