import yaml
import numpy
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.xgboost_model import XGBoostModel
from models.lgbm_model import LGBMModel
from utils.data_loader import load_data
from utils.bayesian_optimization import bayesian_optimization
import mlflow
import mlflow.sklearn

def run_model(model_class, params, data_path):
    model = model_class(params)
    model.load_data(data_path)
    model.preprocess_data()
    model.train()
    evaluation_metric = model.evaluate()

    # MLflowにログ
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("evaluation_metric", evaluation_metric)
        # モデルのログ保存（適宜変更）
        if model_class.__name__ == "XGBoostModel":
            mlflow.xgboost.log_model(model.model, "XGBoostModel")
        elif model_class.__name__ == "LGBMModel":
            mlflow.lightgbm.log_model(model.model, "LGBMModel")

    return model

def main():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    data_path = "hogehoge"

    # パラメータ境界を定義
    xgboost_param_bounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 300),
        'gamma': (0, 5),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }

    lgbm_param_bounds = {
        'num_leaves': (20, 150),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 300),
        'max_depth': (-1, 15),
        'min_split_gain': (0.0, 1.0),
        'min_child_weight': (1e-3, 1.0),
        'min_child_samples': (5, 100),
        'subsample': (0.5, 1.0),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.5, 1.0),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0),
    }

    # ベイズ最適化
    optimized_xgboost_params = bayesian_optimization(XGBoostModel, data_path, xgboost_param_bounds)
    optimized_lgbm_params = bayesian_optimization(LGBMModel, data_path, lgbm_param_bounds)

    # 最適化されたパラメータでモデルを実行
    model_classes = [
        (XGBoostModel, optimized_xgboost_params['params']),
        (LGBMModel, optimized_lgbm_params['params'])
    ]

    predictions_list = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_model, model_class, params, data_path) for model_class, params in model_classes]
        for future in as_completed(futures):
            try:
                model, predictions = future.result()
                predictions_list.append(predictions)
            except Exception as e:
                print(f"Model execution failed with exception: {e}")
    
    # アンサンブル予測
    ensemble_predictions = np.mean(predictions_list, axis=0)
    
    # アンサンブルモデルの評価
    # ここでは、仮にaccuracyを評価指標として使用
    true_labels = load_data(data_path)['label']  # 正しいラベルをロードする
    ensemble_accuracy = np.mean(ensemble_predictions == true_labels)

    # アンサンブルモデルの結果をMLflowにログ
    with mlflow.start_run():
        mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
        # 必要に応じて、その他のメトリックやパラメータをログ
        mlflow.log_param("ensemble_method", "average")

if __name__ == "__main__":
    main()
