# utils/mlflow_logger.py

import mlflow

def log_to_mlflow(model, params, model_name):
    with mlflow.start_run():
        mlflow.log_params(params)
        # モデルのログ保存（適宜変更）
        mlflow.xgboost.log_model(model, model_name) if model_name == "XGBoost" else mlflow.lightgbm.log_model(model, model_name)
        # 追加のメトリクスやアーティファクトをログする場合
        # mlflow.log_metric("metric_name", value)
