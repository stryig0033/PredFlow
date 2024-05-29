# models/xgboost_model.py

import xgboost as xgb
from .base_model import BaseModel
from utils.mlflow_logger import log_to_mlflow

class XGBoostModel(BaseModel):
    def load_data(self, data_path):
        # データをロードするコード
        pass

    def preprocess_data(self):
        # データを前処理するコード
        pass

    def train(self):
        dtrain = xgb.DMatrix(self.train_data, label=self.train_labels)
        self.model = xgb.train(self.params, dtrain)
        log_to_mlflow(self.model, self.params, "XGBoost")

    def evaluate(self):
        # モデルを評価するコード
        pass

    def log_to_mlflow(self):
        # MLflowにログを取るコード
        pass
