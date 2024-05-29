# models/lgbm_model.py

import lightgbm as lgb
from .base_model import BaseModel
from utils.mlflow_logger import log_to_mlflow

class LGBMModel(BaseModel):
    def load_data(self, data_path):
        # データをロードするコード
        pass

    def preprocess_data(self):
        # データを前処理するコード
        pass

    def train(self):
        train_data = lgb.Dataset(self.train_data, label=self.train_labels)
        self.model = lgb.train(self.params, train_data)
        log_to_mlflow(self.model, self.params, "LightGBM")

    def evaluate(self):
        # モデルを評価するコード
        pass

    def log_to_mlflow(self):
        # MLflowにログを取るコード
        pass
