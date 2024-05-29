# models/base_model.py

class BaseModel:
    def __init__(self, params):
        self.params = params

    def load_data(self, data_path):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def log_to_mlflow(self):
        raise NotImplementedError
