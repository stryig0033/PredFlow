# utils/bayesian_optimization.py

from bayes_opt import BayesianOptimization

def bayesian_optimization(model_class, data_path, param_bounds, init_points=5, n_iter=25):
    def model_eval(**params):
        model = model_class(params)
        model.load_data(data_path)
        model.preprocess_data()
        model.train()
        return model.evaluate()

    optimizer = BayesianOptimization(
        f=model_eval,
        pbounds=param_bounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    return optimizer.max
