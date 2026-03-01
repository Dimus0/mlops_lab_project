import hydra
import optuna
import mlflow
from omegaconf import DictConfig
import mlflow.sklearn
import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

def prepared_data(cfg: DictConfig):
    df = pd.read_csv(cfg.data.processed_path)
    X = df.drop(cfg.data.target_column, axis=1)
    y = df[cfg.data.target_column]
    return train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state)


def build_model(cfg: DictConfig, trial: optuna.Trial) -> RandomForestClassifier:
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", cfg.hpo.random_forest.n_estimators.low, cfg.hpo.random_forest.n_estimators.high),

        "max_depth": trial.suggest_int(
            "max_depth", cfg.hpo.random_forest.max_depth.low, cfg.hpo.random_forest.max_depth.high),

        "min_samples_split": trial.suggest_int(
            "min_samples_split", cfg.hpo.random_forest.min_samples_split.low, cfg.hpo.random_forest.min_samples_split.high),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", cfg.hpo.random_forest.min_samples_leaf.low, cfg.hpo.random_forest.min_samples_leaf.high),
    }

    model = RandomForestClassifier(**params)

    return model,params

def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    with mlflow.start_run(nested=True):
        X_train, X_val, y_train, y_val = prepared_data(cfg)

        model,params = build_model(cfg, trial)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        score = f1_score(y_val, preds)

        mlflow.log_params(params)
        mlflow.log_metric(cfg.hpo.metric,score)

        mlflow.set_tags({
            "author": "Puhachevskyi Dmytro",
            "dataset": "Telco Customer Churn",
            "model": cfg.model.type,
            "hpo_method": "Optuna",
            "sampler": cfg.hpo.sampler,
            "random_seed": cfg.data.random_state
        })

        return score
    
@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    if cfg.hpo.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.data.random_state)
    elif cfg.hpo.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.data.random_state)
    else:
        raise ValueError(f"Unsupported sampler: {cfg.hpo.sampler}")

    with mlflow.start_run(run_name="Optuna_HPO_Run"):

        study = optuna.create_study(
            direction=cfg.hpo.direction, 
            sampler=sampler
        )

        study.optimize(
            lambda trial: objective(trial, cfg), 
            n_trials=cfg.hpo.n_trials
        )

        best_score = study.best_value
        last_score = study.trials[-1].value

        mlflow.log_metric("best_score", best_score)
        mlflow.log_metric("last_score", last_score)
        mlflow.log_dict(study.best_params, "best_params.json")
        mlflow.log_text(str(cfg), "config.yaml")

        X_train, X_val, y_train, y_val = prepared_data(cfg)

        model = RandomForestClassifier(
            **study.best_params, 
            random_state=cfg.data.random_state
        )

        model.fit(X_train, y_train)

        model_path = os.path.join("models", "best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=(
                    cfg.mlflow.best_model_name if cfg.mlflow.register_model else None
                )
            )

if __name__ == "__main__":
    main()