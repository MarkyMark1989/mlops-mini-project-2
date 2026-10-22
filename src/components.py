"""Azure ML pipeline components built with mldesigner.

Each component wraps a job function from this folder. Azure ML uploads src/
as the code snapshot and runs it in the registered environment.
"""
from pathlib import Path

from mldesigner import Input, Output, command_component

from data_prep import prepare_data
from train import parse_max_features, train_model

ENVIRONMENT_NAME = "telecom-churn-env"
ENVIRONMENT_VERSION = "2"  # bump when pipeline/conda.yaml changes
ENVIRONMENT = f"azureml:{ENVIRONMENT_NAME}:{ENVIRONMENT_VERSION}"
CODE = Path(__file__).parent


@command_component(
    name="telecom_churn_data_prep",
    display_name="Data preparation",
    description="Clean the churn data, add derived features and split train/test.",
    environment=ENVIRONMENT,
    code=CODE,
)
def data_prep_component(
    input_data: Input(type="uri_file"),
    prepared_data: Output(type="uri_folder"),
    test_size: float = 0.2,
    random_state: int = 42,
):
    prepare_data(input_data, prepared_data, test_size, random_state)


@command_component(
    name="telecom_churn_train",
    display_name="Model training and evaluation",
    description="Train (optionally tune) a random forest, evaluate on the test set and register it.",
    # Registering a model is a side effect, so never reuse a cached run of this step.
    is_deterministic=False,
    environment=ENVIRONMENT,
    code=CODE,
)
def train_component(
    prepared_data: Input(type="uri_folder"),
    model_output: Output(type="uri_folder"),
    n_estimators: int = 100,
    max_depth: int = 0,  # 0 means unlimited depth
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str = "none",
    tune_iterations: int = 0,
    random_state: int = 42,
    registered_model_name: str = "telecom-churn-rf",
    feature_set: str = "baseline",  # a key of features.FEATURE_SETS
    cv_seed: int = 0,
):
    train_model(
        prepared_data,
        model_output,
        n_estimators=n_estimators,
        max_depth=max_depth or None,
        min_samples_leaf=min_samples_leaf,
        max_features=parse_max_features(max_features),
        class_weight=None if class_weight == "none" else class_weight,
        tune_iterations=tune_iterations,
        random_state=random_state,
        registered_model_name=registered_model_name or None,
        feature_set=feature_set,
        cv_seed=cv_seed,
    )
