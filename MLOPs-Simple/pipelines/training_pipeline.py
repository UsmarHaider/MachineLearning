from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path : str):
    ingest = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(ingest)
    model = train_model(x_train, x_test, y_train, y_test)
    r2, rmse = evaluate_model(model, x_test, y_test)

