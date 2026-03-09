import mlflow
import os

# At the beginning of your Python script
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
databricks_host = os.environ.get("DATABRICKS_HOST")
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

if experiment_id is None or databricks_host is None or mlflow_tracking_uri is None:
    raise Exception("Environment variables are not configured correctly.")

@mlflow.trace
def hello_mlflow(message: str):

    hello_data = {
        "experiment_url": f"{databricks_host}/mlflow/experiments/{experiment_id}",
        "experiment_name": mlflow.get_experiment(experiment_id=experiment_id).name,
        "message": message,
    }
    return hello_data

result = hello_mlflow("hello, world!")
print(result)