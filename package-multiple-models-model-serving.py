# Databricks notebook source
# MAGIC %md
# MAGIC ## Host multiple models on the same Model Serving endpoint
# MAGIC This notebook walks through how to set up a model using the Pyfunc flavor that packages your separate models into a singe model for model serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets

# Import mlflow
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
diabetes_Y = np.array([diabetes_y]).transpose()
d = np.concatenate((diabetes_X, diabetes_Y), axis=1)
cols = diabetes.feature_names + ['progression']
diabetes_data = pd.DataFrame(d, columns=cols)
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(diabetes_data)

# The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
train_x = train.drop(["progression"], axis=1)
test_x = test.drop(["progression"], axis=1)
train_y = train[["progression"]]
test_y = test[["progression"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log and Register Multiple Models
# MAGIC Note that the lifecycle of these models does *not* need to be one-to-one with the packaged model. They can be trained, versioned, etc. separately, though any time any of them are moved into a Production stage, the multi-model Pyfunc must be repackaged. We recommend setting up automation for this (through Databricks Model Registry Webhooks or other automation mechanisms)

# COMMAND ----------

models = []
mlflow.sklearn.autolog(log_input_examples=True)
n_models = 4
for i in range(n_models):
    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=0.05, l1_ratio=0.05, random_state=42)
        model = lr.fit(train_x, train_y * 0 + i)
        mv = mlflow.register_model(f'runs:/{run.info.run_id}/model', f'multimodel-serving-{i}')
        client = MlflowClient()
        client.transition_model_version_stage(f'multimodel-serving-{i}', mv.version, "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Package as a Multi-Model Pyfunc Model

# COMMAND ----------

# even if you're writing Pyfunc code in a notebook, note that notebook state is *not* copied into the model's context automatically.
# as demonstrated below, state must be passed in explicitly through artifacts and referenced via the context object.
class MultiModelPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.models = []
        self.n_models = 4
        for i in range(self.n_models):
            self.models.append(mlflow.sklearn.load_model(context.artifacts[f'model-{i}']))
    
    def select_model(self, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Sample model requires Dataframe inputs")
        locale = model_input["locale"].iloc[0]
        if locale == "westus":
            return 0
        elif locale == "centralus":
            return 1
        elif locale == "eastus":
            return 2
        elif locale == "westeurope":
            return 3
        else:
            raise ValueError("Locale field incorrectly specified")
            
    def process_input(self, model_input):
        return model_input.drop("locale", axis=1).values.reshape(1, -1)

    def predict(self, context, raw_input):
        selected_model = self.select_model(raw_input)
        print(f'Selected model {selected_model}')
        model = self.models[selected_model]
        model_input = self.process_input(raw_input)
        return model.predict(model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC As part of packaging the multi-model Pyfunc, we must download the MLflow models associated with the sub-models

# COMMAND ----------

n_models = 4
paths = []
for i in range(n_models):
    paths.append(mlflow.artifacts.download_artifacts(f'models:/multimodel-serving-{i}/Production'))
artifacts = {f'model-{i}': paths[i] for i in range(n_models)}

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we prepare a simple input example to log with the model

# COMMAND ----------

input_example = test_x.iloc[0]
input_example["locale"] = "westus"
input_example = input_example.to_frame().transpose()

# COMMAND ----------

client = MlflowClient()
with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
      "raw-model",
      python_model=MultiModelPyfunc(),
      input_example=input_example,
      artifacts=artifacts,
    )
    model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/raw-model')
    prediction = model.predict(input_example)
    signature = infer_signature(input_example, prediction)
    mlflow.pyfunc.log_model(
        "augmented-model",
        python_model=MultiModelPyfunc(),
        artifacts=artifacts,
        input_example=input_example,
        signature=signature
    )
    mv = mlflow.register_model(f'runs:/{run.info.run_id}/augmented-model', "multimodel-serving")
    client.transition_model_version_stage(f'multimodel-serving', mv.version, "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Query the Multi-Model Model

# COMMAND ----------

model_uri = 'models:/multimodel-serving/Production'
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we load the input example

# COMMAND ----------

path = mlflow.artifacts.download_artifacts('models:/multimodel-serving/Production')
input_example = model.metadata.load_input_example(path)
input_example

# COMMAND ----------

model.predict(input_example)

# COMMAND ----------

input_example["locale"] = "centralus"
model.predict(input_example)
