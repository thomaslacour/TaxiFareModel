# -- import{{{
import pandas as pd
import numpy as np
# sklearn import
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# personal imports
# from encoders import DistanceTransformer
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

import joblib

# }}}

class data:
    url = "s3://wagon-public-datasets/taxi-fare-train.csv"


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true)**2).mean())


class Trainer():


    def __init__(self, X, y):# {{{
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.EXPERIMENT_NAME = "[FR] [Bordeaux] [frodon] TaxiFareModel 1.0"
    # }}}


    # == mlflow == {{{
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    # }}}

    def set_pipeline(self):# {{{
        """defines the pipeline as a class attribute"""

        # -- create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        # == mlflow ==
        self.mlflow_log_param("estimator", "LinearRegression")

        self.pipeline = pipe

        # }}}

    def run(self):# {{{
        """ set and train the pipeline """

        # set the pipeline
        self.set_pipeline()

        # train the pipeline
        self.pipeline.fit(self.X, self.y)# }}}

        # == mlflow
        experiment_id = model.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    def evaluate(self, X_test, y_test):# {{{
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)# }}}

        # == mlflow ==
        self.mlflow_log_metric("rmse", rmse)

        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        return True


if __name__ == "__main__":# {{{
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    target="fare_amount"
    y = df[target]
    X = df.drop(target, axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    model=Trainer(X_train, y_train)
    model.run()
    # evaluate
    rmse=model.evaluate(X_val, y_val)
    print(rmse)
    # save model
    model.save_model()
# }}}
