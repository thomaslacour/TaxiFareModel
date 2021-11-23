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
        self.y = y# }}}

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

        self.pipeline = pipe

        # }}}


    def run(self):# {{{
        """ set and train the pipeline """

        # set the pipeline
        self.set_pipeline()

        # train the pipeline
        self.pipeline.fit(self.X, self.y)# }}}

    def evaluate(self, X_test, y_test):# {{{
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)# }}}

if __name__ == "__main__":
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
