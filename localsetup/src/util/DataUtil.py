import pandas as pd
from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


class DataUtil:

    def __init__(self, train, validation, test, scaler, columns):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.train_results, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test = [], [], [], [], [], []
        self.x_train_res, self.y_train_res = [], []
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.scaler = scaler
        self.drop_data = columns[0]
        self.cat_data = columns[1]
        self.num_data = columns[2]

    # Converts datetime information into meaningful columns
    @staticmethod
    def conv_date(data_set):

        booking_date = pd.to_datetime(data_set['Booking_date'])
        expected_checkin = pd.to_datetime(data_set['Expected_checkin'])
        expected_checkout = pd.to_datetime(data_set['Expected_checkout'])

        booking_delta = pd.DataFrame((expected_checkin - booking_date).dt.days, columns=['booking_delta'])
        stay_delta = pd.DataFrame((expected_checkout - expected_checkin).dt.days, columns=['stay_delta'])
        return pd.concat([data_set, booking_delta, stay_delta], axis=1)

    # Formats data to create x and y arrays for training the model
    def format_data(self, data_set, is_test=False):

        f_data_set = pd.get_dummies(data_set, prefix=self.cat_data, columns=self.cat_data)
        f_data_set = f_data_set.drop(self.drop_data, axis=1)
        if is_test:
            x = f_data_set
            return x
        else:
            x = f_data_set.drop('Reservation_Status', axis=1)
            y = f_data_set['Reservation_Status']
            return x, y

    # Get raw data -> manipulate -> normalize
    def get_normalized_data(self):

        # Regenerate all data sets to our format for datetime information
        self.train_data = self.conv_date(self.train_data)
        self.validation_data = self.conv_date(self.validation_data)
        self.test_data = self.conv_date(self.test_data)

        # Remove drop columns from numerical and categorical column lists
        for element in self.drop_data:
            if element in self.cat_data:
                self.cat_data.remove(element)
            if element in self.num_data:
                self.num_data.remove(element)

        # Formatting the data sets to create training, validation and testing sets
        self.x_train, self.y_train = self.format_data(self.train_data)
        self.x_val, self.y_val = self.format_data(self.validation_data)
        self.x_test = self.format_data(self.test_data, True)

        # Normalizing data
        self.x_train = pd.DataFrame(self.scaler.fit_transform(self.x_train), columns=self.x_train.columns)
        self.x_val = pd.DataFrame(self.scaler.transform(self.x_val), columns=self.x_val.columns)
        self.x_test = pd.DataFrame(self.scaler.transform(self.x_test), columns=self.x_test.columns)

    # Up samples data
    def up_sample(self):

        # Up-sampling data using SMOTE-NC
        cat_index = range(8, len(self.x_train.columns))
        sm = SMOTENC(categorical_features=cat_index, random_state=0)
        self.x_train_res, self.y_train_res = sm.fit_resample(np.array(self.x_train), np.array(self.y_train))

    # Trains a model
    def model_train(self, model, estimators):

        if model == 'Random Forest':
            _classifier = RandomForestClassifier(n_estimators=estimators)
            _classifier.fit(self.x_train_res, self.y_train_res)
            score = _classifier.score(self.x_val, self.y_val)
            self.train_results.append(_classifier)
            self.train_results.append(score)

        elif model == 'XG Boost':
            _classifier = XGBClassifier(n_estimators=estimators)
            _classifier.fit(np.array(self.x_train_res), np.array(self.y_train_res))
            score = _classifier.score(np.array(self.x_val), np.array(self.y_val))
            self.train_results.append(_classifier)
            self.train_results.append(score)

    # Predicts using model and calculates F1 score
    def predict(self):

        classifier = self.train_results[0]
        y_pred = classifier.predict(np.array(self.x_val))
        f1 = f1_score(self.y_val, y_pred, average='macro')
        accuracy = self.train_results[1]
        # Display output
        print("F1 score :", f1)
        print("Accuracy :", accuracy)
