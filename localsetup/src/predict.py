import pandas as pd
from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Choose from Robust scaler or MinMax scaler
scaler = MinMaxScaler()

# Choose Classifier Model from 'Random Forest' , 'XG Boost' and the number of estimators
classifierModel = 'Random Forest'
no_of_estimators = 15

# Imports data from csv files
train_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-train.csv')
validation_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-validation.csv')
test_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-test.csv')

# Useful columns listed - categorical, numerical and columns to remove
cat_data = ['Gender', 'Ethnicity', 'Educational_Level', 'Income', 'Country_region', 'Hotel_Type', 'Meal_Type',
            'Visted_Previously', 'Previous_Cancellations', 'Deposit_type', 'Booking_channel', 'Required_Car_Parking',
            'Use_Promotion']
num_data = ['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate']
drop_data = ['Reservation-id', 'Expected_checkin', 'Expected_checkout', 'Booking_date']


# Converts datetime information into meaningful columns
def conv_date(data_set):
    booking_date = pd.to_datetime(data_set['Booking_date'])
    expected_checkin = pd.to_datetime(data_set['Expected_checkin'])
    expected_checkout = pd.to_datetime(data_set['Expected_checkout'])

    booking_delta = pd.DataFrame((expected_checkin - booking_date).dt.days, columns=['booking_delta'])
    stay_delta = pd.DataFrame((expected_checkout - expected_checkin).dt.days, columns=['stay_delta'])
    return pd.concat([data_set, booking_delta, stay_delta], axis=1)


# Formats data to create x and y arrays for training the model
def format_data(data_set, is_test=False):
    f_data_set = pd.get_dummies(data_set, prefix=cat_data, columns=cat_data)
    f_data_set = f_data_set.drop(drop_data, axis=1)
    if is_test:
        x = f_data_set
        return x
    else:
        x = f_data_set.drop('Reservation_Status', axis=1)
        y = f_data_set['Reservation_Status']
        return x, y


# Trains a model
def model_train(model, xtrain, ytrain, xval, yval, estimators):
    train_results = []
    if model == 'Random Forest':
        _classifier = RandomForestClassifier(n_estimators=estimators)
        _classifier.fit(xtrain, ytrain)
        score = _classifier.score(xval, yval)
        train_results.append(_classifier)
        train_results.append(score)
        return train_results

    elif model == 'XG Boost':
        _classifier = XGBClassifier(n_estimators=estimators)
        _classifier.fit(np.array(xtrain), np.array(ytrain))
        score = _classifier.score(np.array(xval), np.array(yval))
        train_results.append(_classifier)
        train_results.append(score)
        return train_results


# Regenerate all data sets to our format for datetime information
train_data = conv_date(train_data)
validation_data = conv_date(validation_data)
test_data = conv_date(test_data)

# Remove drop columns from numerical and categorical column lists
for element in drop_data:
    if element in cat_data:
        cat_data.remove(element)
    if element in num_data:
        num_data.remove(element)

# Formatting the data sets to create training, validation and testing sets
x_train, y_train = format_data(train_data)
x_val, y_val = format_data(validation_data)
x_test = format_data(test_data, is_test=True)

# Normalizing data
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# Up-sampling data using SMOTE-NC
cat_indx = range(8, len(x_train.columns))
sm = SMOTENC(categorical_features=cat_indx, random_state=0)
x_train_res, y_train_res = sm.fit_resample(np.array(x_train), np.array(y_train))

# Using standard machine learning models to train a model

# With up-sampling
results = model_train(classifierModel, x_train_res, y_train_res, x_val, y_val, no_of_estimators)

# Without upsampling
# results = model_train(classifierModel, x_train, y_train, x_val, y_val, no_of_estimators)

# Calculate F1 score
classifier = results[0]
y_pred = classifier.predict(np.array(x_val))
F1 = f1_score(y_val, y_pred, average='macro')
accuracy = results[1]

# Display output
print("F1 score :", F1)
print("Accuracy :", accuracy)
