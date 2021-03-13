import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from localsetup.src.util.DataUtil import DataUtil

# Get data
train_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-train.csv')
val_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-validation.csv')
test_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-test.csv')

# train_data = pd.concat([train_data, val_data], axis=0)

# Choose from Robust scaler or MinMax scaler
scaler = MinMaxScaler()

# Choose Classifier Model from 'Random Forest' , 'XG Boost' and the number of estimators
classifierModel = 'XG Boost'
no_of_estimators = 15

# Useful columns listed - categorical, numerical and columns to remove
cat_data = ['Gender', 'Ethnicity', 'Educational_Level', 'Income', 'Country_region', 'Hotel_Type', 'Meal_Type',
            'Visted_Previously', 'Previous_Cancellations', 'Deposit_type', 'Booking_channel', 'Required_Car_Parking',
            'Use_Promotion']
num_data = ['Age', 'Adults', 'Children', 'Babies', 'Discount_Rate', 'Room_Rate']
drop_data = ['Reservation-id', 'Expected_checkin', 'Expected_checkout', 'Booking_date']
columns = [drop_data, cat_data, num_data]

# Uses DataUtil Object to do the whole process
data1 = DataUtil(train_data, val_data, test_data, scaler, columns)
data1.get_normalized_data()
data1.up_sample()
# data1.combined_sample()
# data1.under_sample()

# x_up = data1.x_train_res
# y_up = data1.y_train_res
# pd.DataFrame(x_up).to_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/x_up.csv')
# pd.DataFrame(y_up).to_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/y_up.csv')

# data1.x_train_res = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/x_up.csv')
# data1.y_train_res = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/y_up.csv')

data1.estimators = 8
data1.model_train(classifierModel)
data1.predict()

y_test_mapped = pd.Series(data1.y_test).map({'Check-In': 1, 'Canceled': 2, 'No-Show': 3})
y_test_mapped.index = data1.test_data['Reservation-id']
y_test_mapped.to_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Submission2.csv'
                     , header=['Reservation_status'])

# for i in range(2, 20):
#     data1.estimators = i
#     data1.model_train(classifierModel)
#
#     print('\nno of estimators: ', i, '\n')
#     data1.predict()
