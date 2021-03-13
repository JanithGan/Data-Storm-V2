import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from localsetup.src.util.DataUtil import DataUtil

# Get data
train_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-train.csv')
val_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-validation.csv')
test_data = pd.read_csv('/Users/harindujayarathne/PycharmProjects/Data-Storm-V2/Data/Hotel-A-test.csv')

# Choose from Robust scaler or MinMax scaler
scaler = MinMaxScaler()

# Choose Classifier Model from 'Random Forest' , 'XG Boost' and the number of estimators
classifierModel = 'Random Forest'
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
data1.model_train(classifierModel, no_of_estimators)
data1.predict()
