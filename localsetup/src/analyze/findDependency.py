# This code tries to calculate the correlation between categorical features with final outcome
# This is done using chi square contingency test
import pandas as pd
from scipy.stats import chi2_contingency

TrainData = pd.read_csv("/Data/Hotel-A-train.csv")
Cols = TrainData.columns
det_cols = Cols.to_list()
det_cols.remove('Reservation-id')
det_cols.remove('Reservation_Status')
det_cols.remove('Age')
det_cols.remove('Adults')
det_cols.remove('Children')
det_cols.remove('Babies')
det_cols.remove('Discount_Rate')
det_cols.remove('Room_Rate')
det_cols.remove('Expected_checkin')
det_cols.remove('Expected_checkout')
det_cols.remove('Booking_date')


def cal_p_value(column_):
    data_crosstab = pd.crosstab(TrainData[column_], TrainData[Cols[20]], margins=False)
    print(data_crosstab)
    array_cross = data_crosstab.to_numpy()
    output = chi2_contingency(array_cross)
    p_value = output[1] * 100
    print("Column: ", column_, " P-Value = ", p_value, "  Chi Square Statistic = ", output[0], "\n")
    print("Data values if zero correlation existed\n")
    print(output[3], "\n\n\n")


print("\n\nLower the p-value higher the correlation between the feature and the outcome\nA P-Value below 10 means a "
      "good "
      , "correlation exists\n\n\n")
for item in det_cols:
    cal_p_value(item)