# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:04:32 2019

@author: amit
"""

import pandas as pd
import numpy as np
import seaborn as sns
import re
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgbm
from scipy import stats
from datetime import timedelta
import seaborn as sns
import xgboost as xgb
## Pre Processing
original_data = pd.read_csv(
    "E:/Kaggle_Problem/Flight Ticket Prediction/Data_Train.csv")


test_data = pd.read_csv(
    "E:/Kaggle_Problem/Flight Ticket Prediction/Test_Set.csv")

holidays =  pd.read_csv(
    "E:/Kaggle_Problem/Flight Ticket Prediction/Holiday_Dates.csv")

#distance = pd.read_csv(
#    "E:/Kaggle_Problem/Flight Ticket Prediction/distance_data.csv")

holidays['Date'] = pd.to_datetime(holidays['Date'], format='%d-%m-%y')
test_data['Date_of_Journey'] = pd.to_datetime(test_data['Date_of_Journey'], format='%d-%m-%y')


original_data['Date_of_Journey'] = pd.to_datetime(
    original_data['Date_of_Journey'], format='%d-%m-%y')


def clean_data(dataframe):
#    dataframe['Arrival_Time'] = dataframe['Arrival_Time'].apply(
#            lambda x: re.search(r'\d{1,2}:\d{1,2}', x).group())

#    dataframe['Duration'] = dataframe['Duration'].apply(
#            lambda x: int(re.search(r'\d{1,2}', x).group()))
    dataframe['Dep_Time_2'] = dataframe['Dep_Time'].apply(
            lambda x: float(x.split(":")[0])+float(x.split(":")[1])/60)
    dataframe['Dep_Time_Hour'] = dataframe['Dep_Time'].apply(
            lambda x: int(x.split(":")[0]))
    dataframe['Arrival_Time'] = dataframe['Arrival_Time'].apply(
            lambda x: float(x.split(":")[0])+float(x.split(":")[1])/60)
    
    return dataframe



def basic_feature_engineering(dataframe):
    ##Date related
    dataframe['Month'] = dataframe['Date_of_Journey'].apply(
            lambda x: str(x.month))
    dataframe['Weekday'] = dataframe['Date_of_Journey'].apply(
            lambda x: str(datetime.date.weekday(x)))
    dataframe['Week'] = dataframe['Date_of_Journey'].apply(
            lambda x: int(x.week))
    dataframe['Day'] = dataframe['Date_of_Journey'].apply(
            lambda x: int(x.day))
			
	## Meal Bagage Related
    dataframe['meal_baggage_flag'] = dataframe['Additional_Info'].apply(
            lambda x: 1 if(('meal' in x.lower()) | ('baggage' in x.lower())) else 0)
			
	## Binning the days
#    dataframe['Week_2'] = pd.cut(dataframe['Day'], bins=[1,7,14,21,28,33], labels=['1', '2', '3', '4', '5'])
#    dataframe.drop('Day', inplace=True, axis=1)
#    
    ##Time related
#    dataframe['Dep_Time_Bin'] = pd.cut(dataframe['Dep_Time'], 6,
#             labels=['Midnight', 'Early_Morning', 'Morning', 'Afternoon','Evening', 'Night'])
    
#    dataframe['Arrival_Time_Bin'] = pd.cut(dataframe['Arrival_Time'], 6,
#             labels=['Midnight', 'Early_Morning', 'Morning', 'Afternoon','Evening', 'Night'])
    return dataframe

def to_longformat(dataframe, column_to_unpivot, column_as_id):
    """
    Function to convert the columns to long format from wide format assuming
    the column to unpivot is in the form a string of a list
    delimited by space Eg: [value1 value2]

    Parameters
    ----------
    column_to_unpivot: String
        The column of the variables to convert to long format
    column_as_id: List
        The list of columns to keep as Index while converting to long format

    Returns
    -------
    The dataframe converted into long format
    """
    dataframe[column_to_unpivot] = dataframe[column_to_unpivot].apply(
            lambda x: str(x).strip().split("[")[1][:-1])
   
    temp = dataframe[column_to_unpivot].str.split(" ", expand=True)

    dataframe = pd.concat([dataframe, temp], axis=1)
    dataframe = pd.melt(dataframe, id_vars=column_as_id,
                        value_vars=range(0, temp.shape[1]))
    dataframe.dropna(inplace=True)
    dataframe.drop('variable', axis=1, inplace=True)
    return dataframe


def advance_feature_engineering(train_data, test_data, holiday_data):
    ## Historical Rolling Average
    
#    train_data['Dep_Time'] = pd.to_datetime(train_data['Dep_Time'], format='%H:%M')
#    test_data['Dep_Time'] = pd.to_datetime(test_data['Dep_Time'], format='%H:%M')
#    
#    airline_route_date = train_data.groupby(
#            ['Airline', 'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'], as_index=False).agg({'Price': 'mean'})
#    airline_route_date.rename(columns={'Price': 'Average_Price'}, inplace=True)
#    airline_route_date.sort_values(by=['Airline',  'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'], inplace=True)
#    airline_route_date['lagged_average_price'] = airline_route_date.groupby(
#            ['Airline', 'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'], as_index=False)['Average_Price'].shift(1)
#    rolling_average = airline_route_date.groupby(
#            ['Airline', 'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'], as_index=False)['lagged_average_price'].expanding(1).mean()
#    rolling_average.reset_index(inplace=True, drop=True)
#    airline_route_date['Rolling_Average'] = rolling_average
#    airline_route_date.fillna(0, inplace=True)
#    train_data = pd.merge(train_data, airline_route_date[
#            ['Airline',  'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour', 'Rolling_Average']],
#    how='left', on=['Airline',  'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'])
#    test_data = pd.merge(test_data, airline_route_date[
#            ['Airline',  'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour', 'Rolling_Average']],
#    how='left', on=['Airline', 'Source','Destination', 'Date_of_Journey', 'Dep_Time_Hour'])


    ## Number of holidays in a 8 day period
    train_data['Min_Date'] = train_data['Date_of_Journey'].apply(
            lambda x: x-timedelta(4))
     # Creating the maximum date for the upper limit of the window
    train_data['Max_Date'] = train_data['Date_of_Journey'].apply(
            lambda x: x+timedelta(4))
     # Creating a list of range of dates from minimum to maximum date
    train_data['Min_Max_Date'] = train_data.apply(
            lambda x: str(np.arange(x['Min_Date'].date(), x['Max_Date'].date())), axis=1)
    long_formatted = to_longformat(
            train_data, 'Min_Max_Date',
            ['Date_of_Journey'])
    long_formatted.drop_duplicates(inplace=True)
    long_formatted.rename(columns={'value': 'date_range'}, inplace=True)
    long_formatted['date_range'] = long_formatted['date_range'].str.replace("'","").str.strip()
    long_formatted['date_range'] = pd.to_datetime(long_formatted['date_range'],
                  format='%Y-%m-%d')
    long_formatted = pd.merge(long_formatted, holiday_data, how='left',
                              left_on='date_range',
                              right_on='Date')
    holidays_1 = long_formatted.groupby('Date_of_Journey', as_index=False).agg(
            {'Flag': 'count'})
    train_data = pd.merge(train_data, holidays_1, on='Date_of_Journey', how='left')
    test_data = pd.merge(test_data, holidays_1, on='Date_of_Journey', how='left')
    train_data.drop(['Min_Date', 'Max_Date', 'Min_Max_Date'], inplace=True, axis=1)
    
    ## Historical Weekday month average
    airline_month_weekday = train_data.groupby(['Airline' ,'Month', 'Day']).agg(
            {'Price': 'mean'}).reset_index()
    airline_month_weekday.rename(columns={'Price': 'Month_Weekday_Average'}, inplace=True)
    train_data = pd.merge(train_data, airline_month_weekday, how='left',
                          on=['Airline', 'Month', 'Day'])
    test_data = pd.merge(test_data, airline_month_weekday, how='left',
                          on=['Airline', 'Month', 'Day'])

	
	## Historical Standard deviation of Duration to capture the frequency of flights
    std_data = train_data.groupby(['Airline', 'Week', 'Source', 'Destination']).agg({
            'Duration': lambda x: np.std(x)}).reset_index()
    std_data.rename(columns={'Duration': 'duration_std'}, inplace=True)
    
    train_data = pd.merge(train_data, std_data, how='left',
                          on=['Airline', 'Week', 'Source', 'Destination'])
    test_data = pd.merge(test_data, std_data, how='left',
                          on=['Airline', 'Week', 'Source', 'Destination'])
    
    
    ## Number of flights of airline at airline DOJ and DeP Time Level
    airline_month_day_hour = train_data.groupby(['Airline', 'Date_of_Journey', 'Dep_Time_Hour']).size().reset_index()
    airline_month_day_hour.rename(columns={0: 'Airline_Date_Number'}, inplace=True)
    train_data = pd.merge(train_data, airline_month_day_hour, how='left',
                          on=['Airline', 'Date_of_Journey', 'Dep_Time_Hour'])
    test_data = pd.merge(test_data, airline_month_day_hour, how='left',
                          on=['Airline', 'Date_of_Journey', 'Dep_Time_Hour'])
    
    ## Peak Hour
    peak_hour = train_data.groupby(['Date_of_Journey', 'Source', 'Destination']).agg(
            {'Dep_Time_Hour': lambda x: stats.mode(x)[0]}).reset_index()
    peak_hour.rename(columns={'Dep_Time_Hour': 'Peak_Dep_Hour'}, inplace=True)
    train_data = pd.merge(train_data, peak_hour, how='left',
                      on=['Date_of_Journey', 'Source', 'Destination'])
    test_data = pd.merge(test_data, peak_hour, how='left',
                          on=['Date_of_Journey', 'Source', 'Destination'])
    # Most Demanded Day
    most_demand = train_data.groupby(['Month', 'Source', 'Destination']).agg(
            {'Day': lambda x: stats.mode(x)[0]}).reset_index()
    most_demand.rename(columns={'Day': 'Peak_Demand_Day'}, inplace=True)
    train_data = pd.merge(train_data, most_demand, how='left',
                      on=['Month', 'Source', 'Destination'])
    test_data = pd.merge(test_data, most_demand, how='left',
                          on=['Month', 'Source', 'Destination'])
    
    # Most Demanded Source
    most_demand = train_data.groupby(['Date_of_Journey']).agg(
            {'Source': lambda x: stats.mode(x)[0]}).reset_index()
    most_demand.rename(columns={'Source': 'Peak_Demand_Source'}, inplace=True)
    train_data = pd.merge(train_data, most_demand, how='left',
                      on=['Date_of_Journey'])
    test_data = pd.merge(test_data, most_demand, how='left',
                          on=['Date_of_Journey'])
    
    ## Min _Max_Price at DOJ Source Destination Level
    most_demand = train_data.groupby([ 'Date_of_Journey', 'Source', 'Destination']).agg(
            {'Price': ['min', 'max']}).reset_index()
    most_demand.columns = ['Date_of_Journey', 'Source', 'Destination', 'Price_Min', 'Price_Max']
    train_data = pd.merge(train_data, most_demand, how='left',
                      on=['Date_of_Journey', 'Source', 'Destination'])
    test_data = pd.merge(test_data, most_demand, how='left',
                          on=['Date_of_Journey', 'Source', 'Destination'])
       
    

    ## Number of unique routes for a airline-source-destination
    route_data = train_data.groupby(['Airline', 'Source', 'Destination', 'Month']).agg({
            'Route': lambda x: len(x.unique())}).reset_index()
    route_data.rename(columns={'Route': 'Number_of_Routes'}, inplace=True)
    train_data =  pd.merge(train_data, route_data, how='left',
                           on=['Airline', 'Source', 'Destination', 'Month'])
    test_data =  pd.merge(test_data, route_data, how='left',
                           on=['Airline', 'Source', 'Destination', 'Month'])
    
    

    ## Number of unique departure times for airline
    departure_data = train_data.groupby(['Airline', 'Source', 'Destination', 'Week']).agg({
            'Dep_Time': lambda x: len(x.unique())}).reset_index()
    departure_data.rename(columns={'Dep_Time': 'Number_of_Dep_Times'}, inplace=True)
    train_data =  pd.merge(train_data, departure_data, how='left',
                           on=['Airline', 'Source', 'Destination', 'Week'])
    test_data =  pd.merge(test_data, departure_data, how='left',
                           on=['Airline', 'Source', 'Destination', 'Week'])
    
   
    ## Number of unique date of journey
    departure_data = train_data.groupby(['Airline', 'Source', 'Destination']).agg({
            'Date_of_Journey': lambda x: len(x.unique())}).reset_index()
    departure_data.rename(columns={'Date_of_Journey': 'Number_of_Dates'}, inplace=True)
    train_data =  pd.merge(train_data, departure_data, how='left',
                           on=['Airline', 'Source', 'Destination'])
    test_data =  pd.merge(test_data, departure_data, how='left',
                           on=['Airline', 'Source', 'Destination'])
    

    ## Number of flights
    number_of_flights = train_data.groupby(['Date_of_Journey', 'Source', 'Destination']).size().reset_index()
    number_of_flights.rename(columns={0: 'Number_of_Flights'}, inplace=True)
    train_data = pd.merge(train_data, number_of_flights, on=['Date_of_Journey', 'Source', 'Destination'],
                          how='left')
    test_data = pd.merge(test_data, number_of_flights, on=['Date_of_Journey', 'Source', 'Destination'],
                          how='left')
    
    
    ## Average Timedelta in minutes between flights
    timedelta_data = train_data[['Airline', 'Date_of_Journey', 'Source', 'Destination','Dep_Time']]
    timedelta_data['Dep_Time'] = pd.to_datetime(timedelta_data['Dep_Time'], format='%H:%M')
    timedelta_data.sort_values(by=['Airline', 'Source', 'Destination', 'Date_of_Journey','Dep_Time'], inplace=True)
    timedelta_data['lagged_time'] = timedelta_data.groupby(
            ['Airline', 'Source', 'Destination', 'Date_of_Journey'])['Dep_Time'].shift(1)
    timedelta_data['time_diff'] = timedelta_data.apply(
            lambda x: (x['Dep_Time']-x['lagged_time']).seconds/60, axis=1)
    timedelta_data = timedelta_data.groupby(['Airline','Source', 'Destination' ,'Date_of_Journey']).agg(
            {'time_diff': 'mean'}).reset_index()
    train_data = pd.merge(train_data, timedelta_data, how='left', 
                           on=['Airline','Source', 'Destination' ,'Date_of_Journey'])
    test_data = pd.merge(test_data, timedelta_data, how='left', 
                           on=['Airline','Source', 'Destination' ,'Date_of_Journey'])
    
    ## Number of Competitors Flight
    flight_count = train_data.groupby(
            ['Airline', 'Route', 'Month']).size().reset_index()
    flight_count.rename(columns={0: 'flight_count'}, inplace=True)
    
    competitor_df = pd.DataFrame()
    for airline in train_data['Airline'].unique():
        temp = flight_count.loc[flight_count['Airline']!=airline,:]
        competitor_count = temp.groupby(['Route', 'Month']).agg(
                {'flight_count': 'sum'}).reset_index()
        competitor_count.rename(columns={'flight_count': 'competitor_flight_count'}, inplace=True)
        airline_ = pd.DataFrame({'Airline': [airline]*(len(competitor_count))})
        temp_df = pd.concat([airline_, competitor_count], axis=1)
        competitor_df = pd.concat([competitor_df, temp_df], axis=0)
    train_data = pd.merge(train_data, competitor_df, on=
                          ['Airline', 'Route', 'Month'], how='left')
    test_data = pd.merge(test_data, competitor_df, on=
                          ['Airline', 'Route', 'Month'], how='left')
    train_data.fillna({'competitor_flight_count': 0}, inplace=True)
    test_data.fillna({'competitor_flight_count': 0}, inplace=True)
    

    ##  Competitors Flight average price
    
    competitor_df = pd.DataFrame()
    for airline in train_data['Airline'].unique():
        temp = train_data.copy().loc[train_data['Airline']!=airline,:]
        competitor_avg = temp.groupby(['Route']).agg(
                {'Price': 'mean'}).reset_index()
        competitor_avg.rename(columns={'Price': 'competitor_avg_price'}, inplace=True)
        airline_ = pd.DataFrame({'Airline': [airline]*(len(competitor_avg))})
        temp_df = pd.concat([airline_, competitor_avg], axis=1)
        competitor_df = pd.concat([competitor_df, temp_df], axis=0)
    train_data = pd.merge(train_data, competitor_df, on=
                          ['Airline', 'Route'], how='left')
    test_data = pd.merge(test_data, competitor_df, on=
                          ['Airline', 'Route'], how='left')
    train_data.fillna({'competitor_avg_price': 0}, inplace=True)
    test_data.fillna({'competitor_avg_price': 0}, inplace=True)
    
    
    
    return train_data, test_data
    

original_data = clean_data(original_data)
test_data = clean_data(test_data)
original_data  = basic_feature_engineering(original_data)   
test_data = basic_feature_engineering(test_data)    
train_data, test_data  = advance_feature_engineering(original_data, test_data, holidays)


train_data.drop(['Date_of_Journey', 'Route', 'Dep_Time', 'Dep_Time_Hour'], axis=1, inplace=True)
test_data.drop(['Date_of_Journey', 'Route', 'Dep_Time', 'Dep_Time_Hour'], axis=1, inplace=True)

train_data['Additional_Info'].unique()

x_values = pd.get_dummies(train_data.drop('Price', axis=1))
x_values = x_values.drop(['Additional_Info_1 Short layover',
 'Additional_Info_2 Long layover',
 'Additional_Info_No Info',
 'Additional_Info_Red-eye flight',
 'Airline_Trujet',
 'via_1_HBX', 'via_1_IXA', 'via_1_IXZ', 'via_1_JLR', 'via_1_NDC', 'via_1_VTZ'], axis=1)
y_values = train_data['Price']    
#x_values.fillna(0, inplace=True)
lgbm_model = lgbm.LGBMRegressor(num_leaves=105, learning_rate =0.01, lambda_l1=0.0001,
                                n_estimators=1560, min_child_samples=3,
                                colsample_bytree = 0.46, max_bin=900)


train_predictions = pd.DataFrame({'predictions': lgbm_model.predict(x_values)})
train_predictions = pd.concat([train_data, train_predictions, y_values], axis=1)
train_predictions.to_csv("E:/Kaggle_Problem/Flight Ticket Prediction/train_predict.csv", index=False)


test_data = pd.get_dummies(test_data)
test_data.fillna(0, inplace=True)

predictions = pd.DataFrame(lgbm_model.predict(test_data))
predictions.to_csv("E:/Kaggle_Problem/Flight Ticket Prediction/01042019_v3.csv", index=False)




