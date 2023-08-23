import pandas as pd
import glob
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Data_prep:
    def __init__(self):
        self.df = None #Initialise the dataframe as none  (We could get rid of this as well. It's just because we already set it up)

    def load_csv(self, filepath, weeks = 2):
        '''Loads all the csv files present in the demand data folder, generates a timestamp with date and time (in 30 minute windows) and joins the csv files chronologically then filters for dates between 2015 and 2019

        Input: path to folder. Number of weeks to be used for training. Unless user gives a different input, this is set to 2 but is automatically updated if user provides different input

        Output: Updates the csv file created at the initialisation point. Also, a method variable called training duration'''

        #Create a list of all csv files using the provided filepath
        all_csv_files = glob.glob(filepath + '/*.csv')
        
        #Create empty demand list onto which we will read csv files as individual lists of files
        demand_list = []

        #loop to read through all the filenames that we have created using glob module - those that end with csv
        for filename in all_csv_files:
            #reading individual csv file in each cycle
            df = pd.read_csv(filename, index_col=None, header = 0)
            #extract that specific's file first settlement date as the start date - using iloc
            date = pd.to_datetime(df['SETTLEMENT_DATE'].iloc[0])
            #create an empty list onto which we will be appending all the generated timestamps
            timestamps = []
            #for loop to generate the timestamps of corresponding length to the dataframe and in timesteps of 30 minutes then append that data to the timestamps list
            for i in range(0, len(df)*30, 30):
                timestamps.append(date + timedelta(minutes=i))
            #adding the new timestamps lists as an extra column for time identification in the new dataframe
            df['date'] = timestamps
            #appending the new dataframe (with each cycle) into the list of the dataframes
            demand_list.append(df)

        #Now concatenate all the lists into one dataframe
        df_demand = pd.concat(demand_list, axis = 0, ignore_index=True)

        #Filter for only pre-covid demand
        df_demand = df_demand[df_demand['date'] < '2020-01-01 00:00:00']
        df_demand = df_demand[df_demand['date'] > '2014-12-31 23:30:00']

        #Reset the index to identify the split points better later and update df at the init point
        self.df = df_demand.reset_index(drop = True)
        
        #Compute the training duration depending on the number of weeks provided by user
        training_duration = weeks*7*48

        #Print on terminal to show user the method output
        print('DATA LOADING')
        print(f'Input data runs from ', df_demand['date'].iloc[0], 'to', df_demand['date'].iloc[-1], 'and with', len(df_demand), 'datapoints')
        print(f'Training duration is', training_duration, ' timesteps, an equivalent of', weeks, 'weeks')
        
        #return the training duration as a method variable
        return training_duration

    def train_val_test_split(self, training_duration, date_1 = '2016-01-01 00:00:00', date_2 = '2017-01-01 00:00:00', date_3 = '2018-01-01 00:00:00'):

        '''Picks the loaded csv file and using the date column, identifies the split points the performs the splitting. Note that before the split, it retrieves the column to be split and converts it to a list of floats in GW (division by 1000) rather than MW. Note that to change the split points, the user has to update the dates manually in the code. Note that the dates for splitting have been preset but the user can provide appropriate dates in the provided format.

        Input - training_duration, dates and the initialised/ updated dataframe df
        Output - method-level updates train, val and test sets'''
        if self.df is not None:
            #RETRIEVE DEMAND AND TRAINING DURATION DATA, CONVERT IT TO A LIST OF FLOAT VALUES AFTER MOVING IT FROM MW TO GW
            demand = list(map(float, (self.df['ND']/1000)))

            #IDENTIFY THE SPLIT POINTS

            #First years for training ('15). Adjust this when it comes to main training to 2015
            split_point_one = self.df[self.df['date'] == date_1].index[0]

            #'18 set aside as the valudation set and 2019 for the test set
            split_point_two = self.df[self.df['date'] == date_2].index[0]

            #'18 set aside as the valudation set and 2019 for the test set. Just for quick testing
            split_point_three = self.df[self.df['date'] == date_3].index[0]

            #PERFORMING THE SPLIT

            #Then splitting the data with the training weeks being an overlap window
            train, val, test = demand[0:split_point_one], demand[(split_point_one - training_duration):(split_point_two)], demand[(split_point_two - training_duration):split_point_three]

            #Printing on terminal just to confirm to user that method is working
            print('\nSPLITTING')
            print('Split results')
            print('_____________') 
            print('Set Length Datapoint_1')
            print(f'Train', len(train), train[0])
            print(f'Val', len(val), val[0])
            print(f'Test', len(test), test[0])

        else:
            print('Demand data not found, please provide path to folder first')
        


        return train, val, test
    
    def train_val_test_scaling(self, train, val, test):
        '''Takes the previously updated train, validation and test sets, performs the scaling on them then updates them at the init point
        Input - train, test and validation data which have been already instantiated
        Output - an update to the instantiated train, test and validation data
        Note that we have saved all the data at the init point to allow for the user to track the process by running each method hence ensure that everything is functional. Later on, we will update this such that '''

        #PERFORMING NORMALISATION
        #Create normalisation object
        scaler = MinMaxScaler()

        # fit data. This identifies the max/ min demand and computes the mean and standard deviation to be used by all the other transformations
        scaler.fit(np.array(train).reshape(-1, 1))

        #Perform the transform but the data needs to be converted from list to array then reshaped
        train_scaled = scaler.transform(np.array(train).reshape(-1, 1))
        #Reshaping the train_scaled into a one dimensional array again
        train_scaled = train_scaled.reshape(-1)

        #Transforming the validation data
        val_scaled = scaler.transform(np.array(val).reshape(-1,1))
        #Reshaping the val_scaled into a one dimensional array again
        val_scaled = val_scaled.reshape(-1)

        #Fitting the test data on its own
        scaler.fit(np.array(test).reshape(-1, 1))

        #Transforming the test data on its own
        test_scaled = scaler.transform(np.array(test).reshape(-1, 1))
        #Reshaping the test_scaled into a one dimensional array again
        test_scaled = test_scaled.reshape(-1)

        #Print just to confirm to user that the values have changed hence MinMax transformation effective
        print('\nSCALING')
        print('Scaled results')
        print('_____________') 
        print('Set Length Datapoint_1')
        print(f'Train', len(train_scaled), train_scaled[0])
        print(f'Val', len(val_scaled), val_scaled[0])
        print(f'Test', len(test_scaled), test_scaled[0])
        
        return train_scaled, val_scaled, test_scaled

    def reshaping(self, data, n_features):
        '''Takes the initialised and updated (following the scaling) train, val and test lists and reshapes them for the time series prediction model
        The reshaping is implemeted through the split_sequence function which returns results in the shape of batches, features, batch_size. The method then reshapes this into batches, batch_size, features which is appropriate for time forecasting models
        Input - previously initialised and updated demand data (lists), and the number of input features in the data as an integer (in this case, 1)
        Output - the reshaped arrays in order of X_train, y_train, X_val, y_val, X_test, y_test. However, these are saved at the init method hence easily accessible'''

        #SETTING UP THE FUNCTION TO PERFORM THE RESHAPING
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            #loop through the list and update the days so that instead of sliding the window across one point, its across 48 points (one day)
            for i in range(0, len(sequence), 48):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence)-48:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix: end_ix+48]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
        #Calling the split_sequence function to split training into x_train and y_train
        X_train, y_train = split_sequence(data[0], data[1])

        #reshaping the x_train so that the features is the last value
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

        #Calling the split sequence function to split validation into x_val and y_val
        X_val, y_val = split_sequence(data[2], data[1])
        #reshaping the x_train so that the features is the last value
        #X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

        #Calling the split sequence function to split validation into x_val and y_val
        X_test, y_test = split_sequence(data[3], data[1])

        #reshaping the x_test so that the features is the last value
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        print('\n RESHAPING')
        print('Reshaping results')
        print('_________________')
        print(f'X_train', X_train.shape)
        print(f'y_train', y_train.shape)
        print(f'X_val', X_val.shape)
        print(f'y_val', y_val.shape)
        print(f'X_test', X_test.shape)
        print(f'y_test', y_test.shape)


        return X_train, y_train, X_val, y_val, X_test, y_test
    


