"""
download and preprocess data for model.py

training target data : https://drive.google.com/file/d/1XsbdKgdQsuzvZbpZgj5wKtZEwezw91P5/view?usp=drive_link
training input data : https://drive.google.com/file/d/1JvHfyv8M6jvjYeTDIo8yyDw2j3-v8u9I/view?usp=drive_link
testing target data: https://drive.google.com/file/d/1gjyac9WKbFafmDVK-MB-RwTeU5421E0b/view?usp=drive_link
testing input data: https://drive.google.com/file/d/1-OCHp2bVz3wa_0JFnDe61qILWPZyQqPs/view?usp=drive_link


"""

import csv
import requests
import numpy as np
import pandas as pd
import torch

class getData():
    def __init__(self):
        return None
        # self.train_input_link = train_input_link
        # self.train_target_link = train_target_link
        # self.test_input_link = test_input_link
        # self.test_target_link = test_target_link

    def download_data(self, CSV_URL, folder_dir, file_name_with_extension):

        # not inplement

        df = pd.read_csv(CSV_URL)   
        print(df.head())
        df.to_csv(folder_dir + file_name_with_extension)
        
        # # url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&apikey=demo&datatype=csv'
        # response = requests.get(CSV_URL)        

        # with open(folder_dir + file_name_with_extension, 'w') as f:
        #     writer = csv.writer(f)
        #     for line in response.iter_lines():
        #         writer.writerow(line.decode('utf-8').split(','))

    def prepare_data(self, batch_size=32, shuffle=False, train_val_split=0.2):

        """
        convert csv data to numpy arrays data
        """
        
        data_list = []
        with open("./data/train_input_matrix.csv", 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                row_data = [float(data) for data in row]
                data_list.append(row_data)

            data_arr = np.array(data_list, dtype="float32")
            print(f"data shape:{data_arr.shape}")
            train_input_matrix = data_arr
        
        data_list = []
        with open("./data/train_target_matrix.csv", 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                row_data = [float(data) for data in row]
                data_list.append(row_data)

            data_arr = np.array(data_list, dtype="float32")
            print(f"data shape:{data_arr.shape}")
            train_target_matrix = data_arr

        data_list = []
        with open("./data/test_input_matrix.csv", 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                row_data = [float(data) for data in row]
                data_list.append(row_data)

            data_arr = np.array(data_list, dtype="float32")
            print(f"data shape:{data_arr.shape}")
            test_input_matrix = data_arr

        data_list = []
        with open("./data/test_target_matrix.csv", 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                row_data = [float(data) for data in row]
                data_list.append(row_data)

            data_arr = np.array(data_list, dtype="float32")
            print(f"data shape:{data_arr.shape}")
            test_target_matrix = data_arr

        num_of_train_samples = train_input_matrix.shape[0]
        num_of_test_samples = test_input_matrix.shape[0]

        if shuffle:
            np.random.shuffle(train_input_matrix)
            np.random.shuffle(train_target_matrix)
            np.random.shuffle(test_input_matrix)
            np.random.shuffle(test_target_matrix)

        start = 0
        end = batch_size
        input_batches = []
        target_batches = []
        while start < num_of_train_samples:
            try:
                input_batches.append(torch.from_numpy(np.expand_dims(train_input_matrix[start:end][:], axis=1)))
                target_batches.append(torch.from_numpy(np.expand_dims(train_target_matrix[start:end][:], axis=1)))
            except: # if the size of the last batch is smaller than the specified batch size 
                input_batches.append(torch.from_numpy(np.expand_dims(train_input_matrix[start:][:], axis=1)))
                target_batches.append(torch.from_numpy(np.expand_dims(train_target_matrix[start:][:], axis=1)))
            start += batch_size 
            end += batch_size
        train_input_batches = input_batches
        train_target_batches = target_batches

        start = 0
        end = batch_size
        input_batches = []
        target_batches = []
        while start < num_of_test_samples:
            try:
                input_batches.append(torch.from_numpy(np.expand_dims(test_input_matrix[start:end][:], axis=1)))
                target_batches.append(torch.from_numpy(np.expand_dims(test_target_matrix[start:end][:], axis=1)))
            except: # if the size of the last batch is smaller than the specified batch size 
                input_batches.append(torch.from_numpy(np.expand_dims(test_input_matrix[start:][:], axis=1)))
                target_batches.append(torch.from_numpy(np.expand_dims(test_target_matrix[start:][:], axis=1)))
            start += batch_size 
            end += batch_size
        test_input_batches = input_batches
        test_target_batches = target_batches

        print(f"train input shape:{len(train_input_batches)}{train_input_batches[0].shape}")
        print(f"train target shape:{len(train_target_batches)}{train_target_batches[0].shape}")
        
        print(f"test input shape:{len(test_input_batches)}{test_input_batches[0].shape}")
        print(f"test target shape:{len(test_target_batches)}{test_target_batches[0].shape}")
    

        assert train_val_split < 0.5 and train_val_split > 0, "split ratio should be larger than 0 and smaller than 0.5"
        
        if int(train_val_split*10):
            val_input_batches = train_input_batches[:int(train_val_split * len(train_input_batches))]
            train_input_batches = train_input_batches[int(train_val_split * len(train_input_batches)):]

            val_target_batches = train_target_batches[:int(train_val_split * len(train_input_batches))]
            train_target_batches = train_target_batches[int(train_val_split * len(train_input_batches)):]
        else:
            val_input_batches = []
            val_target_batches = []

        return train_input_batches, train_target_batches, val_input_batches, val_target_batches,test_input_batches, test_target_batches

       
    



if __name__ == "__main__":
    
    gd = getData()
    train_input, train_target, test_input, test_tartget = gd.prepare_data(batch_size=32, shuffle=True)


