import random
import numpy as np

def naive_split(raw_data):
    X = []
    y = []

    for i in range(len(raw_data)):
        k = random.randint(0, len(raw_data[i]) - 1)
        X.append(raw_data[i][k])
        y.append(raw_data[i][len(raw_data[i]) - 1])

    return np.array(X), np.array(y)



def nested_split(raw_data):
    X = []
    y = []

    for i in range(len(raw_data)):
        curr_ls = raw_data[i]
        for j in range(1, len((curr_ls))):
            curr_X = []
            for k in range(j):
                curr_X.append(curr_ls[k])
            y.append(curr_ls[j])
            X.append(curr_X)

    return X, y

def split_data(raw_data):
    X = []
    y = []

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i]) - 1):
            X.append(raw_data[i][j])
            y.append(raw_data[i][j + 1])

    return np.array(X), np.array(y)


def convert_to_list(input_str):
    input_list_by_line = input_str.split()
    for i in range(len(input_list_by_line)):
        input_list_by_line[i] = input_list_by_line[i].split(',')
        for j in range(len(input_list_by_line[i])):
            input_list_by_line[i][j] = int(input_list_by_line[i][j])

    return input_list_by_line

data_dir = './datasets/diginetica/'




'''
This function will get total number of items in the file of num_items.txt
'''
def get_total_number_of_item():
    f = open(data_dir + "num_items.txt", 'r')
    return int(f.read())

"""
a helper function for the get_as_list, the input is a 2-d list, for the session (all data in the session), and the output will be 
X: an numpy_array(list()) that contain all the previous items in the session
y: an numpy_array of the final item in the session
"""
def simple_split(inputs):
    X_list = []
    y_list = []
    for curr in inputs:
        X_list.append(curr[:len(curr) - 1])
        y_list.append(curr[len(curr) - 1])

    return np.array(X_list), np.array(y_list)

"""
return the regular split of X and y for train set and test set
"""
def get_as_list():
    f_train = open(data_dir + "train.txt", 'r')
    f_test = open(data_dir + "test.txt", 'r')

    train_data = convert_to_list(f_train.read())
    test_data = convert_to_list(f_test.read())

    X_test, y_test = simple_split(test_data)
    X_trian, y_train = simple_split(train_data)

    return X_trian, X_test, y_train, y_test


def process_data():
    # folder for data
    f_train = open(data_dir + "train.txt", 'r')
    f_test = open(data_dir + "test.txt", 'r')

    train_data = convert_to_list(f_train.read())
    test_data = convert_to_list(f_test.read())

    X_train, y_train = naive_split(train_data)
    X_test, y_test = naive_split(test_data)
    return X_train, X_test, y_train, y_test


def get_as_nested_list():
    # folder for data
    f_train = open(data_dir + "train.txt", 'r')
    f_test = open(data_dir + "test.txt", 'r')

    train_data = convert_to_list(f_train.read())
    test_data = convert_to_list(f_test.read())

    X_train, y_train = nested_split(train_data)
    X_test, y_test = nested_split(test_data)
    return X_train, X_test, y_train, y_test

def num_items():
    f = open(data_dir + "num_items.txt", 'r')
    return int(f.read())