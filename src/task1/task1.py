import csv
import numpy
import scipy.optimize
import random


def loadfile(file_path):

    file_contents = []

    with open(file_path) as datafile:
        reader = csv.reader(datafile)

        for row in reader:
            file_contents += [row]

    return file_contents


def generate_features(data):

        features = []

        # features += [float(item[1]) for item in data]  # Adds the 10 previous days prices as features.

        features += [float(data[9][1])]  # Adds the previous days price as a feature.

        # features += [float(data[9][1]) - float(data[0][1])]  # Adds the rate of change of price over prev 10 days as a feature

        # features += [float(item[0]) for item in data]  # Adds the 10 previous days volume as features.

        # features += [1]  # Adds a constant feature.

        return features


def compile_data(file_content):

    compiled_data = []

    for i in xrange(10, len(file_content) - 1):

        current_price = float(file_content[i][1])

        prev_data = file_content[i - 10: i]

        features = generate_features(prev_data)

        compiled_data_row = [current_price, features]

        compiled_data += [compiled_data_row]

    return compiled_data


def generate_data_folds(data):

    random.shuffle(data)

    folded_data = data[0::2], data[1::2]

    return folded_data


def calculate_mean_sq_error(theta, data):

    total_sq_error = 0

    for data_row in data:
        total_sq_error += (numpy.dot(theta, data_row[1]) - data_row[0]) ** 2

    total_sq_error /= len(data)

    return total_sq_error


def generate_theta_zero(data):

    return [0] * len(data[0][1])


def perform_regression(theta0, training_data, test_data):

    result = scipy.optimize.fmin(calculate_mean_sq_error, x0=theta0, args=tuple([training_data]), disp=False)

    error = calculate_mean_sq_error(result, test_data)

    return error


def linear(file_path):

    file_content = loadfile(file_path)

    data = compile_data(file_content)

    theta0 = generate_theta_zero(data)

    folded_data = generate_data_folds(data)

    result0 = perform_regression(theta0, folded_data[0], folded_data[1])

    result1 = perform_regression(theta0, folded_data[1], folded_data[0])

    return (result0 + result1) / 2

print "MSE = " + str(linear("../../data/stock_price.csv"))