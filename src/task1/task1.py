import csv
import numpy
import scipy.optimize


def loadfile(filepath):

    filecontents = []

    with open(filepath) as datafile:
        reader = csv.reader(datafile)

        for row in reader:
            filecontents += [row]

    return filecontents


def compile_data(file_content):

    data = []

    for i in xrange(10, len(file_content) - 1):

        prev_data = file_content[i - 10: i]

        # This flattens the list and creates two arrays, one for price, the other for previous days data.
        data += [[float(file_content[i][1]), [float(item) for sublist in prev_data for item in sublist]]]

    return data


def generate_training_data(data):

    return data[1::2]


def generate_test_data(data):

    return data[0::2]


def calculate_mean_sq_error(theta, data):

    total_sq_error = 0

    for data_row in data:
        total_sq_error += (numpy.dot(theta, data_row[1]) - data_row[0]) ** 2

    total_sq_error /= len(data)

    return total_sq_error


def linear(file):

    theta0 = [0] * 20

    file_content = loadfile(file)

    data = compile_data(file_content)

    training_data = generate_training_data(data)

    result = scipy.optimize.fmin(calculate_mean_sq_error, x0=theta0, args=tuple([training_data]))

    return calculate_mean_sq_error(result, generate_test_data(data))

print "MSE = " + str(linear("../../data/stock_price.csv"))