import csv
import numpy
import scipy.optimize


def loadfile(file_path):

    file_contents = []

    with open(file_path) as datafile:
        reader = csv.reader(datafile)

        for row in reader:
            file_contents += [row]

    return file_contents


def compile_data(file_content):

    compiled_data = []

    for i in xrange(10, len(file_content) - 1):

        current_price = float(file_content[i][1])

        prev_data = file_content[i - 10: i]

        features = []

        features += [float(item[1]) for item in prev_data]  # Adds the 10 previous days prices as features.

        # row_data += [float(item[0]) for item in prev_data]  # Adds the 10 previous days volume as features.

        features += [1]  # Adds a constant feature.

        compiled_data_row = [current_price, features]

        compiled_data += [compiled_data_row]

    return compiled_data


def generate_training_data(data):

    return data[0::2]


def generate_test_data(data):

    return data[1::2]


def calculate_mean_sq_error(theta, data):

    total_sq_error = 0

    for data_row in data:
        total_sq_error += (numpy.dot(theta, data_row[1]) - data_row[0]) ** 2

    total_sq_error /= len(data)

    return total_sq_error


def generate_theta_zero(data):

    return [0] * len(data[0][1])


def linear(file_path):

    file_content = loadfile(file_path)

    data = compile_data(file_content)

    theta0 = generate_theta_zero(data)

    training_data = generate_training_data(data)

    result = scipy.optimize.fmin(calculate_mean_sq_error, x0=theta0, args=tuple([training_data]))

    print "Theta = " + str(result)

    return calculate_mean_sq_error(result, generate_test_data(data))

print "MSE = " + str(linear("../../data/stock_price.csv"))