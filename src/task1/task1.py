import csv
import numpy
import scipy.optimize
import scipy.misc
import random


def load_file(file_path):

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

    # features += [float(data[8][1])]  # Adds the stock price from two days ago as a feature.

    # features += [float(data[9][1]) - float(data[0][1])]  # Adds the rate of change of price over prev 10 days as a feature

    # features += [float(item[0]) for item in data]  # Adds the 10 previous days volume as features.

    # features += [1]  # Adds a constant feature.

    return features


def compile_data_linear(file_content):

    compiled_data = []

    for i in xrange(10, len(file_content)):

        current_price = float(file_content[i][1])

        prev_data = file_content[i - 10: i]

        features = generate_features(prev_data)

        compiled_data_row = [current_price, features]

        compiled_data += [compiled_data_row]

    return compiled_data


def compile_data_logistic(file_content):

    compiled_data = []

    for i in xrange(10, len(file_content)):

        current_price = float(file_content[i][1])

        prev_data = file_content[i - 10: i]

        prev_price = float(prev_data[9][1])

        row_class = classify_data(current_price, prev_price)

        features = generate_features(prev_data)

        compiled_data_row = [current_price, row_class, features]

        compiled_data += [compiled_data_row]

    return compiled_data


def classify_data(price, prev_price):

    minus5 = prev_price * 0.95
    plus5 = prev_price * 1.05
    minus10 = prev_price * 0.9
    plus10 = prev_price * 1.1

    if minus5 <= price <= plus5:
        return 0
    elif plus5 < price <= plus10:
        return 1
    elif minus5 > price >= minus10:
        return 2
    elif price > plus10:
        return 3
    elif price < minus10:
        return 4
    else:
        raise Exception("Unable to classify, maths is broken!")


def generate_data_folds(data):

    random.shuffle(data)

    folded_data = data[0::2], data[1::2]

    return folded_data


def get_features(data_row):

    return data_row[-1]


def get_price(data_row):

    return data_row[0]


def mean_sq_error(theta, data):

    total_sq_error = 0

    for data_row in data:
        total_sq_error += (numpy.dot(theta, get_features(data_row)) - get_price(data_row)) ** 2

    total_sq_error /= float(len(data))

    return total_sq_error


def get_theta_for_class(theta, class_index, features):

    return theta[class_index * features: (class_index + 1) * features]


def get_features_size(data):

    return len(get_features(data[0]))


def get_class(data_row):

    return data_row[1]


def classifier_estimation_logarithmic(theta, data, classes):

    estimation = 0

    features_size = get_features_size(data)

    for data_row in data:

        row_class = get_class(data_row)

        features = get_features(data_row)

        current_features_dot_thetas = numpy.dot(features, get_theta_for_class(theta, row_class, features_size))

        max_feature = max(features)

        features_dot_thetas = []

        for clazz in range(classes):

            features_dot_thetas.append(numpy.dot(features, get_theta_for_class(theta, clazz, features_size)) - max_feature)

        sum_features_dot_thetas = max_feature + scipy.misc.logsumexp(features_dot_thetas)

        current_estimation = current_features_dot_thetas - sum_features_dot_thetas

        estimation -= current_estimation

    return estimation


def classifier_error(theta, data, classes):

    error = 0

    features_size = get_features_size(data)

    for data_row in data:

        computed_probabilities = []

        sum_exp_features_dot_theta = 0

        for clazz in range(classes):
            sum_exp_features_dot_theta += numpy.exp(numpy.dot(get_features(data_row), get_theta_for_class(theta, clazz, features_size)))

        for clazz in range(classes):
            sum_exp_current_feature_dot_theta = numpy.exp(numpy.dot(get_features(data_row), get_theta_for_class(theta, clazz, features_size)))

            computed_probabilities.append(sum_exp_current_feature_dot_theta / sum_exp_features_dot_theta)

        predicted_class = computed_probabilities.index(max(computed_probabilities))

        if predicted_class != get_class(data_row):
            error += 1

    return error / float(len(data))


def generate_theta_zero(data, classes=1):

    return [[0] * len(get_features(data[0])) * classes]


def perform_linear_regression(training_data, test_data):

    theta0 = generate_theta_zero(training_data)

    result = scipy.optimize.fmin(mean_sq_error, x0=theta0, args=tuple([training_data]), disp=False)

    error = mean_sq_error(result, test_data)

    return error


def perform_logistic_regression(training_data, test_data):

    classes = 5

    theta0 = generate_theta_zero(training_data, classes=classes)

    result = scipy.optimize.fmin(classifier_estimation_logarithmic, x0=theta0, args=(training_data, classes), disp=False)

    error = classifier_error(result, test_data, classes)

    return error


def linear(file_path):

    file_content = load_file(file_path)

    data = compile_data_linear(file_content)

    folded_data = generate_data_folds(data)

    result0 = perform_linear_regression(folded_data[0], folded_data[1])

    result1 = perform_linear_regression(folded_data[1], folded_data[0])

    return (result0 + result1) / 2


def logistic(file_path):

    file_content = load_file(file_path)

    data = compile_data_logistic(file_content)

    folded_data = generate_data_folds(data)

    result0 = perform_logistic_regression(folded_data[0], folded_data[1])

    result1 = perform_logistic_regression(folded_data[1], folded_data[0])

    return (result0 + result1) / 2


print "Mean Squared Error   = " + str(linear("../../data/stock_price.csv"))
print "Hard Classifer Error = " + str(logistic("../../data/stock_price.csv"))