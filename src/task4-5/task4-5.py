import csv
import numpy
import itertools
from random import uniform
from scipy.stats import beta


def load_structure(structure_file_path):
    file_contents = []

    with open(structure_file_path) as data_file:
        reader = csv.reader(data_file)

        for row in reader:
            file_contents += [[int(data) for data in row]]

    return file_contents


def load_data(data_file_path):
    file_contents = []

    with open(data_file_path) as data_file:
        reader = csv.reader(data_file)

        for row in reader:
            file_contents += [[int(data) for data in row]]

    return file_contents


def get_prior_distribution():
    return [0.5, 0.5]


def generate_parent_matrix(child_matrix):
    return numpy.transpose(child_matrix)


def generate_beta_distribution_mean(alpha_val, beta_val):
    return beta.mean(alpha_val, beta_val)


def find_parent_nodes(node_id, structure):
    parents = []

    for possible_parent in range(len(structure)):

        if structure[node_id][possible_parent] == 1:
            parents += [possible_parent]

    return parents


def is_valid_data_row(data_row, parents, permutation):
    for parent_loc in range(len(parents)):

        if data_row[parents[parent_loc]] != permutation[parent_loc]:
            return False

    return True


def generate_permutations(len_parents):
    return list(itertools.product([0, 1], repeat=len_parents))


def generate_distributions_for_node(node_id, data, structure):
    parents = find_parent_nodes(node_id, structure)

    distribution_data = [node_id, parents]

    permutations = generate_permutations(len(parents))

    if len(permutations) == 0:
        permutations = [0]

    for permutation in permutations:

        current_counts = get_prior_distribution()

        for data_row in data:

            if is_valid_data_row(data_row, parents, permutation):
                current_counts[data_row[node_id]] += 1

        distribution_data.append(generate_beta_distribution_mean(current_counts[1], current_counts[0]))

    return distribution_data


def generate_distributions(data, structure):
    distributions = []

    for node_id in range(len(structure)):
        distributions.append(generate_distributions_for_node(node_id, data, structure))

    return distributions


def shuffle_node_order(bayes_net_spec):
    sorted_nodes = []

    sorted_node_indices = []

    while len(sorted_nodes) < len(bayes_net_spec):

        for current_node_index in range(len(bayes_net_spec)):

            parent_nodes = bayes_net_spec[current_node_index][1]

            if (not current_node_index in sorted_node_indices) and (
                    all([parent in sorted_node_indices for parent in parent_nodes])):
                sorted_nodes.append(bayes_net_spec[current_node_index])
                sorted_node_indices.append(current_node_index)

    return sorted_nodes


def generate_random_truth_value(distribution_mean):
    generated_random = uniform(0, 1)

    if generated_random < distribution_mean:
        return 1
    else:
        return 0


def get_permutation_index(parent_values):
    permutations = generate_permutations(len(parent_values))

    for index in range(len(permutations)):

        if list(permutations[index]) == parent_values:
            return index

    return None


def get_distribution_mean_at_permutation_index(index, node):
    return node[2 + index]


def get_parent_truth_values(node, data_row):
    parent_truth_values = []

    parents = node[1]

    for parent in parents:
        parent_truth_values += [data_row[parent]]

    return parent_truth_values


def find_distribution_mean(data_row, node):
    parents = node[1]

    if len(parents) == 0:

        return node[2]

    else:

        parent_values = get_parent_truth_values(node, data_row)

        permuatation_index = get_permutation_index(parent_values)

        return get_distribution_mean_at_permutation_index(permuatation_index, node)


def generate_data_row(ordered_fitted_bn):
    data_row = [0 for _ in range(len(ordered_fitted_bn))]

    for node in ordered_fitted_bn:

        parents = node[1]

        if len(parents) == 0:

            data_row[node[0]] = generate_random_truth_value(get_distribution_mean_at_permutation_index(0, node))

        else:

            distribution_mean = find_distribution_mean(data_row, node)

            data_row[node[0]] = generate_random_truth_value(distribution_mean)

    return data_row


def generate_data(ordered_fitted_bn, samples):
    data = []

    for _ in xrange(samples):
        data.append(generate_data_row(ordered_fitted_bn))

    return data


def bnbayesfit(structure_file_path, data_file_path):
    structure = load_structure(structure_file_path)

    structure_parents = generate_parent_matrix(structure)

    data = load_data(data_file_path)

    return generate_distributions(data, structure_parents)


def bnsample(fitted_bn, nsamples):
    shuffled_nodes = shuffle_node_order(fitted_bn)

    return generate_data(shuffled_nodes, nsamples)


bayes_net_spec = bnbayesfit("../../data/bnstruct.csv", "../../data/bndata.csv")

for thing in bayes_net_spec:
    print thing

generated_sample = bnsample(bayes_net_spec, 10000)

structure = load_structure("../../data/bnstruct.csv")

structure_fixed = generate_parent_matrix(structure)

generated_bayes_net_spec = generate_distributions(generated_sample, structure_fixed)
print

for thing in generated_bayes_net_spec:
    print thing
