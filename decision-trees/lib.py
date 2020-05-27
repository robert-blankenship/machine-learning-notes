from math import log
import operator
import matplotlib.pyplot


def calculate_shannon_entropy(data_set):
    label_counts = {}

    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_entropy = 0.0
    for key in label_counts:
        probability = float(label_counts[key]) / len(data_set)
        shannon_entropy -= probability * log(probability, 2)

    return shannon_entropy


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    return_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            return_data_set.append(reduced_feature_vector)

    return return_data_set


def choose_best_feature_to_split_robert(data_set):
    entropies = []

    for feature in range(len(data_set[0]) - 1):

        possible_values = set([vector[feature] for vector in data_set])

        entropy = 0
        for value in possible_values:
            split_data = split_data_set(data_set, feature, value)
            probability = len(split_data) / float(len(data_set))
            entropy += probability * calculate_shannon_entropy(split_data)

        entropies.append(entropy)

    for i, value in enumerate(entropies):
        if value == min(entropies):
            return i


def choose_best_feature_to_split(data_set):
    number_of_features = len(data_set[0]) - 1
    base_entropy = calculate_shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(number_of_features):
        feature_list = [el[i] for el in data_set]
        unique_values = set(feature_list)
        new_entropy = 0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            probability = len(sub_data_set) / float(len(data_set))
            new_entropy += probability * calculate_shannon_entropy(sub_data_set)

        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


data_set, labels = create_data_set()
print choose_best_feature_to_split_robert(data_set)
print choose_best_feature_to_split(data_set)


def create_tree(data_set, labels):
    class_list = [el[-1] for el in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_count(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]

    my_tree = {best_feature_label: {}}
    del(labels[best_feature])

    feature_values = set([el[best_feature] for el in data_set])
    for value in feature_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)

    return my_tree


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_txt, center_point, parent_point, node_type, axis):
    axis.annotate(node_txt, xy=parent_point, xycoords='axes fraction',
                             xytext=center_point, textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)

def create_plot():
    figure = matplotlib.pyplot.figure(1, facecolor="white")
    figure.clf()
    ax1 = matplotlib.pyplot.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node, ax1)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), decision_node, ax1)
    matplotlib.pyplot.show()

create_plot()

