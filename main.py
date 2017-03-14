from sys import argv
from collections import defaultdict
from random import uniform
from potential_classifier import PotentialClassifier
from drawer import show_results


INTERVAL = -10, 10
VECTORS_COUNT = 1000


def load_training_data(training_data_path):
    result = []
    with open(training_data_path, 'r') as input_file:
        for line in input_file.readlines():
            items = line.split()
            vector = list(map(int, items[:2]))
            decision = items[2] == 'True'
            result.append((vector, decision))
    return result


def main():
    training_data_path = argv[1]
    training_data = load_training_data(training_data_path)
    classifier = PotentialClassifier(training_data)
    print('Coefficients: {}'.format(classifier.coefficients))
    classified_data = defaultdict(list)
    for vector, decision in training_data:
        classified_data[decision].append(vector)
    show_results(classifier.coefficients, classified_data)
    for _ in range(VECTORS_COUNT):
        vector = (uniform(*INTERVAL), uniform(*INTERVAL))
        decision = classifier.get_decision(vector)
        classified_data[decision].append(vector)
    show_results(classifier.coefficients, classified_data)


if __name__ == '__main__':
    main()
