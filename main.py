from potential_method import PotentialClassifier
from drawer import show_results
from collections import defaultdict
from random import uniform


TRAINING_DATA = [
    ([-1, 0], True),
    ([1, 1], True),
    ([2, 0], False),
    ([1, -2], False)
]


def main():
    classifier = PotentialClassifier(TRAINING_DATA)
    print(classifier.coefficients)
    classified_data = defaultdict(list)
    for vector, decision in TRAINING_DATA:
        classified_data[decision].append(vector)
    show_results(classifier.decision_funtion_for_plotting, classifier.get_function_break_point(), classified_data)
    for _ in range(1000):
        vector = (uniform(-10, 10), uniform(-10, 10))
        decision = classifier.get_decision(vector)
        classified_data[decision].append(vector)
    show_results(classifier.decision_funtion_for_plotting, classifier.get_function_break_point(), classified_data)


if __name__ == '__main__':
    main()
