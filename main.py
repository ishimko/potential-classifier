from potential_method import PotentialClassifier
from drawer import plot_function_with_break_point

def main():
    training_data = [
        ([-1, 0], True),
        ([1, 1], True),
        ([2, 0], False),
        ([1, -2], False)
    ]
    classifier = PotentialClassifier(training_data)
    plot_function_with_break_point(classifier.decision_funtion_for_plotting, classifier.get_function_break_point(), (-10, 10))

if __name__ == '__main__':
    main()
