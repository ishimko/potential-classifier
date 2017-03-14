from potential_method import PotentialClassifier

def main():
    training_data = [
        ([-1, 0], True),
        ([1, 1], True),
        ([2, 0], False),
        ([1, -2], False)
    ]
    classifier = PotentialClassifier(training_data)
    print(classifier.coefficients)

if __name__ == '__main__':
    main()
