import numpy as np

class PotentialClassifier:
    ERMIT_COEFFICIENTS = np.array([1, 4, 4, 16])
    SUBSTITUTION_MAP = {
        1: [0],
        2: [1],
        3: [0, 1]
    }

    def __init__(self, training_data):
        self.coefficients = np.array([0]*len(PotentialClassifier.ERMIT_COEFFICIENTS))
        self._train(training_data)

    def _train(self, training_data):
        assert len(training_data) > 1
        for vector, expected_result in training_data:
            local_potential = self.get_local_potential(vector)
            decision = self.get_decision(vector)
            punishment = (int(expected_result) - int(decision)) * local_potential
            self.coefficients += punishment

    def get_decision(self, vector):
        return self.get_potential(vector) > 0

    def get_potential(self, vector):
        return sum(PotentialClassifier._substitute(self.coefficients, vector))

    @staticmethod
    def get_local_potential(vector):
        return PotentialClassifier._substitute(PotentialClassifier.ERMIT_COEFFICIENTS, vector)

    @staticmethod
    def _substitute(coefficients, vector):
        assert len(vector) == 2
        coefficients = coefficients.copy()
        for index, substitution_elements in PotentialClassifier.SUBSTITUTION_MAP.items():
            coefficients[index] *= np.prod([vector[i] for i in substitution_elements])
        return coefficients
