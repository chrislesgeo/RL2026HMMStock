import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_state
        )

    def fit(self, features):
        self.model.fit(features)

    def predict_states(self, features):
        return self.model.predict(features)

    def predict_probabilities(self, features):
        return self.model.predict_proba(features)