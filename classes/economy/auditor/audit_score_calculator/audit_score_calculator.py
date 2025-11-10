



### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic to calculate audit scores based on various metrics


import numpy as np 

from typing import List , Tuple , Union , Dict

class AuditScoreCalculator:
    """
    Computes an audit score based on weighted risk metrics using a sigmoid-type activation function.
    """

    def __init__(self, activation: str = "sigmoid"):
        """
        Parameters
        ----------
        activation : str
            Activation function to apply. Options:
            - 'sigmoid': standard logistic activation.
            - 'logit': inverse of sigmoid (log-odds transformation).
            - 'softmax': normalized exponential weighting.
        """
        valid_activations = {"sigmoid", "logit", "softmax"}
        if activation not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}")
        self.activation = activation

    def _sigmoid(self, z):
        """Standard sigmoid activation."""
        return 1 / (1 + np.exp(-z))

    def _logit(self, z):
        """Logit function (inverse of sigmoid)."""
        # Avoid division by zero or 1 boundaries
        eps = 1e-10
        z = np.clip(z, eps, 1 - eps)
        return np.log(z / (1 - z))

    def _softmax(self, values):
        """Softmax normalization for scores."""
        exp_vals = np.exp(values - np.max(values))
        return exp_vals / np.sum(exp_vals)

    def compute_score(self, metrics_dict: dict, weights: list):
        """
        Computes the audit likelihood score s = σ(Σ β_j M_j).

        Parameters
        ----------
        metrics_dict : dict
            Dictionary mapping metric names to numeric values.
        weights : list
            List of weights corresponding to each metric, in the same order as in metrics_dict.

        Returns
        -------
        float or dict
            The computed audit score (or normalized vector if softmax is used).
        """
        metric_values = np.array(list(metrics_dict.values()), dtype=float)
        weights = np.array(weights, dtype=float)

        if len(weights) < len(metric_values):
            raise ValueError("The number of weights must be equal to or greater than the number of metrics." , 'weights',len(weights) ,'#metrics', len(metric_values))
        
        elif len(weights) > len(metric_values):
            weights = weights[:len(metric_values)]

        # Normalize weights
        normalized_weights = weights / np.sum(weights)

        # Linear combination (z)
        z = np.dot(normalized_weights, metric_values)

        # Activation
        if self.activation == "sigmoid":
            score = self._sigmoid(z)
        elif self.activation == "logit":
            score = self._logit(self._sigmoid(z))
        elif self.activation == "softmax":
            score = self._softmax(metric_values * normalized_weights)

        return score