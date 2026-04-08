
### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic of an auditor that recieves an interface of informatio 
### of the list of good matrices, normalized matrices, good flow and monetary flow and calculates measures 



from src.simulation.economy.auditor.audit_orchestrator.audit_patterns.spectral_measures.spectral_methods import SpectralMethods

from src.simulation.economy.auditor.audit_orchestrator.info_processer.parser_transactions_matrix import TransactionsMatrixParser

from typing import Dict , Tuple 

import numpy as np

class SpectralAuditor:
    def __init__(self):
        """
        SpectralAuditor aggregates multiple spectral-based network metrics
        into weighted indices across different goods.


        """
        self.spectral_methods = SpectralMethods()


    # ------------------------------------------------------------------
    def _calculate_weighted_metric(self, matrix_dict, weights_dict, metric_func):
        """
        Helper to compute a weighted sum of a given spectral metric
        across multiple goods.
        """
        total = 0.0
        total_weight = 0.0
        for good, matrix in matrix_dict.items():
            if good in weights_dict and weights_dict[good] > 0:
                try:
                    metric = metric_func(matrix)
                    weight = weights_dict[good]
                    total += metric * weight
                    total_weight += weight
                except (ValueError, np.linalg.LinAlgError) as e:
                    print(f"Warning: Could not compute metric for good {good}: {e}")
                    continue
        return total

    # ------------------------------------------------------------------
    def _calculate_dispersion_metric(self, matrix_dict: Dict, metric_func):
        """
        Helper to compute a dispersion-based outlier flag metric.

        Parameters
        ----------
        matrix_dict : dict of {str: np.ndarray}
            Dictionary of goods to adjacency/transition matrices.
        metric_func : callable
            Function from SpectralMethods that computes a scalar metric.

        Returns
        -------
        float
            Max standardized deviation (scale-invariant):
            max |value - mean| / std across goods.
            Returns 0 if <2 goods or std = 0.
        """
        values = []
        for good, matrix in matrix_dict.items():
            try:
                values.append(metric_func(matrix))
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"Warning: Could not compute metric for good {good}: {e}")
                continue

        if len(values) < 2:
            return 0.0

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return 0.0

        deviations = [abs(v - mean_val) / std_val for v in values]
        return max(deviations)

    # ------------------------------------------------------------------
    # Existing weighted metrics
    def calculate_closed_walks_index(self, matrix_dict, weights_dict):
        """
        Computes the weighted sum of closed walks intensity across multiple goods.

        Parameters
        ----------
        matrix_dict : dict of {str: np.ndarray}
            Dictionary of goods to adjacency/transition matrices.
        weights_dict : dict of {str: float}
            Dictionary of goods to weights.

        Returns
        -------
        float
            Weighted sum of closed walks intensity across goods.
        """
        return self._calculate_weighted_metric(
            matrix_dict, weights_dict,
            self.spectral_methods.closed_walk_metric_spectral
        )

    def calculate_entropy_index(self, matrix_dict, weights_dict):
        return self._calculate_weighted_metric(
            matrix_dict, weights_dict,
            self.spectral_methods.entropy_metric
        )

    def calculate_slem_index(self, matrix_dict, weights_dict):
        return self._calculate_weighted_metric(
            matrix_dict, weights_dict,
            self.spectral_methods.slem_metric
        )

    def calculate_circularity_index(self, matrix_dict, weights_dict):
        return self._calculate_weighted_metric(
            matrix_dict, weights_dict,
            self.spectral_methods.cycle_strength_metric
        )

    # ------------------------------------------------------------------
    # New: outlier-sensitive dispersion metrics
    def calculate_closed_walks_dispersion(self, matrix_dict):
        return self._calculate_dispersion_metric(
            matrix_dict, self.spectral_methods.closed_walk_metric_spectral
        )

    def calculate_entropy_dispersion(self, matrix_dict):
        return self._calculate_dispersion_metric(
            matrix_dict, self.spectral_methods.entropy_metric
        )

    def calculate_slem_dispersion(self, matrix_dict):
        return self._calculate_dispersion_metric(
            matrix_dict, self.spectral_methods.slem_metric
        )

    def calculate_circularity_dispersion(self, matrix_dict):
        return self._calculate_dispersion_metric(
            matrix_dict, self.spectral_methods.cycle_strength_metric
        )
