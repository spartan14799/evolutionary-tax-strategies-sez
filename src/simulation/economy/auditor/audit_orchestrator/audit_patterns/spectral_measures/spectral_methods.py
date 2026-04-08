



### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 

### DESCRIPTION: This class encapsulates the logic to calculate spectral measures for matrices 



import numpy as np
import networkx as nx

from src.simulation.economy.auditor.audit_orchestrator.info_processer.parser_transactions_matrix import TransactionsMatrixParser


class SpectralMethods():
    def __init__(self):
        """Collection of spectral graph methods for circularity, entropy, and walk metrics."""
        pass

    def calculate_cycle_strength_spectral_radius(matrix: np.array):
        """
        Compute the cycle strength proxy using the spectral radius.

        Parameters
        ----------
        matrix : np.ndarray
            Square adjacency or transition matrix (weighted or unweighted).

        Returns
        -------
        float
            Circularity index defined as the spectral radius (largest absolute eigenvalue).
            Higher values indicate stronger cyclic structure.
        """
        eigenvalues = np.linalg.eig(matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
        circularity_index = spectral_radius
        return circularity_index

    def calculate_closed_walks(self, matrix: np.array, walk_length: int):
        """
        Count the number of closed walks of a given length in the graph.

        Parameters
        ----------
        matrix : np.ndarray
            Square adjacency or transition matrix.
        walk_length : int
            Length of the closed walk.

        Returns
        -------
        float
            The number of closed walks of the specified length, equal to
            the trace of A^k where A is the input matrix and k is walk_length.
        """
        A_power = np.linalg.matrix_power(matrix, walk_length)
        closed_walks = np.trace(A_power)
        i_k = closed_walks
        return i_k

    def add_teleportation(self, matrix: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """
        Apply PageRank-style teleportation to a matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Square row-stochastic matrix.
        eps : float, optional
            Teleportation probability in (0,1). Default is 1e-3.

        Returns
        -------
        np.ndarray
            Teleported matrix: (1 - eps) * matrix + eps * J,
            where J is the uniform matrix (1/n for all entries).
        """
        n = matrix.shape[0]
        J = np.ones((n, n), dtype=float) / n
        return (1 - eps) * matrix + eps * J

    def slem_metric(self , matrix: np.ndarray, eps: float = 1e-3) -> float:
        """
        Compute the second-largest eigenvalue modulus (SLEM) of the teleported matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Square row-stochastic matrix.
        eps : float, optional
            Teleportation parameter (default=1e-3) to ensure irreducibility.

        Returns
        -------
        float
            1 - SLEM, a scalar in [0, 1). Lower values indicate slower mixing rates.
        """
        P_eps = self.add_teleportation(matrix, eps=eps)
        vals = np.linalg.eigvals(P_eps)
        mod = np.sort(np.abs(vals))[::-1]  # descending by modulus
        return 1 - float(mod[1].real if mod.size > 1 else 0.0)

    def closed_walk_metric_spectral(self, matrix: np.ndarray, alpha: float = 0.85) -> float:
        """
        Compute discounted closed-walk intensity using spectral decomposition.

        Parameters
        ----------
        matrix : np.ndarray
            Square adjacency or transition matrix.
        alpha : float, optional
            Discount factor in (0,1). Default is 0.85.

        Returns
        -------
        float
            Discounted closed-walk metric, normalized by matrix size.
            Excludes the trivial eigenvalue λ=1 to avoid stationary effects.
        """
        vals = np.linalg.eigvals(matrix)
        mask = np.abs(vals - 1.0) > 1e-9
        vals = vals[mask]
        terms = (alpha * vals) / (1.0 - alpha * vals)
        return float(np.real(np.sum(terms)) / matrix.shape[0])

    def entropy_metric(self, matrix: np.ndarray) -> float:
        """
        Compute average row-wise entropy of a normalized weighted matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Square row-stochastic matrix (rows sum to 1).

        Returns
        -------
        float
            Mean Shannon entropy across rows. Higher values indicate
            more uncertainty in transition probabilities.
        """
        entropies = []
        for row in matrix:
            if row.sum() > 0:
                p = row[row > 0]
                entropies.append(-np.sum(p * np.log(p)))
        return np.mean(entropies)

    def cycle_strength_metric(self, matrix: np.ndarray) -> float:
        """
        Compute a circularity/cycle strength metric using non-trivial eigenvalues.

        Parameters
        ----------
        matrix : np.ndarray
            Square adjacency or transition matrix.

        Returns
        -------
        float
            Circularity strength based on geometric terms from eigenvalues,
            excluding the Perron-Frobenius eigenvalue (~1). Returns 0 if
            no non-trivial eigenvalues exist.
        """
        eigen = np.linalg.eigvals(matrix)
        tolerance = 1e-12
        eigen_sorted = np.sort(eigen)[::-1]  # descending order
        non_trivial_eigen = eigen_sorted[1:][np.abs(eigen_sorted[1:]) < 1 - tolerance]

        if len(non_trivial_eigen) == 0:
            return 0.0

        geom_terms = (non_trivial_eigen**2) / (1 - non_trivial_eigen)
        circularity_metric = np.sum(np.real(geom_terms))
        return circularity_metric


