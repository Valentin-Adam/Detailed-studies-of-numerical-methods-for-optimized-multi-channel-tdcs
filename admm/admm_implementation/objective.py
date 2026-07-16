"""Objective functional for ADMM optimization."""

import numpy as np


def objective_functional(I, B, e_tilde, alpha, beta):
    """
    Compute the objective functional value for given current injection pattern.
    
    Parameters
    I : Current injection pattern.
    B : Lead field matrix.
    e_tilde :  Desired electric field.
    alpha : Weight for the L2 norm of the current injection.
    beta : Weight for the L1 norm of the current injection.

    Returns
        The value of the objective functional.
    """
    current_density = B @ I
    l2_penalty = alpha * np.linalg.norm(I)**2
    l1_penalty = beta * np.sum(np.abs(I))
    return - np.dot(current_density, e_tilde) + l2_penalty + l1_penalty
