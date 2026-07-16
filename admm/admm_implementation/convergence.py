"""Convergence criteria for ADMM algorithm."""

import numpy as np


def residual_convergence(norm1, norm2, norm3, rel_tol, abs_tol, primal_res, dual_res, verbosity=0):
    """
    Check convergence based on primal and dual residuals. According to Boyd ADMM paper.

    tol_primal = sqrt(p)*abs_tol + rel_tol*max(||Ax||, ||Bz||, ||c||)
    tol_dual = sqrt(n)*abs_tol + rel_tol*norm(A'*y)
    
    Parameters:
    norm1 (float): ||Ax|| 
    norm2 (float): ||Bz|| 
    norm3 (float): ||A'*y|| 
    rel_tol (float): Relative tolerance.
    abs_tol (float): Absolute tolerance.

    Returns:
    bool: True if both residuals are below their respective tolerances, False otherwise.
    """
    r = np.asarray(primal_res)
    s = np.asarray(dual_res)

    tol_primal = np.sqrt(r.size) * abs_tol + rel_tol * max(norm1, norm2)
    tol_dual = np.sqrt(s.size) * abs_tol + rel_tol * np.linalg.norm(norm3)

    if verbosity>=1:
        print(f"Tolerances: Primal = {tol_primal:.3e}, Dual = {tol_dual:.3e}")

    return np.linalg.norm(primal_res) < tol_primal and np.linalg.norm(dual_res) < tol_dual

def wagner_convergence(I, I_prev, tol):
    """
    Check convergence based on Wagner's criterion.

    ||I - I_prev|| < tol
    
    Parameters:
    I (numpy.vec): Current estimate.
    I_prev (numpy.vec): Previous estimate.
    tol (float): The tolerance for convergence.

    Returns:
    bool: True if the change in estimates is below the tolerance, False otherwise.
    """
    return np.linalg.norm(I - I_prev) < tol