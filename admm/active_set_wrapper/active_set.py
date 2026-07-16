import duneuropy_optimization as dpo
import numpy as np
from scipy.linalg import qr


def active_set_setup(B, omega_value_target, omega_value_non_target, grey_matter_indices, target_indices, target_vector_val):
    """
    Sets up the scliced forward matrix, the sliced omega vector and the sliced target vector for the active set optimization based on the input parameters.
    Parameters:
        B (numpy.ndarray): The forward stimulation matrix (shape: n_grid x n_chan).
        omega_value_target (float): The omega value to be assigned to the target region and non grey matter.
        omega_value_non_target (float): The omega value to be assigned to the grey matter non-target region.
        grey_matter_indices (numpy.ndarray): The indices of the grey matter elements in the mesh.
        target_indices (list): The indices of the target region in the mesh.
        target_vector_val (numpy.ndarray): The target vector values for the target region (3d).
    Returns:
        omega (numpy.ndarray): The weight vector for the constraint in the Wagner optimization (shape: n_grid x 1).
        e_tilde (numpy.ndarray): The target vector (shape: n_grid x 1).
    """

    if target_vector_val.shape[0] != 3:
        raise ValueError(f"Target vector values (target_vector_val) must have 3 elements for the 3D target region. Provided shape: {target_vector_val.shape}")
    
    n_grid = B.shape[0]
    omega = np.ones((n_grid, 1))*omega_value_target
    B_indices = np.sort(np.concatenate((grey_matter_indices*3, grey_matter_indices*3 + 1, grey_matter_indices*3 + 2)))
    B_sliced = B[B_indices,:]

    for idx in grey_matter_indices:
        omega[3*idx, 0] = omega_value_non_target  # all grey matter elements have weight omega_value_non_target
        omega[3*idx+1, 0] = omega_value_non_target
        omega[3*idx+2, 0] = omega_value_non_target
    for idx in target_indices:
        omega[3*idx, 0] = omega_value_target
        omega[3*idx+1, 0] = omega_value_target
        omega[3*idx+2, 0] = omega_value_target

    e_tilde = np.zeros((n_grid, 1))
    for idx in target_indices:
        e_tilde[3*idx, 0] = target_vector_val[0]
        e_tilde[3*idx+1, 0] = target_vector_val[1]
        e_tilde[3*idx+2, 0] = target_vector_val[2]

    e_tilde_sliced = e_tilde[B_indices]
    omega_sliced = omega[B_indices]

    return B_sliced ,omega_sliced, e_tilde_sliced ,omega,e_tilde

def compute_rank_qr(matrix, tol=1e-10):
    """
    Fast rank computation using QR decomposition instead of SVD. As QR is faaster then numpy.linalg.rank SVD.
    """
    if matrix.size == 0:
        return 0
    Q, R = qr(matrix)
    return np.sum(np.abs(np.diag(R)) > tol)

def active_set_evaluation(alpha, beta, B, e_tilde, n_electrodes, omega, epsilon, I_ind = None, I_total = None,verbosity=0):
    """
    Computes the active set solution for the Wagner given stimulation problem matrix and parameters using the Duneuro Optimization for the actual QP solve. Use active_set_setup in advance.
    Parameters:
        alpha (float): alpha of Wagner function.
        beta (float): beta of Wagner function.
        B (numpy.ndarray): The forward stimulation mattrix (slice B in beforehand to optain faster solution).
        e_tilde (numpy.ndarray): The target vector (has the same shape as the columns of B).
        n_electrodes (int): The number of electrode positions (ATTENTION: Should match #cols in B for electrode number restriction use Branch and Bound).
        omega (numpy.ndarray): The weight vector for the constraint in the Wagner optimization (has the same shape as the columns of B).
        epsilon (float): The epsilon parameter for the constraint in the Wagner optimization.
        I_ind (float): Individual current constraint value (leave empty if not used).
        I_total (float): The total current constraint value (leave empty if not used).
        verbosity (int): The verbosity level for the optimization solver (passes directly to dunero solver).
    Returns:
        exit_status (int): The exit status of the optimization solver.
        active_set_solution (numpy.ndarray): The solution to the active set problem (ATTENTION: might not be the full solution vector. Matches the #cols in B).
        optimal_lagrange_eq (numpy.ndarray): The optimal Lagrange multipliers for the equality constraints.
        optimal_lagrange_ineq (numpy.ndarray): The optimal Lagrange multipliers for the inequality constraints.
        opt_func_value (float): The optimal function value.
    """

    if B.shape[1] != n_electrodes:
        raise ValueError(f"Number of electrodes (n_electrodes={n_electrodes}) must match the number of columns in B ({B.shape[1]}). For electrode number restriction use Branch and Bound!")

    Q,r,c,C,b,D = active_set_assembly(alpha, beta, B, e_tilde, n_electrodes, omega, epsilon, I_ind, I_total)
    
    # Solve the QP problem using Duneuro Optimization
    initial_guess, working_set = initial_guess_and_working_set_from_active_constraints(Q,D,c,C,b)

    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count = dpo.solve_quadratic_program_with_initial_guess(Q, r, D, c, C, b, initial_guess, working_set, verbosity)

    # removing artificial variables
    active_set_solution = active_set_solution[:n_electrodes]
    return exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count

def initial_guess_and_working_set_from_active_constraints(Q_tilde,D_tilde,c_tilde,C_tilde,b_tilde,tol=1e-10):
    """
    Constructs initial guess from the active constraints from
        Equality:   D x = c
        Inequality: C x <= b
    """

    Q_tilde = np.asarray(Q_tilde, dtype=np.float64)
    D_tilde = np.asarray(D_tilde, dtype=np.float64)
    C_tilde = np.asarray(C_tilde, dtype=np.float64)

    c_tilde = np.asarray(c_tilde, dtype=np.float64).reshape(-1, 1)
    b_tilde = np.asarray(b_tilde, dtype=np.float64).reshape(-1, 1)

    N = Q_tilde.shape[0]
    n_eq = D_tilde.shape[0]

    x0 = np.zeros((N, 1), dtype=np.float64)

    # Equality feasibility: D x = c
    eq_res = D_tilde @ x0 - c_tilde
    eq_norm = np.max(np.abs(eq_res)) if n_eq > 0 else 0.0

    #print("eq residual max:", eq_norm)

    if eq_norm > tol:
        raise ValueError(
            f"x0 = 0 is not equality-feasible. "
            f"Max equality residual: {eq_norm:.3e}"
        )

    # Inequality feasibility: C x <= b
    ineq_res = C_tilde @ x0 - b_tilde
    max_ineq_violation = np.max(ineq_res)
    max_ineq_violation_idx= np.argmax(ineq_res)

    if max_ineq_violation > tol:
        raise ValueError(
            f"x0 = 0 is not inequality-feasible. "
            f"Max violation: {max_ineq_violation:.3e}"
            f" at index {max_ineq_violation_idx}, "
            f" b_tilde at that index: {b_tilde[max_ineq_violation_idx,0]:.3e},"
        )

    # Start with all equality constraints
    initial_working_set = list(range(n_eq))

    W_rows = []

    for k in range(n_eq):
        W_rows.append(D_tilde[k:k + 1, :])

    if W_rows:
        W = np.vstack(W_rows)
        current_rank = compute_rank_qr(W, tol)
    else:
        current_rank = 0

    # Active inequalities at x0:
    # C_i x0 == b_i  <=>  C_i x0 - b_i == 0
    active_ineq_rows = np.where(np.abs(ineq_res.ravel()) <= tol)[0]

    # fülle working set auf mit aktiven Ungleichungen, solange sie den Rang erhöhen, bis die benötigte Anzahl (N = Anzahl Variablen) erreicht ist
    for j in active_ineq_rows:
        candidate_row = C_tilde[j:j + 1, :]

        if W_rows:
            W_candidate = np.vstack(W_rows + [candidate_row])
        else:
            W_candidate = candidate_row

        candidate_rank = compute_rank_qr(W_candidate, tol)

        if candidate_rank > current_rank:
            W_rows.append(candidate_row)
            initial_working_set.append(n_eq + j)
            current_rank = candidate_rank
        else:
            #print(f"Skipping inequality {j} for working set (rank {candidate_rank} <= {current_rank})")
            pass

        if len(initial_working_set) == N:
            break

    W = np.vstack(W_rows)


    return np.ascontiguousarray(x0, dtype=np.float64), initial_working_set

def active_set_assembly_with_precomp(Be_pre,C_pre,eps_omega_pre_2times,alpha,beta,n_electrodes,I_ind=None,I_total=None, remove_o_e_constraint=False):
    n = n_electrodes

    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = 2 * alpha * np.eye(n)

    r = np.empty(2 * n)
    r[:n] = -Be_pre
    r[n:] = beta

    c = np.array([0.0])

    # If remove_o_e_constraint=True, we remove the large omega/epsilon field constraints:
    # [ B, 0] x <= eps/omega
    # [-B, 0] x <= eps/omega
    if remove_o_e_constraint:
        field_rows = 0
    else:
        field_rows = C_pre.shape[0]

    extra_rows = 2 * n  # constraints for y >= |I|

    if I_total is not None:
        extra_rows += 1

    if I_ind is not None:
        extra_rows += 2 * n

    C = np.zeros((field_rows + extra_rows, 2 * n))
    b = np.empty(field_rows + extra_rows)

    row = 0

    if not remove_o_e_constraint:
        C[row:row + C_pre.shape[0], :] = C_pre
        b[row:row + eps_omega_pre_2times.shape[0]] = eps_omega_pre_2times
        row += C_pre.shape[0]

    I = np.eye(n)

    # I - y <= 0
    C[row:row+n, :n] = I
    C[row:row+n, n:] = -I
    b[row:row+n] = 0.0
    row += n

    # -I - y <= 0
    C[row:row+n, :n] = -I
    C[row:row+n, n:] = -I
    b[row:row+n] = 0.0
    row += n

    # sum(y) <= I_total
    if I_total is not None:
        C[row, n:] = 1.0
        b[row] = I_total
        row += 1

    # I <= I_ind and -I <= I_ind
    if I_ind:
        C[row:row+n, :n] = I
        b[row:row+n] = I_ind
        row += n

        C[row:row+n, :n] = -I
        b[row:row+n] = I_ind
        row += n

    D = np.zeros((1, 2 * n))
    D[0, :n] = 1.0

    return Q, r, c, C, b, D
    
    # Q = np.zeros((2 * n, 2 * n))
    # Q[:n, :n] = 2 * alpha * np.eye(n)

    # r = np.empty(2 * n)
    # r[:n] = -Be_pre
    # r[n:] = beta

    # c = np.array([0.0])

    # extra_rows = 2 * n
    # if I_total is not None:
    #     extra_rows += 1
    # if I_ind:
    #     extra_rows += 2 * n

    # C = np.zeros((C_pre.shape[0] + extra_rows, 2 * n))
    # b = np.empty(eps_omega_pre_2times.shape[0] + extra_rows)

    # row = 0
    # C[row:row + C_pre.shape[0], :] = C_pre
    # b[row:row + eps_omega_pre_2times.shape[0]] = eps_omega_pre_2times
    # row += C_pre.shape[0]

    # I = np.eye(n)

    # C[row:row+n, :n] = I
    # C[row:row+n, n:] = -I
    # b[row:row+n] = 0.0
    # row += n

    # C[row:row+n, :n] = -I
    # C[row:row+n, n:] = -I
    # b[row:row+n] = 0.0
    # row += n

    # if I_total is not None:
    #     C[row, n:] = 1.0
    #     b[row] = I_total
    #     row += 1

    # if I_ind:
    #     C[row:row+n, :n] = I
    #     b[row:row+n] = I_ind
    #     row += n

    #     C[row:row+n, :n] = -I
    #     b[row:row+n] = I_ind
    #     row += n

    # D = np.zeros((1, 2 * n))
    # D[0, :n] = 1.0

    # return Q, r, c, C, b, D
    # # # assemble matrices for active set implementation
    # # Q_elec = 2 *alpha* np.eye(n_electrodes)
    # # Q = np.block([[Q_elec, np.zeros((n_electrodes, n_electrodes))], [np.zeros((n_electrodes, n_electrodes)), np.zeros((n_electrodes, n_electrodes))]])
    # # r = np.concatenate((-1*Be_pre, beta * np.ones(n_electrodes)))
    # # c = np.array([0])
    # # C = np.vstack((C_pre,
    # #                np.block([[np.eye(n_electrodes), -1*np.eye(n_electrodes)],
    # #                          [-1* np.eye(n_electrodes), -1* np.eye(n_electrodes)]])))
    # # b = np.concatenate((eps_omega_pre_2times,np.zeros(2*n_electrodes)))
    # # if I_total is not None:
    # #     C_I_total = np.block([
    # #         np.zeros((1,n_electrodes)), np.ones((1,n_electrodes))
    # #     ])
    # #     b_I_total = I_total * np.ones(1)
    # #     C = np.vstack((C, C_I_total))
    # #     b = np.concatenate((b, b_I_total))
    # # if I_ind:
    # #     C_I_max = np.block([
    # #         [np.eye(n_electrodes), np.zeros((n_electrodes, n_electrodes))],
    # #         [-np.eye(n_electrodes), np.zeros((n_electrodes,n_electrodes))]
    # #     ])

    # #     b_Imax = I_ind * np.ones(2 * n_electrodes)

    # #     C = np.vstack((C, C_I_max))
    # #     b = np.concatenate((b, b_Imax))
    # # D = np.concatenate((np.ones(n_electrodes), np.zeros(n_electrodes)))  
    # # D = D[None,:]
    # # return Q,r,c,C,b,D

def active_set_assembly(alpha, beta, B, e_tilde, n_electrodes, omega, epsilon, I_ind = False, I_total = None):
    """
    Central function for the DUNEUROPY Active set call. Assembles the matrices according to the Input s.t. the Solver can be called (Ref. Section 5.3 and more in Thesis). 
    Constructs matrices and vectors s.t. the extended Wagner et al approach 
    
    min_I \int <BI,e_tilde> + alpha <I,I> + beta ||I||_1
    s.t. 
        omega|BI| <= epsilon
        1^t I = 0
        ||I||_1 <= I_total
        |I_i| \leq I_ind for all i
        
    is brought into standard QP form 

    min_I I^QI + r^tI
    s.t. 
        CI + b <= 0
        DI + c = 0

    s.t. it can be solved with the DUNEURO Active Set Solver.

    Parameter:
        - alpha: the alpha value of Wagner functional
        - beta: the beta value of the Wagner functional
        - B: the discretized forward operator
        - e_tilde: the long target vector (same shape as cols of B)
        - n_electrodes: B.shape[1]
        - omega: the omega vector for the side constraint
        - epsilon: the epsilon for the side constraint
        - I_ind (optional): the individual current restriction value
        - I_total (optional): the total current restriction value
    """
    # Q assembly
    Q_elec = 2 *alpha* np.eye(n_electrodes)
    Q = np.block([[Q_elec, np.zeros((n_electrodes, n_electrodes))], [np.zeros((n_electrodes, n_electrodes)), np.zeros((n_electrodes, n_electrodes))]])

    # r assembly
    Be = (B.T@e_tilde.flatten())
    r = np.concatenate((-1*Be, beta * np.ones(n_electrodes)))

    # C and b assembly
    C = np.block([[B, np.zeros((B.shape[0], n_electrodes))], 
                  [-B, np.zeros((B.shape[0],n_electrodes))], 
                  [np.eye(n_electrodes), -1* np.eye(n_electrodes)], 
                  [-1* np.eye(n_electrodes), -1* np.eye(n_electrodes) 
                   ]])
    epsilon_omega_vector = epsilon/omega.flatten() 
    b = np.concatenate((epsilon_omega_vector, epsilon_omega_vector, np.zeros(2*n_electrodes)))

    # adaptions for I_total and I_ind
    if I_total is not None:
        C_I_total = np.block([
            np.zeros((1,n_electrodes)), np.ones((1,n_electrodes))
        ])
        b_I_total = I_total * np.ones(1)
        C = np.vstack((C, C_I_total))
        b = np.concatenate((b, b_I_total))
    if I_ind:
        C_I_max = np.block([
            [np.eye(n_electrodes), np.zeros((n_electrodes, n_electrodes))],
            [-np.eye(n_electrodes), np.zeros((n_electrodes,n_electrodes))]
        ])

        b_Imax = I_ind * np.ones(2 * n_electrodes)

        C = np.vstack((C, C_I_max))
        b = np.concatenate((b, b_Imax))


    # D and c assembly
    c = np.array([0])
    D = np.concatenate((np.ones(n_electrodes), np.zeros(n_electrodes)))  
    D = D[None,:]
    return Q,r,c,C,b,D
