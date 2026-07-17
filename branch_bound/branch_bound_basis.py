import time
import numpy as np 
import duneuropy_optimization as dpo
from admm.active_set_wrapper.active_set import active_set_evaluation, active_set_setup, active_set_assembly, initial_guess_and_working_set_from_active_constraints, initial_guess_and_working_set_from_active_constraints, active_set_assembly_with_precomp

def one_pos_one_neg_then_abs_picking_rule(solution_vector, electrode_set, n_usable_electrodes):
    """
    Picks unassigned electrodes for the upper bound construction

    Rule:
    - ensure at least one positive active electrode if possible
    - ensure at least one negative active electrode if possible
    - fill the remaining slots by largest absolute current values
    """

    active_indices = np.where(electrode_set == 1)[0]
    unassigned_indices = np.where(electrode_set == 0)[0]

    n_active = len(active_indices)
    n_to_choose = n_usable_electrodes - n_active

    if n_to_choose <= 0:
        return np.array([], dtype=int)

    n_to_choose = min(n_to_choose, len(unassigned_indices))

    chosen = []

    active_values = solution_vector[active_indices]

    has_positive = np.any(active_values > 0)
    has_negative = np.any(active_values < 0)

    unassigned_values = solution_vector[unassigned_indices]

    # If no positive active electrode exists yet choose strongest positive unassigned one
    if not has_positive and len(chosen) < n_to_choose:
        positive_local = np.where(unassigned_values > 0)[0]

        if len(positive_local) > 0:
            best_pos_local = positive_local[np.argmax(unassigned_values[positive_local])]
            chosen.append(unassigned_indices[best_pos_local])

    # If no negative active electrode exists yet choose strongest negative unassigned one
    if not has_negative and len(chosen) < n_to_choose:
        negative_local = np.where(unassigned_values < 0)[0]

        if len(negative_local) > 0:
            best_neg_local = negative_local[np.argmin(unassigned_values[negative_local])]
            candidate = unassigned_indices[best_neg_local]

            if candidate not in chosen:
                chosen.append(candidate)

    # Fill remaining slots by largest absolute current among unassigned electrodes
    remaining_slots = n_to_choose - len(chosen)

    if remaining_slots > 0:
        already_chosen = np.array(chosen, dtype=int)

        remaining_unassigned = np.setdiff1d(
            unassigned_indices,
            already_chosen,
            assume_unique=False,
        )

        remaining_values = solution_vector[remaining_unassigned]

        order = np.argsort(np.abs(remaining_values))[::-1]
        fill_indices = remaining_unassigned[order[:remaining_slots]]

        chosen.extend(fill_indices.tolist())

    return np.sort(np.array(chosen, dtype=int))

def advanced_ub_picking_rule(solution_vector, electrode_set, n_usable_electrodes):
    """
    Picks unassigned electrodes such that positive/negative currents are balanced,
    but never picks more than the remaining number of electrode slots.
    """
    print("Using advanced upper bound picking rule...")

    if n_usable_electrodes % 2 != 0:
        raise ValueError("n_usable_electrodes must be even for the balanced picking rule.")

    active_indices = np.where(electrode_set == 1)[0]
    unassigned_indices = np.where(electrode_set == 0)[0]

    n_active = len(active_indices)
    n_to_choose = n_usable_electrodes - n_active

    if n_to_choose <= 0:
        return np.array([], dtype=int)

    n_to_choose = min(n_to_choose, len(unassigned_indices))

    active_values = solution_vector[active_indices]

    n_pos_active = np.sum(active_values > 0)
    n_neg_active = np.sum(active_values < 0)

    target_pos = n_usable_electrodes // 2
    target_neg = n_usable_electrodes // 2

    desired_pos = max(0, target_pos - n_pos_active)
    desired_neg = max(0, target_neg - n_neg_active)

    desired_total = desired_pos + desired_neg

    if desired_total > n_to_choose:
        # Reduce the side that needs more electrodes less urgently
        # This keeps the rule balanced but respects the hard cardinality limit
        overflow = desired_total - n_to_choose

        while overflow > 0:
            if desired_pos >= desired_neg and desired_pos > 0:
                desired_pos -= 1
            elif desired_neg > 0:
                desired_neg -= 1
            else:
                break
            overflow -= 1

    unassigned_values = solution_vector[unassigned_indices]

    order = np.argsort(unassigned_values)

    neg_local_idx = order[:desired_neg]
    pos_local_idx = order[order.shape[0] - desired_pos:] if desired_pos > 0 else np.array([], dtype=int)

    chosen_local_idx = np.concatenate([neg_local_idx, pos_local_idx])

    chosen_global_idx = np.unique(unassigned_indices[chosen_local_idx])

    # If we still have free slots, fill by largest absolute current.
    if len(chosen_global_idx) < n_to_choose:
        remaining_slots = n_to_choose - len(chosen_global_idx)

        remaining_unassigned = np.setdiff1d(
            unassigned_indices,
            chosen_global_idx,
            assume_unique=False,
        )

        remaining_values = solution_vector[remaining_unassigned]
        abs_order = np.argsort(np.abs(remaining_values))[::-1]

        fill = remaining_unassigned[abs_order[:remaining_slots]]
        chosen_global_idx = np.concatenate([chosen_global_idx, fill])

    chosen_global_idx = np.unique(chosen_global_idx)

    if len(chosen_global_idx) > n_to_choose:
        values = solution_vector[chosen_global_idx]
        keep = np.argsort(np.abs(values))[::-1][:n_to_choose]
        chosen_global_idx = chosen_global_idx[keep]

    return np.sort(chosen_global_idx.astype(int))
    
def lower_bound_slicing_function(electrode_set, alpha, beta, B, e_tilde, omega, epsilon, n_usable_electrodes, I_ind, I_total, precomp = None,remove_o_e_constraint=False):
    """
    Assembles the matrices for the lower bound function s.t. active set evalueation can be done. Precomputation and remove omega epsilon constraint functionality not functionable/not really developed
    """
    
    active_electrodes_idx = np.where(electrode_set == 1)[0]  # Get the indices of active electrodes
    unassigned_electrodes_idx = np.where(electrode_set == 0)[0]  # Get the indices of unassigned electrodes
    ae = len(active_electrodes_idx)
    ue = len(unassigned_electrodes_idx)

    # Slice B s.t. just active and unassigned electrodes are used 
    active_and_unassigned = np.concatenate((active_electrodes_idx, unassigned_electrodes_idx)) # No Sorting here! as otherwise np.arange would not work 
    B_sliced = B[:,active_and_unassigned]
    local_unassigned_idx = np.arange(ae, ae + ue)  # Local indices of unassigned electrodes in the sliced matrices


    if precomp is not None:
        C_cols = np.concatenate((active_and_unassigned, precomp["n_electrodes_total"] + active_and_unassigned))
        C_pre = precomp["C"][:,C_cols]
        Q_bar, r_bar, c_bar, C_bar, b_bar, D_bar = active_set_assembly_with_precomp(precomp["Be"][active_and_unassigned],C_pre,precomp["e_o_2x"],alpha,beta,ae+ue,I_ind,I_total, remove_o_e_constraint=remove_o_e_constraint)
    else:
        Q_bar, r_bar, c_bar, C_bar, b_bar, D_bar = active_set_assembly(alpha, beta, B_sliced, e_tilde, B_sliced.shape[1], omega, epsilon, I_ind, I_total)
    

    Q_tilde = np.block([[Q_bar, np.zeros((2*(ae + ue), ue))],
                        [np.zeros((ue,2*( ae + ue))), np.zeros((ue, ue))]])
    r_tilde = np.concatenate((r_bar, np.zeros(ue)))
    c_tilde = c_bar

    # Add the charactaristical extra relaxed version of the 0-norm constraint. Ref Thesis lower bound function (5.31)
    if I_ind is not None:
        E_u = np.zeros((ue, 2 * (ae+ue)))
        E_u[np.arange(ue), local_unassigned_idx] = 1.0
        C_tilde = np.block([[C_bar, np.zeros((C_bar.shape[0], ue))],
                        [np.zeros((1,2*(ae+ue))), np.ones((1,ue))],
                        [E_u, -1 * np.eye(ue)],
                        [-1 * E_u, -1 * np.eye(ue)],
                        ])
        b_tilde = np.concatenate((b_bar, [ (n_usable_electrodes - ae) * I_ind], np.zeros(2*ue)))
    else:
        C_tilde = np.block([[C_bar, np.zeros((C_bar.shape[0], ue))],
                        ])
        b_tilde = b_bar
    D_bar = D_bar.flatten()
    D_tilde = np.concatenate((D_bar, np.zeros(ue)))
    D_tilde = D_tilde[None,:]
    return Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde

def lower_bound(electrode_set,  alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity=0,precomp=None, timing=None,remove_o_e_constraint=False):
    """
    Computes the lower bound value and its solution for the lower bound function ref (5.31) in Thesis
    """

    # slice the matrices based on the current electrode set and make them usable for the active set implementation
    t0 = time.perf_counter()
    Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde = lower_bound_slicing_function(electrode_set,  alpha, beta, B, e_tilde, omega, epsilon, n_usable_electrodes, I_ind, I_total,precomp=precomp, remove_o_e_constraint=remove_o_e_constraint)
    if timing is not None:
        timing["lower_assembly"] += time.perf_counter() - t0

    # Compute an initial guess and working set for solver
    t0 = time.perf_counter()
    initial_guess, initial_working_set = initial_guess_and_working_set_from_active_constraints(Q_tilde, D_tilde, c_tilde, C_tilde, b_tilde)
    if timing is not None:
        timing["lower_initial_guess"] += time.perf_counter() - t0
    t0 = time.perf_counter()

    # Active set call
    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iterations = dpo.solve_quadratic_program_with_initial_guess(Q_tilde, r_tilde, D_tilde, c_tilde, C_tilde, b_tilde, initial_guess, initial_working_set,verbosity)
    
    # Timinig issues for debugging
    if timing is not None:
        timing["lower_solver"] += time.perf_counter() - t0
        timing["lower_total"] += time.perf_counter() - t_total
        timing["lower_calls"] += 1
    return opt_func_value, active_set_solution  

def upper_bound_slicing_function(electrode_set, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total,precomp=None, remove_o_e_constraint=False):
    """
    Assembles the matrices for the upper bound function s.t. active set evalueation can be done. Precomputation and remove omega epsilon constraint functionality not functionable/not really developed
    """
    active_electrodes_idx = np.where(electrode_set == 1)[0]  # Get the indices of active electrodes
    unassigned_electrodes_idx = np.where(electrode_set == 0)[0]  # Get the indices of unassigned electrodes
    ae = len(active_electrodes_idx)
    ue = len(unassigned_electrodes_idx)

    active_and_unassigned = np.concatenate((active_electrodes_idx, unassigned_electrodes_idx))  # Combine active and unassigned electrode indices
    active_and_unassigned = np.sort(active_and_unassigned)

    if precomp is not None:
        C_cols = np.concatenate((active_and_unassigned, precomp["n_electrodes_total"] + active_and_unassigned))
        C_pre = precomp["C"][:,C_cols]
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde = active_set_assembly_with_precomp(precomp["Be"][active_and_unassigned],C_pre,precomp["e_o_2x"],alpha,beta,ae+ue,I_ind,I_total, remove_o_e_constraint=remove_o_e_constraint)
    else:
        B_sliced = B[:,active_and_unassigned]
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde = active_set_assembly(alpha, beta, B_sliced, e_tilde, B_sliced.shape[1], omega, epsilon, I_ind, I_total
                                                                                   )

    return Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde, active_and_unassigned

def upper_bound(electrode_set,  alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity=0, advanced_picking_rule=False, precomp=None,     timing=None, remove_o_e_constraint=False):
    """
    Computes the upper bound value and its solution for the upper bound function ref (5.33) in Thesis
    """

    t_total = time.perf_counter()
    upper_bound_set = electrode_set.copy()
    # slice the matrices based on the current electrode set and make them usable for the active set implementation
    if precomp is not None:
        t0 = time.perf_counter()
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde, active_and_unassigned = upper_bound_slicing_function(electrode_set, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, precomp, remove_o_e_constraint=remove_o_e_constraint)
        if timing is not None:
            timing["upper_assembly"] += time.perf_counter() - t0
    else:
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde, active_and_unassigned = upper_bound_slicing_function(electrode_set, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total)
    
    # compute initial guess and initial working set for solver 
    initial_guess = np.zeros(Q_tilde.shape[1], dtype=np.float64)
    t0 = time.perf_counter()
    initial_guess, initial_working_set = initial_guess_and_working_set_from_active_constraints(Q_tilde, D_tilde, c_tilde, C_tilde, b_tilde)
    if timing is not None:
        timing["upper_initial_guess"] += time.perf_counter() - t0

    active_set_indices = np.where(upper_bound_set == 1)[0]  # Get the indices of active electrodes
    unassigned_set_indices = np.where(upper_bound_set == 0)[0]  # Get the indices of unassigned electrodes

    # First active set call
    t0 = time.perf_counter()
    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iterations = dpo.solve_quadratic_program_with_initial_guess(Q_tilde, r_tilde, D_tilde, c_tilde, C_tilde, b_tilde,initial_guess, initial_working_set,verbosity)
    if timing is not None:
        timing["upper_solver"] += time.perf_counter() - t0

    # sort the unaassigned electrodes based on their value in active_set_solution 
    # here the heruistic picking rule in the upper bound function can be decided
    full_active_set_solution = np.zeros(len(electrode_set))
    full_active_set_solution[active_and_unassigned] = active_set_solution[:len(active_and_unassigned)]

    if advanced_picking_rule == 'Half':
        max_mindices = advanced_ub_picking_rule(full_active_set_solution, electrode_set, n_usable_electrodes)
    elif advanced_picking_rule == 'One':
        max_mindices = one_pos_one_neg_then_abs_picking_rule(full_active_set_solution, electrode_set, n_usable_electrodes)
    else:
        ae = len(active_set_indices)
        ue = len(unassigned_set_indices)
        local_unassigned_I_idx = np.arange(ae, ae + ue)
        order = np.argsort(np.abs(active_set_solution[local_unassigned_I_idx]))[::-1]
        n_to_choose = n_usable_electrodes - ae
        n_to_choose = max(0, min(n_to_choose, ue))
        max_mindices = unassigned_set_indices[order[:n_to_choose]]

    upper_bound_set[unassigned_set_indices] = -1
    upper_bound_set[max_mindices] = 1
    n_active_after_picking = np.sum(upper_bound_set == 1)

    if n_active_after_picking != n_usable_electrodes:
        raise RuntimeError(
            f"Upper-bound picking produced wrong number of active electrodes: "
            f"expected={n_usable_electrodes}, got={n_active_after_picking}, "
            f"already_active_before={len(active_set_indices)}, "
            f"picked={len(max_mindices)}, "
            f"max_mindices={max_mindices}, "
            f"upper_bound_set={upper_bound_set}"
        )
                
    # Again slice the matrices now just with the newly selected electrodes in upper_bound_set
    if precomp is not None:
        t0 = time.perf_counter()
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde, active_and_unassigned = upper_bound_slicing_function(upper_bound_set, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, precomp, remove_o_e_constraint=remove_o_e_constraint)
        if timing is not None:
            timing["upper_assembly"] += time.perf_counter() - t0   
    else:
        Q_tilde, r_tilde, c_tilde, C_tilde, b_tilde, D_tilde, active_and_unassigned = upper_bound_slicing_function(upper_bound_set, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total)
    
    # Second Active Set Call during Upper bound function
    t0 = time.perf_counter()
    initial_guess, initial_working_set = initial_guess_and_working_set_from_active_constraints(Q_tilde, D_tilde, c_tilde, C_tilde, b_tilde)
    if timing is not None:
        timing["upper_initial_guess"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iterations = dpo.solve_quadratic_program_with_initial_guess(Q_tilde, r_tilde, D_tilde, c_tilde, C_tilde, b_tilde,initial_guess, initial_working_set,verbosity)
    if timing is not None:
        timing["upper_solver"] += time.perf_counter() - t0
        timing["upper_total"] += time.perf_counter() - t_total
        timing["upper_calls"] += 1
    full_upper_bound_solution = np.zeros_like(electrode_set, dtype=float)

    full_upper_bound_solution = np.asarray(full_upper_bound_solution, dtype=np.float64).ravel()
    active_indices = np.where(upper_bound_set == 1)[0]
    full_upper_bound_solution[active_indices] = active_set_solution[:len(active_indices)]

    upper_bound_solution = active_set_solution[:len(active_indices)]   

    return opt_func_value, upper_bound_solution, full_upper_bound_solution, upper_bound_set

def split_electrode_set(full_upper_bound_solution, electrode_set):
    """ 
    Splitting the electrode set based on precomputed upper bound solution . Ref Algorithm 8 in the thesis.
    """
    print(f'Splitting electrode set...')
    
    unassigned_indices = np.where(electrode_set == 0)[0]  # Get the indices of unassigned electrodes
    local_pos = np.argmax(np.abs(full_upper_bound_solution[unassigned_indices]))  # Get the index of the unassigned electrode with the largest absolute value in the optimal solution
    max_unassigned_idx = unassigned_indices[local_pos]  # Get the global index of the selected unassigned electrode

    electrode_set_copy1 = electrode_set.copy()
    electrode_set_copy2 = electrode_set.copy()

    electrode_set_copy1[max_unassigned_idx] = 1  # Set the selected electrode to active in the first copy
    electrode_set_copy2[max_unassigned_idx] = -1  # Set the selected electrode to inactive in the second copy

    return [electrode_set_copy1, electrode_set_copy2]  

def active_set_solution_heuristic(alpha, beta, B, target_indices, target_vector, grey_matter_indices, omega_value,omega_value_non_target, epsilon, n_usable_electrodes,I_ind, I_total, heuristic='Half', verbosity=0):
    """
    Use the standard active set method without any cardinality constraint and the nuse heuristic to pick the active electrodes and then optimze on them again.

    Heuristic solution for the number of electrode restirction:"
    "Half":
       - take the n_usuable_electrodes/2 largest and n_usable_electrodes/2 smallest values "
       from the active set solution of the uncronstrained (sense of number of usable electrodes) problem and set the corresponding electrodes as active (rest not active)"
       - compute an active set solution for these active electrodes"
    "One":
      - take the n_usuable_electrodes largest positive values from the active set solution of the uncronstrained (sense of number of usable electrodes) problem and set the corresponding electrodes as active (rest not active)"
        - ensure that at least one poisitive and one negative electrode is active"
    "Simple":
       - take the n_usuable_electrodes largest absolute values from the active set solution of the uncronstrained (sense of number of usable electrodes) problem and set the corresponding electrodes as active (rest not active)"
    """


    # check if n_usable_electrodes is even,if not raise error
    if n_usable_electrodes % 2 != 0:
        raise ValueError("n_usable_electrodes must be even for the heuristic solution.")
    
    B_sliced, omega_sliced, e_tilde_sliced, omega,e_tilde = active_set_setup(B,omega_value, omega_value_non_target, grey_matter_indices,target_indices,target_vector)
    print(f'Starting Active Set algorithm for initial unconstrained solution.')
    t_start_iteration = time.time()
    
    exit_status, resulting_I, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count = active_set_evaluation(alpha,beta,B_sliced,e_tilde_sliced, B.shape[1],omega_sliced,epsilon, I_ind = I_ind, I_total=I_total,verbosity=verbosity)
    t_end_iteration = time.time()
    print(f"[TIME] Active Set solving: {t_end_iteration - t_start_iteration:.3f}s")

    # Remove the auxiliary variable from the soltuion
    full_solution_first = resulting_I

    if heuristic == 'Half':
        # pick to n/2 biggest positive and n/2 smallest negative values:
        ordered_electrodes = np.argsort(full_solution_first)
        n_pos = n_usable_electrodes // 2
        n_neg = n_usable_electrodes - n_pos
        selected_pos = ordered_electrodes[-n_pos:]
        selected_neg = ordered_electrodes[:n_neg]
        selected_electrodes = np.concatenate((selected_pos, selected_neg))
        selected_electrodes = np.sort(selected_electrodes)
    elif heuristic == 'One':
        ordered_electrodes = np.argsort(full_solution_first)
        selected_electrodes = np.array([ordered_electrodes[-1], ordered_electrodes[0]])  # pick the largest positive and the smallest negative
        remaining_electrodes = ordered_electrodes[1:-1]
        remaining_solution = full_solution_first[remaining_electrodes]

        n_remaining = n_usable_electrodes - 2
        ordered_remaining_local = np.argsort(np.abs(remaining_solution))
        selected_remaining = remaining_electrodes[ordered_remaining_local[-n_remaining:]]

        selected_electrodes = np.concatenate((selected_electrodes, selected_remaining))
        selected_electrodes = np.sort(selected_electrodes)
    elif heuristic == 'Simple':
        # pick the n biggest absolute values:
        ordered_electrodes = np.argsort(np.abs(full_solution_first))
        selected_electrodes = ordered_electrodes[-n_usable_electrodes:]
        selected_electrodes = np.sort(selected_electrodes) 

    # slice the matrices and compute corresponding solution for the selected electrodes:
    B_sliced = B[:,selected_electrodes]
    Q,r,c,C,b,D = active_set_assembly(alpha, beta, B_sliced, e_tilde, B_sliced.shape[1], omega, epsilon, I_ind, I_total)
    initial_guess, initial_working_set = initial_guess_and_working_set_from_active_constraints(Q, D, c, C, b)
    print(f'Starting Active Set algorithm with {n_usable_electrodes} electrodes.')
    t_start_iteration = time.time()
    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count_second = dpo.solve_quadratic_program_with_initial_guess(Q, r, D, c, C, b, initial_guess, initial_working_set,verbosity)
    t_end_iteration = time.time()
    print(f"[TIME] Active Set solving: {t_end_iteration - t_start_iteration:.3f}s")
    # map the solution back to the full electrode space
    full_solution = np.zeros(B.shape[1], dtype=np.float64)
    full_solution[selected_electrodes] = active_set_solution[:len(selected_electrodes)]

    return opt_func_value, full_solution, selected_electrodes, full_solution_first

def precompute_full_active_set(B,e_tilde,omega,epsilon):
    """
    Attempt for precomputing things. Not Used."""
    return {
        "n_electrodes_total":B.shape[1],
        "Be": B.T @ e_tilde.flatten(),
        "e_o_2x": np.concatenate((epsilon / omega.flatten(), epsilon/omega.flatten())),
        "C": np.block([[B, np.zeros_like(B)], 
                  [-B, np.zeros_like(B)]] )
    }

def branch_and_bound_basis(n_electrodes, n_usable_electrodes, tolerance, k_max, alpha, beta, B_sliced, omega_sliced, epsilon, e_tilde_sliced, I_ind, I_total, verbosity=0, advanced_picking_rule=False, precompute = False, remove_o_e_constraint=False):
    """
    Branch and bound algorithm from Saturnino et al. (2019) for finding the optimal stimulation pattern.

    Parameters:
    n_electrodes (int): The number of total available electrodes.
    n_usable_electrodes (int): The number of electrodes to be used in the stimulation pattern.
    tolerance (float): The error tolerance for the algorithm.
    k_max (int): The maximum number of iterations for the algorithm.
    alpha (float): The alpha of objective function.
    beta (float): The beta of objective function.
    B_sliced (np.array): The lead field matrix representing the relationship between electrode activations and the resulting electric field in the brain.
    omega_sliced (np.array): The omega in the constraint.
    epsilon (float): The epsilon in the constraint.
    e_tilde_sliced (np.array): The target vector we want to achieve with the stimulation pattern.
    I_total (np.array): The total current constraint for the stimulation pattern.
    I_ind (np.array): The individual current constraint for each electrode.
    verbosity (int): The level of verbosity for printing debug information (0 for no output, higher values for more detailed output).

    Returns:
    np.array: The optimal stimulation pattern found by the algorithm.

    Description:
    electrode_set contains 0 for unassigned electrodes, 1 for active electrodes, -1 for inactive electrodes
    """
    #Assuming sliced inputs
    B = B_sliced
    e_tilde = e_tilde_sliced
    omega = omega_sliced


    print('Starting branch and bound algorithm...')
    electrode_set = np.zeros(n_electrodes)  # Initialize the electrode set (all electrodes are initially unassigned)

    k = 0
    timing = {
        "lower_total": 0.0,
        "lower_assembly": 0.0,
        "lower_initial_guess": 0.0,
        "lower_solver": 0.0,
        "lower_calls": 0,

        "upper_total": 0.0,
        "upper_assembly": 0.0,
        "upper_initial_guess": 0.0,
        "upper_solver": 0.0,
        "upper_calls": 0,
    }
    skipped_node_counter = 0
    L = [0]
    U = [tolerance + 1 ] # set U s.t. the tolerance criterion is not directly fullfilled

    open_nodes = []  
    total_nodes = [] 
    root_config = np.zeros(n_electrodes)  
    
    # Compute initial lower und upper bound solutions and their values
    if precompute == True:
        print(f"Precomputing matrices ......")
        pre_prcomp_time = time.time()
        precomp = precompute_full_active_set(B,e_tilde,omega,epsilon)
        post_precomp_time = time.time()
        print(f"Precomputing done: Time it took {post_precomp_time - pre_prcomp_time} seconds.")
        root_lb = lower_bound(root_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, precomp, timing=timing, remove_o_e_constraint=remove_o_e_constraint)[0]
        root_ub,_,last_ub_sol,_ = upper_bound(root_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, advanced_picking_rule, precomp, timing=timing, remove_o_e_constraint=remove_o_e_constraint)
    else:
        root_lb = lower_bound(root_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, timing=timing)[0]
        root_ub,_,last_ub_sol,_ = upper_bound(root_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, advanced_picking_rule, timing=timing)

    # Append to dictionaries
    open_nodes.append({
        "config": root_config,
        "lb": root_lb,
        "ub": root_ub,
        "terminal": False,
        "total_nodes_position": 0
    })
    total_nodes.append({
        "config": root_config,
        "lb": root_lb,
        "ub": root_ub,
        "terminal": False,
        "parent": None,
        "iteration_created": 0
    })


    best_ub = root_ub+1  # Initialize the best upper bound with a value greater than the root upper bound
    best_config = root_config
    terminal_nodes = []
    while open_nodes and k < k_max and  U[-1]-L[-1] > tolerance:
        print(f'------------------------Iteration {k}-------------------------:')
        print(f'Current global lower bound: {min(node["lb"] for node in open_nodes)}')
        print(f'Current global upper bound: {best_ub}')
        print(f'Number of configurations in delta: {len(open_nodes)}')
        
        idx = min(range(len(open_nodes)), key=lambda i: open_nodes[i]["lb"])
        selected_node = open_nodes.pop(idx)
        selected_config = selected_node["config"]
        print(f'Selected configuration index: {idx}')
        print(f'Selected configuration: {selected_config}')
        
        if np.sum(selected_config == 0) != 0:
            # Branching: Create new configurations by activating or deactivating the next unassigned electrode
            if np.sum(selected_config == 1) == n_usable_electrodes:
                # here we can set all remaining unassigned electrodes to -1 and get a terminal node
                selected_config[selected_config == 0] = -1
                selected_node["terminal"] = True
                terminal_nodes.append(selected_node)   
                total_nodes[selected_node["total_nodes_position"]]["terminal"] = True
                k += 1
                continue
            new_configs = split_electrode_set(last_ub_sol,selected_config)
        else:
            selected_node["terminal"] = True
            terminal_nodes.append(selected_node)    
            k += 1    
            continue  # No unassigned electrodes left, skip branching
        
        for child_config in new_configs:
            ae = np.sum(child_config == 1)  # Count the number of active electrodes in the child configuration
            ue = np.sum(child_config == 0)  # Count the number of unassigned electrodes in the child configuration

            if ae > n_usable_electrodes:
                skipped_node_counter += 1
                continue  # Skip this configuration if it has more active electrodes than allowed
            if precompute:
                child_lb = lower_bound(child_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, precomp, timing=timing, remove_o_e_constraint=remove_o_e_constraint)[0]
            else:
                child_lb = lower_bound(child_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, timing=timing)[0]
            if child_lb + 1e-2 < L[-1] and k >0:
                raise RuntimeError(
                f"Global lower bound decreased: old L={L[-1]}, new L={child_lb}, "
                f"difference={child_lb - L[-1]},"
                f"child config={child_config}, parent config={selected_config}"
            )
            if child_lb > best_ub:
                skipped_node_counter += 1
                continue  # Prune this branch if the lower bound is greater than the best upper bound
            
            if precompute:
                child_ub,_,last_ub_sol,ub_concrete_config = upper_bound(child_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, advanced_picking_rule, precomp, timing=timing, remove_o_e_constraint=remove_o_e_constraint)
            else:
                child_ub,_,last_ub_sol,ub_concrete_config = upper_bound(child_config, alpha, beta, B, e_tilde, omega, epsilon, I_ind, I_total, n_usable_electrodes, verbosity, advanced_picking_rule, timing=timing)

            if child_ub < best_ub:
                best_ub = child_ub
                best_config = ub_concrete_config.copy()
            
            open_nodes.append({
                "config": child_config,
                "lb": child_lb,
                "ub": child_ub,
                "terminal": False,
                "total_nodes_position": len(total_nodes)  # Store the position of this node in the total_nodes list
            })
            total_nodes.append({
                "config": child_config,
                "lb": child_lb,
                "ub": child_ub,
                "terminal": False,
                "parent": selected_node["total_nodes_position"],
                "iteration_created": k
            })

        # Again prune if lb y best ub 
        open_nodes = [
            node for node in open_nodes
            if node["lb"] < best_ub
        ]
    
        if open_nodes:
            L.append(min(node["lb"] for node in open_nodes))
        else:
            L.append(best_ub)  # If no open nodes left, set global lower bound to best upper bound

        U.append(best_ub)
        print(f'Number of skipped nodes so far: {skipped_node_counter}')
        k += 1  # Increment the iteration counter
        
    print('Branch and bound algorithm finished.')
    print(f'Best upper bound found: {best_ub}')
    print(f'Best configuration found: {best_config}')
    print(f'Number of iterations: {k}')
    print(f'Number of open nodes at the end: {len(open_nodes)}')
    print(f'Final global lower bound: {min(node["lb"] for node in open_nodes) if open_nodes else "N/A"}')

    print(f"Compute corresponding optimal solution for the best configuration found...")
    # slice the matrices and compute corresponding solution for the selected electrodes:
    selected_electrodes = np.where(best_config == 1)[0]
    B_sliced = B[:, selected_electrodes]    
    t0 = time.perf_counter()
    Q,r,c,C,b,D = active_set_assembly(alpha, beta, B_sliced, e_tilde, B_sliced.shape[1], omega, epsilon, I_ind, I_total)
    initial_guess, initial_working_set = initial_guess_and_working_set_from_active_constraints(Q, D, c, C, b)
    exit_status, active_set_solution, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iterations = dpo.solve_quadratic_program_with_initial_guess(Q, r, D, c, C, b, initial_guess, initial_working_set,verbosity)
    # map the solution back to the full electrode space
    full_solution = np.zeros(B.shape[1], dtype=np.float64)
    full_solution[selected_electrodes] = active_set_solution[:len(selected_electrodes)]
    print("---------- DETAILED TIMING SUMMARY ----------")

    print(f"lower total:          {timing['lower_total']:.3f}s")
    print(f"lower assembly:       {timing['lower_assembly']:.3f}s")
    print(f"lower initial guess:  {timing['lower_initial_guess']:.3f}s")
    print(f"lower solver:         {timing['lower_solver']:.3f}s")
    print(f"lower calls:          {timing['lower_calls']}")

    if timing["lower_calls"] > 0:
        print(f"lower avg total:      {timing['lower_total'] / timing['lower_calls']:.6f}s")
        print(f"lower avg assembly:   {timing['lower_assembly'] / timing['lower_calls']:.6f}s")
        print(f"lower avg init guess: {timing['lower_initial_guess'] / timing['lower_calls']:.6f}s")
        print(f"lower avg solver:     {timing['lower_solver'] / timing['lower_calls']:.6f}s")

    print()
    print(f"upper total:          {timing['upper_total']:.3f}s")
    print(f"upper assembly:       {timing['upper_assembly']:.3f}s")
    print(f"upper initial guess:  {timing['upper_initial_guess']:.3f}s")
    print(f"upper solver:         {timing['upper_solver']:.3f}s")
    print(f"upper calls:          {timing['upper_calls']}")

    if timing["upper_calls"] > 0:
        print(f"upper avg total:      {timing['upper_total'] / timing['upper_calls']:.6f}s")
        print(f"upper avg assembly:   {timing['upper_assembly'] / timing['upper_calls']:.6f}s")
        print(f"upper avg init guess: {timing['upper_initial_guess'] / timing['upper_calls']:.6f}s")
        print(f"upper avg solver:     {timing['upper_solver'] / timing['upper_calls']:.6f}s")

        if timing["lower_calls"] > 0:
            print(f"lower avg:    {timing['lower_total'] / timing['lower_calls']:.6f}s")

        if timing["upper_calls"] > 0:
            print(f"upper avg:    {timing['upper_total'] / timing['upper_calls']:.6f}s")
    return best_config, terminal_nodes, open_nodes, L, U, total_nodes, full_solution, selected_electrodes, opt_func_value  # Return the optimal stimulation pattern found by the algorithm
