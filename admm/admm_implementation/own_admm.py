import numpy as np 
from scipy.linalg import solve
import time
from admm_implementation.convergence import residual_convergence, wagner_convergence
from admm_implementation.io_utils import plot_admm_convergence


def z_update(alpha,mu_1,beta,I_S,p_1):
    """
    The corrected z update step.
    """
    z = np.zeros_like(I_S)
    for i in range(len(I_S)):
        gamma = -1*mu_1*I_S[i]-p_1[i]
        if gamma > beta:
            z[i] = -1*(gamma - beta)/(2*alpha + mu_1)
        elif gamma < -1*beta:
            z[i] = -1*(gamma + beta)/(2*alpha + mu_1)
        else:
            z[i] = 0
    return z
        
def z_update_compact(alpha, mu_1, beta, I_S, p_1):
    """
    The corrected z update step in a compact form.
    """
    gamma = -mu_1 * I_S - p_1
    shrink = np.sign(gamma) * np.maximum(np.abs(gamma) - beta, 0)
    return -shrink / (2*alpha + mu_1)

def z_update_old_version(alpha, mu_1, beta, I_S, p_1):
    """
    The old z updated step according to Wagner et al.
    """
    z = np.zeros_like(I_S)
    id_tilde = np.sqrt(mu_1/2) * np.eye(len(I_S))
    id_tilde_alpha = id_tilde.T @ id_tilde + alpha * np.eye(len(I_S))
    b_z = id_tilde.T @( np.sqrt(mu_1/2) * I_S + (1/np.sqrt(2*mu_1)) * p_1 - 1/(np.sqrt(2*mu_1)) * beta )
    z = solve(id_tilde_alpha, b_z)
    return z

def soft_scalar(x, tau):
    """
    The soft scalar operator.
    """
    return float(np.sign(x) * max(abs(x) - tau, 0.0))

def tDCS_admm(I_0, B,eps, mu_1, mu_2, alpha, beta, omega, e_tilde,max_iter=10000, reL_tol=1e-2, abs_tol=1e-3, tol_wagner=1e-10, tau_increase=2, tau_decrease=2, rho = 10,
                 old_version=False, print_residuals=False, output_I_history=False,  output_all_history=False, just_wagner_conv = False, output_last_values=False, plot_lagrangian=False,
                 include_ref_penalty=True,   update_mu_every = 20, mu_3 = None, warm_start = False, verbosity=0):
    """
    Main function for the tdcs ADMM (ref. Algorithm 7 in Thesis).

    Parameter:
     - I_0 : initial starting point (dim = B.shape[1])
     - B : Discretized stimulation forward operator
     - eps : the epsilon for the omega-epsilon side constraint
     - mu_1 : starting penalty paramter mu_1
     - mu_2 : starting penalty paramter mu_2
     - alpha : regularization parameter in the objective functional
     - eta : regularization parameter in the objective functional
     - omega : omega vector for the side constraints on all nodes (dim = B.shape[0])
     - e_tilde : the target vector (dim = B.shape[0])

    Optional/Predefined parameters (A lot of them for testing functionalities or extra printing):
     - max_iter = Maximum number of iterations
     - reL_tol = Relative tolerance for stopping
     - abs_tol = Absolute tolerance for stopping
     - tol_wagner = Tolerance for Wagner stopping criterion
     - tau_increase = Factor by which the mu_i increases
     - tau_decrease = Factor by which the mu_i decreases
     - rho = rho_value for the mu_i updating
     - old_version = Flag if the old error containing z update should be used
     - print_residuals = Flag if the resiudals should be plottet
     - output_I_history = Flag if the output should contain all I
     - output_all_history = Flag if the output should contain all information (I_history, primals_history, duals_history, ...)
     - just_wagner_conv = Flag if just the wagner convergence should be used
     - output_last_values = Falg if all values of tthe last iteration should be returned
     - plot_lagrangian = Flag if the Lagragian should be plotted
     - include_ref_penalty = Flag if the Ref electrode should be included in the objective function --> Should be True
     - update_mu_every = Parameter to control how often the mu_i are updated
     - mu_3 = starting penalty paramter mu_3 (if None it chooses mu_3 = mu_1)
     - warm_start = Warum start functionality (not checked if useful)
     - verbosity = verbosity to toggle prints 

    Returns: Depending on the parameter choices
     - (optimal I) or (I_history) or (I_history, z, y, p1, p2, p3, mu_1, mu_2, mu_3, x) or (I_history, p_1_history, p_2_history, p_3_history, mu_1_history, mu_2_history, mu_3_history, z_history, y_history, x_history)
    
     
    """
    initializing_time = 0.0
    copy_time = 0.0
    solving_time = 0.0
    i_update_time = 0.0
    y_update_time = 0.0
    z_update_time = 0.0
    x_update_time = 0.0
    p_update_time = 0.0
    conv_check_time = 0.0
    mask_counter = 0
    
    # Add dimension checks
    if verbosity >=1:
        print(f"Dimensions:")
        print(f"B: {B.shape}")
        print(f"omega: {omega.shape}")
        print(f"e_tilde: {e_tilde.shape}")
    
    t_temp = time.time()
    n_grid, n_chan = B.shape[0], B.shape[1] 

    # Initialize variables
    I = I_0
    I_history = []
    p_1_history = []
    p_2_history = []
    mu_1_history = []
    mu_2_history = []
    z_history = []
    y_history = []
    z = np.zeros((n_chan, 1))
    y = np.zeros((n_grid, 1))
    p1 = np.zeros((n_chan, 1))
    p2 = np.zeros((n_grid, 1))

    if include_ref_penalty:
        if mu_3 is None:
            mu_3 = mu_1  # default to same as mu_1 if not specified

    if warm_start:
        p2 = -e_tilde.copy()
        t  = np.where(omega == 0, np.inf, eps/omega)
        y  = np.clip(B @ I, -t, t)   

    
    if include_ref_penalty:
        ones = np.ones((n_chan, 1))
        ones_onesT = ones @ ones.T      
        x= (ones.T@I_0).item()           
        p3 = float(0.0)                
        x_history = []
        p_3_history = []
        mu_3_history = []

    initializing_time += time.time() - t_temp

    # Precompute matrices
    B_small = B.T @ B 

    # store residual history for plotting
    residual_history_1 = []
    residual_history_2 = []
    # residual history for ref constraint
    residual_history_3 = []
    Lagrangian_history = []
    Lagrangian_0_history = []
    # convergence bookkeeping
    wagner_iter = None
    residual_iter = None
    if just_wagner_conv:
        residual_iter = 0
    both_converged = False


    mu1_old, mu2_old = mu_1, mu_2
    if include_ref_penalty:
        mu3_old = mu_3

    for iteration in range(max_iter):
        t_temp = time.time()
        I_prev = I.copy()
        z_prev = z.copy()
        y_prev = y.copy()
        
        if include_ref_penalty:
            x_prev = x 
        copy_time += time.time() - t_temp
        t_temp = time.time()

        # ---------- I update -------------------------
        b_I = mu_1*z - p1 + B.T @ (mu_2*y- p2) 
        if include_ref_penalty:
            b_I = b_I + ones * (mu_3 * x - p3)
        if mu_1 != mu1_old or mu_2 != mu2_old or iteration == 0: # check if we need to recompute A
            A = mu_1*np.eye(n_chan) + mu_2*B_small
            if include_ref_penalty:
                A = A + mu_3 * ones_onesT
        
        t_temp2 = time.time()
        I = solve(A, b_I)
        solving_time += time.time() - t_temp2
        i_update_time += time.time() - t_temp

        # ---------- y update -------------------------
        t_temp = time.time()

        B_prev = B @ I + (1/mu_2)*(p2 + e_tilde)
        threshold = np.where(omega == 0, np.inf, eps / omega)
        mask = np.abs(B_prev) > threshold
        # track how many entries are being thresholded 
        mask_counter += np.sum(mask)
        y = np.where(mask, np.sign(B_prev) * threshold, B_prev)
        y_update_time += time.time() - t_temp
        if verbosity >=1:
            print(f"mask counter: {mask_counter}")

        # ---------- z update -------------------------
        t_temp = time.time()
        if old_version:
            z = z_update_old_version(alpha, mu_1, beta, I, p1)
        else:
            z = z_update_compact(alpha, mu_1, beta, I, p1)
        z_update_time += time.time() - t_temp
        

        # ---------- x update -------------------------
        if include_ref_penalty:
            t_temp = time.time()
            a = (ones.T @ I).item()  # a = 1^T I^{k+1}
            denom = (mu_3 + 2*alpha)
            b = (mu_3 * a + p3) / denom
            tau = beta / denom
            x = soft_scalar(b, tau)
            x_update_time = time.time() - t_temp

        # p1 and p2 updates
        t_temp = time.time()
        prim_res1 = I - z
        prim_res2 = B @ I - y
        p1 = p1 + mu_1 * prim_res1
        p2 = p2 + mu_2 * prim_res2
        #p3 update
        if include_ref_penalty:
            prim_res3 = (ones.T @ I).item() - x 
            p3 = p3 + mu_3 * prim_res3

        p_update_time += time.time() - t_temp

        # Check convergence
        t_temp = time.time()
        dual_residual_1 = -mu_1 * (z - z_prev)
        dual_residual_2 = -mu_2 * B.T @ (y - y_prev)

        pr_norm = np.linalg.norm(prim_res1)
        du_norm = np.linalg.norm(dual_residual_1)   
        pr_norm_2 = np.linalg.norm(prim_res2)
        du_norm_2 = np.linalg.norm(dual_residual_2)

        residual_history_1.append((pr_norm, du_norm))
        residual_history_2.append((pr_norm_2, du_norm_2))

        # residuals for ref constraint
        if include_ref_penalty:
            pr_norm_3 = abs((ones.T @ I).item() - x )
            dual_residual_3 = -mu_3 * (x - x_prev) * ones
            du_norm_3 = np.linalg.norm(dual_residual_3)
            residual_history_3.append((pr_norm_3, du_norm_3))

        mu1_old, mu2_old = mu_1, mu_2
        if include_ref_penalty:
            mu3_old = mu_3

        # adaptive rhoupdate
        if iteration > 50  and iteration % update_mu_every == 0:
            if pr_norm > rho * du_norm:
                mu_1 *= tau_increase
                if verbosity >=1:
                    print(f"mu_1 increased to {mu_1}")
            elif du_norm > rho * pr_norm:
                mu_1 /= tau_decrease
                if verbosity >=1:
                    print(f"mu_1 decreased to {mu_1}")
            if pr_norm_2/np.sqrt(n_grid) > rho * du_norm_2/np.sqrt(n_chan) and mu_2 < 1e10:
                mu_2 *= tau_increase    
                if verbosity >=1:
                    print(f"mu_2 increased to {mu_2}")
            elif du_norm_2/np.sqrt(n_chan) > rho * pr_norm_2/np.sqrt(n_grid) and mu_2 > 1e-10:
                mu_2 /= tau_decrease
                if verbosity >=1:
                    print(f"mu_2 decreased to {mu_2}")
            
            # mu_3 update
            if include_ref_penalty:
                if pr_norm_3 > rho * du_norm_3:
                    mu_3 *= tau_increase
                    if verbosity >=1:
                        print(f"mu_3 increased to {mu_3}")
                elif du_norm_3 > rho * pr_norm_3:
                    mu_3 /= tau_decrease
                    if verbosity >=1:
                        print(f"mu_3 decreased to {mu_3}")

        # record when each convergence criterion is first satisfied (but continue)
        if iteration >= 3 and wagner_iter is None and wagner_convergence(I, I_prev, tol_wagner):
            wagner_iter = iteration
            if verbosity >=1:
                print(f"Wagner criterion met at iteration {wagner_iter}")

        # residual criterion
        residual_ok = False
        if iteration >= 3 and residual_iter is None:
            residual_ok = residual_convergence(np.linalg.norm(I), np.linalg.norm(z), np.linalg.norm(p1),
                                               reL_tol, abs_tol, prim_res1, dual_residual_1,verbosity) and \
                          residual_convergence(np.linalg.norm(B@I), np.linalg.norm(y), np.linalg.norm(B.T @ p2),
                                               reL_tol, abs_tol, prim_res2, dual_residual_2,verbosity)
            if include_ref_penalty:
                residual_ok = residual_ok and residual_convergence(np.linalg.norm(I), np.linalg.norm(x), np.linalg.norm(p3),
                                               reL_tol, abs_tol, prim_res3, dual_residual_3,verbosity)
            # keep track of when residual criterion is met, but only stop when both criteria are satisfied 
            if residual_ok:
                residual_iter = iteration
                if verbosity >=1:
                    print(f"Residual criterion met at iteration {residual_iter}")

        # stop only when both criteria have been met (or when max_iter reached)
        if (wagner_iter is not None) and (residual_iter is not None):
            if verbosity >=1:
                print(f"Both criteria satisfied (Wagner at {wagner_iter}, Residual at {residual_iter}). Stopping at iter {iteration}.")
            both_converged = True
            break
        conv_check_time += time.time() - t_temp
        if verbosity >=1:
            print(f"Iteration {iteration}: Primal Residual_1 = {pr_norm:.3e}, Dual Residual_1 = {du_norm:.3e}")
            print(f"Iteration {iteration}: Primal Residual_2 = {pr_norm_2:.3e}, Dual Residual_2 = {du_norm_2:.3e}")
        if include_ref_penalty:
            if verbosity >=1:
                print(f"Iteration {iteration}: Primal Residual_3 = {pr_norm_3:.3e}, Dual Residual_3 = {du_norm_3:.3e}")
        if verbosity >=1:
            print(f"change in I: {np.linalg.norm(I - I_prev):.3e}")
        
        # fill the histories
        if output_I_history:
            I_history.append(I.copy())
        if output_all_history:
            p_1_history.append(p1.copy())
            p_2_history.append(p2.copy())
            mu_1_history.append(mu_1)
            mu_2_history.append(mu_2)
            z_history.append(z.copy())
            y_history.append(y.copy())
            if include_ref_penalty:
                p_3_history.append(p3)
                mu_3_history.append(mu_3)
                x_history.append(x )

        # compute and plot Lagrangian L_0
        if plot_lagrangian:
            lagrangian_0 = (alpha * z.T @ z) + beta * np.sum(np.abs(z)) - (y.T @ e_tilde) + p1.T @ (I - z) + p2.T @ (B @ I - y)
            if include_ref_penalty:
                a = (ones.T @ I).item()
                lagrangian_0 = lagrangian_0 + (alpha * (x **2)) + (beta * abs(x )) + (p3 * (a - x ))
            Lagrangian_0_history.append(lagrangian_0.item())

            lagrangian = lagrangian_0 + (mu_1/2) * np.linalg.norm(I - z)**2 + (mu_2/2) * np.linalg.norm(B @ I - y)**2
            # Add augmented penalty for ref constraint
            if include_ref_penalty:
                lagrangian = lagrangian + (mu_3/2) * (a - x )**2
            Lagrangian_history.append(lagrangian.item())
            if verbosity >=1:
                print(f"Lagrangian at iteration {iteration}: {lagrangian_0.item():.3e}")

    if not both_converged:
        if verbosity >=1:
            print("Reached maximum iterations without convergence.")

    # plot residual history when max iterations reached
    if residual_history_1 and residual_history_2 and print_residuals:
        plot_admm_convergence(
            residual_history_1, residual_history_2,
            residual_history_3=residual_history_3 if include_ref_penalty else None,
            Lagrangian_0_history=Lagrangian_0_history if plot_lagrangian else None,
            Lagrangian_history=Lagrangian_history if plot_lagrangian else None,
            wagner_iter=wagner_iter,
            residual_iter=residual_iter,
            include_ref_penalty=include_ref_penalty,
            plot_lagrangian=plot_lagrangian
        )

    if verbosity >=1:
        print(f"Timing summary (in seconds):")
        print(f"Initialization time: {initializing_time:.3f}s")
        print(f"Copy time: {copy_time:.3f}s")
        print(f"Solving time: {solving_time:.3f}s")
        print(f"I update time: {i_update_time:.3f}s")
        print(f"Y update time: {y_update_time:.3f}s")
        print(f"Z update time: {z_update_time:.3f}s")
        print(f"P update time: {p_update_time:.3f}s")
        print(f"Convergence check time: {conv_check_time:.3f}s")

    if output_I_history and not output_all_history and not output_last_values:
        return I_history
    if output_last_values:
        if include_ref_penalty:
            return I_history, z, y, p1, p2, p3, mu_1, mu_2, mu_3, x
        return I_history, z, y, p1, p2, mu_1, mu_2
    if output_all_history:
        if include_ref_penalty:
            return I_history, p_1_history, p_2_history, p_3_history, mu_1_history, mu_2_history, mu_3_history, z_history, y_history, x_history
        return I_history, p_1_history, p_2_history, mu_1_history, mu_2_history, z_history, y_history
    return I,iteration if 'I' in locals() else None
