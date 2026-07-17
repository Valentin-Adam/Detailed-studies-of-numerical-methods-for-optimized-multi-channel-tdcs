"""This file is for testing the ADMM implementation"""
import numpy as np
import h5py as h5
import sys
import os
import configparser
import time

# setting ADMM parameters 
max_iterations = 5000
alpha = 1
beta = 1
rel_tol = 1.e-7
abs_tol = 1.e-8
eps = 10000             # In Table 6.1 Thesis eps waas also set to 6100, 0.0001
omega_value_non_target = 1.0 # grey matter
omega_value = 0.001  # target region &  non grey matter
grey_matter_non_target = True
# initial dual parameters
mu_1 = 0.1 
mu_2 = 0.1 
mu_3 = 0.1
update_mu_every = 1     # In Table 6.1 Thesis update_mu_every waas also set to 20
verbosity =1

# load configs
head_model = "Sphere" # Select head model here, 'Sphere' or 'Head'
configs = configparser.ConfigParser()
configs.optionxform = str
configs.read('configs.ini')

# libraries
duneuropy_path = configs.get('libraries', 'duneuro')
sys.path.append(duneuropy_path)
from admm_implementation.io_utils import create_admm_run_directory
import duneuropy as dp

# data paths
if head_model=="Sphere":
    data_cfg = configs['data_sphere']
else:
    data_cfg = configs['data_head']
current_densities_path = data_cfg.get('current_densities')
element_centers_path = data_cfg.get('element_centers')
element_labels_path = data_cfg.get('element_labels')
meshfile = data_cfg.get('meshfile')
registered_sensors_path = data_cfg.get('registered_sensors')
outputdir = data_cfg.get('outputdir_admm', './admm_results')

# ensure paths exist and output dir
os.makedirs(outputdir, exist_ok=True)

from admm_implementation.io_utils import h5_to_np_array  

# TIMING: Data loading
t_start_data = time.time()
data = h5_to_np_array(current_densities_path, element_centers_path, element_labels_path)
t_end_data = time.time()
print(f"[TIME] Data loading: {t_end_data - t_start_data:.3f}s")
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)

# Target generation for Sphere or Head (10 targets in sphere case, 5 in head case (ADMM doesnt converge for head see thesis))
from admm_implementation.io_utils import generate_targets_from_element_centers
from admm_implementation.own_admm import tDCS_admm

if head_model=="Sphere":
    grey_matter_indices = np.where(data[2]==3)[0]
else:
    grey_matter_indices = np.where(data[2]==1)[0]
element_centers = data[1]
element_labels = data[2]
if head_model=="Sphere":
    target_element_indices, target_points, radial_vectors, tangential_vectors = (
        generate_targets_from_element_centers(
            element_centers=element_centers,
            element_labels=element_labels,
            target_label=3,
            n_targets=10,
            center=np.array([127,127,127]),
            outer_percentile_within_label=95,
            seed=1,
        )
    )
    target_element_indices_radial = target_element_indices
    target_element_indices_tangential = target_element_indices
    print(target_element_indices.shape)
    print(target_points.shape)
    print(radial_vectors.shape)
    print(tangential_vectors.shape)
else:
    target_element_indices_radial, target_points_radial, radial_vectors,target_element_indices_tangential, target_points_tangential, tangential_vectors = (
    generate_targets_from_element_centers(
        element_centers=element_centers,
        element_labels=element_labels,
        model_type="head",
        target_label=1,
        n_targets=5,
        center=None,
        inner_percentile_within_label=80,
        outer_percentile_within_label=95,
        seed=1,
        )
    )


# forward stimulation matrix, excluding reference electrode column
B = data[0][:,1:] 

# Setting sizes
n_chan = B.shape[1]
n_grid = B.shape[0]  

# initial I 
I_0 = np.zeros((n_chan, 1))

# Setting up the initial omega 
t_start_setup = time.time()
omega = np.zeros((n_grid, 1))
if grey_matter_non_target:
    gr_or = "GR"
    omega[grey_matter_indices*3, 0] = omega_value_non_target  # all grey matter elements have weight 1.0
    omega[grey_matter_indices*3+1, 0] = omega_value_non_target
    omega[grey_matter_indices*3+2, 0] = omega_value_non_target
else:
    gr_or = "NGR"

# Tracking solutions and iterations for the different targets
number_of_iterations_total_radial = []
number_of_iterations_total_tangential= []
I_total_radial = []
I_total_tangential = []

# Creating directory for saving the results
run_dir, run_name = create_admm_run_directory(
    outputdir=outputdir,
    subdir='random_target_evaluation',
    gr_or_ngr=gr_or,
    abs_tol=abs_tol,
    rel_tol=rel_tol,
    alpha=alpha,
    beta=beta,
    omega=omega_value,
    epsilon=eps,
    update_mu_every=update_mu_every,
    seed= 1,
    outer_percentile=95
)
from pathlib import Path
run_dir = Path(run_dir)

# Slicing the B's for better performances see thesis
B_indices = np.concatenate((grey_matter_indices*3, grey_matter_indices*3 + 1, grey_matter_indices*3 + 2))
B_sliced = B[B_indices,:]


print(f"---------------Radial Indices-------------")
for i in range(target_element_indices_radial.shape[0]):   
    target_indices = target_element_indices_radial[i]
    print(f'-----------[ADMM Call Iteration {i}]: Target_indice {target_indices}--------------------------')
    t_start_setup = time.time()
    omega_iteration = omega
    e_tilde_radial = np.zeros((n_grid,1))
    e_tilde_tangential = np.zeros((n_grid,1))

    #Setting up Omega and e_tildes for this specific target
    omega_iteration[3*target_indices] = omega_value
    omega_iteration[3*target_indices+1] = omega_value
    omega_iteration[3*target_indices+2] = omega_value
    e_tilde_radial[3*target_indices] = radial_vectors[i][0]
    e_tilde_radial[3*target_indices+1] = radial_vectors[i][1]
    e_tilde_radial[3*target_indices+2] = radial_vectors[i][2]
    t_end_setup = time.time()
    print(f"[TIME] Setup (omega, e_tilde): {t_end_setup - t_start_setup:.3f}s")

    # ADMM Call
    print(f'Starting ADMM algorithm for target......')
    t_start_iteration = time.time()
    omega_iteration_sliced = omega_iteration[B_indices]
    e_tilde_radial_sliced = e_tilde_radial[B_indices]
    I_radial, number_of_iteration_radial= tDCS_admm(I_0, B_sliced, eps, mu_1, mu_2, alpha, beta, omega_iteration_sliced, e_tilde_radial_sliced, 
                      max_iter=max_iterations, reL_tol=rel_tol, abs_tol=abs_tol, tol_wagner=1e-6, output_I_history=False, output_all_history=False, print_residuals=False , output_last_values=False, tau_decrease=2, tau_increase=2, plot_lagrangian=False
                      ,include_ref_penalty=True, update_mu_every=update_mu_every, verbosity=verbosity)
    if number_of_iteration_radial < max_iterations:
        print(f'ADMM converged after {number_of_iteration_radial} iterations.')
    else:
        print(f'Reached max iterations.')
    t_end_iteration = time.time()

    # Saving the results
    number_of_iterations_total_radial.append(number_of_iteration_radial)
    I_total_radial.append(I_radial)
    print(f"[TIME] ADMM solving for target: {t_end_iteration - t_start_iteration:.3f}s")


print(f"---------------Tangential Indices-------------")
for i in range(target_element_indices_tangential.shape[0]):
    target_indices = target_element_indices_tangential[i]
    print(f'-----------[ADMM Call Iteration {i}]: Target_indice {target_indices}--------------------------')
    t_start_setup = time.time()
    omega_iteration = omega
    e_tilde_tangential = np.zeros((n_grid,1))

    #Setting up Omega and e_tildes for this specific target
    omega_iteration[3*target_indices] = omega_value
    omega_iteration[3*target_indices+1] = omega_value
    omega_iteration[3*target_indices+2] = omega_value
    e_tilde_tangential[3*target_indices] = tangential_vectors[i][0]
    e_tilde_tangential[3*target_indices+1] = tangential_vectors[i][1]
    e_tilde_tangential[3*target_indices+2] = tangential_vectors[i][2]
    t_end_setup = time.time()
    print(f"[TIME] Setup (omega, e_tilde): {t_end_setup - t_start_setup:.3f}s")

    # ADMM Call
    print(f'Starting ADMM algorithm for target......')
    t_start_iteration = time.time()
    omega_iteration_sliced = omega_iteration[B_indices]
    e_tilde_tangential_sliced = e_tilde_tangential[B_indices]
    I_tangential, number_of_iteration_tangential= tDCS_admm(I_0, B_sliced, eps, mu_1, mu_2, alpha, beta, omega_iteration_sliced, e_tilde_tangential_sliced, 
                      max_iter=max_iterations, reL_tol=rel_tol, abs_tol=abs_tol, tol_wagner=1e-6, output_I_history=False, output_all_history=False, print_residuals=False , output_last_values=False, tau_decrease=2, tau_increase=2, plot_lagrangian=False
                      ,include_ref_penalty=True, update_mu_every=update_mu_every, verbosity=verbosity)
    if number_of_iteration_tangential < max_iterations:
        print(f'ADMM converged after {number_of_iteration_tangential} iterations.')
    else:
        print(f'Reached max iterations.')
    t_end_iteration = time.time()

    # Saving the results
    number_of_iterations_total_tangential.append(number_of_iteration_tangential)
    I_total_tangential.append(I_tangential)
    print(f"[TIME] ADMM solving for target: {t_end_iteration - t_start_iteration:.3f}s")

# Saving files
print("Saving solution arrays and computation information..........")
rad_I_file = run_dir / "radial_I.npy"
rad_iterations_file = run_dir / "radial_iterations.npy"
tang_I_file = run_dir / "tangential_I.npy"
tang_iterations_file = run_dir / "tangential_iterations.npy"
targets_radial_file = run_dir / "target_indices_radial.npy"
targets_radial_vector_file = run_dir / "target_vector_radial.npy"
targets_tan_file = run_dir / "target_indices_tan.npy"
targets_tan_vector_file = run_dir / "target_vector_tan.npy"


np.save(rad_I_file, I_total_radial)
np.save(rad_iterations_file, number_of_iterations_total_radial)
np.save(tang_I_file, I_total_tangential)
np.save(tang_iterations_file, number_of_iterations_total_tangential)
np.save(targets_tan_file, target_element_indices_tangential)
np.save(targets_tan_vector_file, tangential_vectors)
np.save(targets_radial_file, target_element_indices_radial)
np.save(targets_radial_vector_file, radial_vectors)

print(f'Saved under {run_dir}')
