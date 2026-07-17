import numpy as np
import h5py as h5
import sys
import os
import ast
import configparser
import time
import subprocess
import re

# Select if...
#   - one wants to test multiple targets and write one random radial and one random tangential 
#   - one wants one radial and one tangential target computation and visualisation that are near each other
one_artificial_target = True

# setting active_set_parameters
alphas = [10,1e-5,10] # here varying alphas and betas for the thesis
betas = [0,10,10]
eps = 2000
omega_value_non_target = 1.0 # grey matter
omega_value = 1e-06 # target region &  non grey matter
grey_matter_non_target = True
tolerance = 0.001
I_total = None
I_ind = None

# load configss
head_model = "Sphere"
configs = configparser.ConfigParser()
configs.optionxform = str
configs.read('configs.ini')
n_targets_sphere = 1
n_targets_head = 1
verbosity = 1

# libraries
duneuropy_path = configs.get('libraries', 'duneuro')
duneuropt_path = configs.get('libraries', 'duneuropt')
MA_path = configs.get('libraries', 'master_thesis')
sys.path.append(duneuropy_path)
sys.path.append(duneuropt_path)
sys.path.append(MA_path)
from admm_implementation.io_utils import create_active_set_run_directory
from admm_implementation.io_utils import _write_admm_results
from admm_implementation.io_utils import generate_targets_from_element_centers
from admm_implementation.io_utils import check_whole_domain_constraints
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
outputdir = data_cfg.get('outputdir_active_set')

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

from master_thesis.admm.active_set_wrapper.active_set import active_set_setup, active_set_evaluation

# Target generation for Sphere or Head (n_targets_sphere targets in sphere case, n_targets_head in head case)
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
            n_targets=n_targets_sphere,
            center=np.array([127,127,127]),
            outer_percentile_within_label=95,
            seed=1,
        )
    )
    target_element_indices_radial = target_element_indices
    target_element_indices_tangential = target_element_indices
    target_points_radial_priori = target_points
    target_points_tangential_priori = target_points
    radial_vectors_priori = radial_vectors
    tangential_vectors_priori = tangential_vectors
    print(target_element_indices.shape)
    print(target_points.shape)
    print(radial_vectors_priori.shape)
    print(tangential_vectors_priori.shape)
elif not one_artificial_target:
    target_element_indices_radial, target_points_radial, radial_vectors,target_element_indices_tangential, target_points_tangential, tangential_vectors = (
    generate_targets_from_element_centers(
        element_centers=element_centers,
        element_labels=element_labels,
        model_type="head",
        target_label=1,
        n_targets=n_targets_head,
        center=None,
        inner_percentile_within_label=80,
        outer_percentile_within_label=95,
        seed=1,
        )
    )
else:
    target_element_indices_radial =ast.literal_eval(configs.get('data_head', 'single_radial_indice'))
    radial_vectors = np.array([
        ast.literal_eval(configs.get('data_head', 'single_radial_moment'))
    ], dtype=float)
    target_element_indices_tangential = ast.literal_eval(configs.get('data_head', 'single_tangential_indice'))
    tangential_vectors = np.array([
         ast.literal_eval(configs.get('data_head', 'single_tangential_moment'))
             ], dtype=float)

# forward stimulation matrix, excluding reference electrode column
B = data[0][:,:] 

# Setting sizes
n_chan = B.shape[1]
n_grid = B.shape[0]  

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


rad_stats = []
tan_stats = []

# Running the evaluation for the different alpha-beta combinations
for i in range(len(alphas)):#
    I_total_radial = []
    I_total_tangential = []
    alpha = alphas[i]
    beta = betas[i]
    # Creating directory for saving the results
    run_dir, run_name = create_active_set_run_directory(
    outputdir,
    "random_target_evaluation",
    abs_tol=1e-8,
    rel_tol=1e-7,
    alpha=alpha,
    beta=beta,
    omega=omega_value,
    epsilon=eps,
    seed= 1,
    outer_percentile=95,
    I_total=I_total,
    I_ind=I_ind)

    from pathlib import Path

    run_dir = Path(run_dir)
    iterations_radial = []
    iterations_tangential = []
    print(f"---------------Radial Indices-------------")
    for i in range(len(target_element_indices_radial)):
        target_indices = [target_element_indices_radial[i]]
        print(f'-----------[Active Set Call Iteration {i}]: Target_indice {target_indices}--------------------------')
        t_start_setup = time.time()
        e_tilde_radial = np.zeros((n_grid,1))
        e_tilde_tangential = np.zeros((n_grid,1))

        # Setting up active Set for this specific target
        B_sliced, omega_sliced, e_tilde_sliced, omega,e_tilde = active_set_setup(B,omega_value, omega_value_non_target, grey_matter_indices,target_indices,radial_vectors[i])
        t_end_setup = time.time()
        print(f"[TIME] Setup (omega, e_tilde): {t_end_setup - t_start_setup:.3f}s")

        # Active Set call 
        print(f'Starting Active Set algorithm for target......')
        t_start_iteration = time.time()
        
        exit_status, I_radial, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count = active_set_evaluation(alpha,beta,B_sliced,e_tilde_sliced, n_chan,omega_sliced,eps,verbosity=verbosity, I_total = I_total, I_ind = I_ind)
        t_end_iteration = time.time()
        print(f"[TIME] Active Set solving for target: {t_end_iteration - t_start_iteration:.3f}s")

        # Checking wohle epsilon-omega constraints
        if check_whole_domain_constraints(B,I_radial,omega,eps,tolerance):
            iterations_radial.append(iteration_count)
            I_total_radial.append(I_radial)
        else:
            iterations_radial.append(4999)
            I_total_radial.append(np.zeros_like(I_radial))
        
    print(f"---------------Tangential Indices-------------")
    for i in range(len(target_element_indices_tangential)):
        target_indices = [target_element_indices_tangential[i]]
        print(f'-----------[Active Set Call Iteration {i}]: Target_indice {target_indices}--------------------------')
        t_start_setup = time.time()
        omega_iteration = omega
        e_tilde_tangential = np.zeros((n_grid,1))
        e_tilde_tangential = np.zeros((n_grid,1))

        # Setting up active Set for this specific target
        B_sliced, omega_sliced, e_tilde_sliced,omega, e_tilde = active_set_setup(B,omega_value, omega_value_non_target, grey_matter_indices,target_indices,tangential_vectors[i])
        t_end_setup = time.time()
        print(f"[TIME] Setup (omega, e_tilde): {t_end_setup - t_start_setup:.3f}s")

        # Active Set call 
        print(f'Starting Active Set algorithm for target......')
        t_start_iteration = time.time()
        exit_status, I_tangential, optimal_lagrange_eq, optimal_lagrange_ineq, opt_func_value, iteration_count = active_set_evaluation(alpha,beta,B_sliced,e_tilde_sliced, n_chan,omega_sliced,eps,verbosity=verbosity, I_total = I_total, I_ind = I_ind)
        t_end_iteration = time.time()
        print(f"[TIME] Active Set solving for target: {t_end_iteration - t_start_iteration:.3f}s")

        # Checking wohle epsilon-omega constraints
        if check_whole_domain_constraints(B,I_tangential,omega,eps,tolerance):
            iterations_tangential.append(iteration_count)
            I_total_tangential.append(I_tangential)
        else:
            iterations_tangential.append(4999)
            I_total_tangential.append(np.zeros_like(I_tangential))



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
    stats_file_rad = run_dir / "rad_statistics.npy"
    stats_file_tan = run_dir / "tan_statistics.npy"


    np.save(rad_I_file, I_total_radial)
    print(I_total_radial[0])
    np.save(rad_iterations_file, np.array(iterations_radial))
    np.save(tang_I_file, I_total_tangential)
    np.save(tang_iterations_file, np.array(iterations_tangential))
    np.save(targets_tan_file, np.array(target_element_indices_tangential))
    np.save(targets_tan_vector_file, tangential_vectors)
    np.save(targets_radial_file, np.array(target_element_indices_radial))
    np.save(targets_radial_vector_file, radial_vectors)



    print(f'Saved under {run_dir}')

    # Write Vtk files for visualization for one target 
    rng = np.random.default_rng(seed=2)
    if head_model == "Sphere":
        random_indice = rng.integers(0,n_targets_sphere)
    else:
        if one_artificial_target:
            random_indice = 0        
        else:
            random_indice = rng.integers(0,n_targets_head)

    from master_thesis.admm.admm_implementation.io_utils import compute_stats
    print(f'-------------------------------------------------------------')
    print(f'--------Computing statististics for radial/quasiradial target with indice {target_element_indices_radial[random_indice]}--------')
    cda_rad,foc_rad,cdt_rad,par_rad = compute_stats(B,I_radial,target_element_indices_radial[random_indice],radial_vectors[random_indice])
    print(f'CDA: {cda_rad:.4f}, FOC: {foc_rad:.8f}, CDT: {cdt_rad:.4f}, PAR: {par_rad:.4f}')
    print(f'--------Computing statististics for tangential/quasitangential target with indice {target_element_indices_tangential[random_indice]}--------')
    cda_tan,foc_tan,cdt_tan,par_tan = compute_stats(B,I_tangential,target_element_indices_tangential[random_indice],tangential_vectors[random_indice])
    print(f"CDA: {cda_tan:.4f}, FOC: {foc_tan:.8f}, CDT: {cdt_tan:.4f}, PAR: {par_tan:.4f}")
    np.save(stats_file_rad, np.array([cda_rad,foc_rad,cdt_rad,par_rad]))
    np.save(stats_file_tan, np.array([cda_tan,foc_tan,cdt_tan,par_tan]))
    print(f'-------------------------------------------------------------')
    rad_stats.append(np.array([cda_rad,foc_rad,cdt_rad,par_rad]))
    tan_stats.append(np.array([cda_tan,foc_tan,cdt_tan,par_tan]))
    # Assembling for printing 
    e_tilde =np.zeros(B.shape[0])
    e_tilde_2 =np.zeros(B.shape[0])
    e_tilde[3*target_element_indices_radial[random_indice]:3*(target_element_indices_radial[random_indice]+1)] = radial_vectors[random_indice]
    e_tilde_2[3*target_element_indices_tangential[random_indice]:3*(target_element_indices_tangential[random_indice]+1)] = tangential_vectors[random_indice]
    B_ng_zero = np.zeros_like(B)
    dof_indices = np.concatenate([
        3 * grey_matter_indices,
         3 * grey_matter_indices + 1,
         3 * grey_matter_indices + 2
    ])
    B_ng_zero[dof_indices] = B[dof_indices]
    _write_admm_results(I_total_radial[random_indice],meshfile,registered_sensors_path,run_dir,'',B_ng_zero,e_tilde,target_element_indices_radial[random_indice],B.shape[0],dp,second_I=I_total_tangential[random_indice], e_tilde_2=e_tilde_2)
print('-------------------- Total Radial stats --------------------')
for i in range(len(rad_stats)):
    print(f"CDA: {rad_stats[i][0]:.4f}, FOC: {rad_stats[i][1]:.4f}, CDT: {rad_stats[i][2]:.4f}, PAR: {rad_stats[i][3]:.4f}")
print('-------------------- Total Tangential stats --------------------')
for i in range(len(tan_stats)):
    print(f"CDA: {tan_stats[i][0]:.4f}, FOC: {tan_stats[i][1]:.4f}, CDT: {tan_stats[i][2]:.4f}, PAR: {tan_stats[i][3]:.4f}")
