import os
import numpy as np
import numpy as np
import h5py as h5
from pathlib import Path

def _write_admm_results(scaled_I, meshfile, registered_sensors_path, outputdir, subdir, B, e_tilde, target_indices, n_chan, dp, second_I=None, third_I=None, all_targets_data= None,e_tilde_2=None):
    """
    Extracted printing / VTK export routine from comparison_admm.py.
    Keeps original behaviour and filenames, writes into outputdir/subdir.
    """
    out_path = Path(outputdir) / subdir if subdir else Path(outputdir)
    out_path.mkdir(parents=True, exist_ok=True)

    vc_data = np.load(meshfile)
    nodes = vc_data['nodes']
    elements = vc_data['elements']
    labels = vc_data['labels']
    conductivities = vc_data['conductivities']

    mesh_info = vc_data['mesh_info']
    for info in mesh_info:
        if info[0] == 'type':
            fem_type = info[1]
        elif info[0] == 'element_type':
            if info[1] == 'tetrahedrons':
                el_type = 'tetrahedron'
                dim = 3
            elif info[1] == 'hexahedrons':
                el_type = 'hexahedron'
                dim = 3
            elif info[1] == 'triangles':
                el_type = 'triangle'
                dim = 2
            elif info[1] == 'quadrangles':
                el_type = 'quadrangle'
                dim = 2
            else:
                raise ValueError('DUNEuro currently only supports tetrahedral and hexahedral elements for 3d meshes, and triangles and quadrangles for 2d meshes')

    mesh_cfg = {'nodes' : nodes, 'elements' : elements}
    tensor_cfg = {'labels' : labels, 'conductivities' : conductivities}
    vc_cfg = {'grid' : mesh_cfg, 'tensors' : tensor_cfg}
    driver_cfg = {'type' : str(fem_type), 'solver_type' : 'cg', 'element_type' : str(el_type), 'post_process' : 'true', 'subtract_mean' : 'true'}
    solver_cfg = {'reduction' : '1e-16', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly', 'verbose' : '0'}
    meg_cfg = {'intorderadd' : '2', 'type' : 'physical'}
    driver_cfg['solver'] = solver_cfg
    driver_cfg['meg'] = meg_cfg
    driver_cfg['volume_conductor'] = vc_cfg
    driver_cfg['numberOfThreads'] = os.cpu_count()
    meeg_driver = dp.MEEGDriver3d(driver_cfg)

    element_statistics = meeg_driver.elementStatistics()
    evaluation_positions = np.array(element_statistics['elementCenters'])
    del element_statistics

    # perform function evaluation
    wrapped_evaluation_positions = [dp.FieldVector3D(pos) for pos in evaluation_positions]

    VTKWriterTarget = dp.PointVTKWriter3d(wrapped_evaluation_positions, True)

    wrapped_target = [dp.FieldVector3D(e_tilde[3*i:3*(i+1)]) for i in range(len(e_tilde)//3)]
    VTKWriterTarget.addVectorData('desired_current_density',wrapped_target)
    if e_tilde_2 is not None:
        wrapped_target_2 = [dp.FieldVector3D(e_tilde_2[3*i:3*(i+1)]) for i in range(len(e_tilde_2)//3)]
        VTKWriterTarget.addVectorData('desired_current_density_tangential',wrapped_target_2)

    VTKWriterTarget.write(str(out_path / "admm_target_current_density"))    

    sensor_data = np.load(registered_sensors_path)
    electrode_positions = sensor_data['electrode_positions']


    VTKWriterElecs = dp.PointVTKWriter3d([dp.FieldVector3D(pos) for pos in electrode_positions], True)
    VTKWriterElecs.addScalarData('current_injection', scaled_I)
    if second_I is not None:
        VTKWriterElecs.addScalarData('current_injection_second_last', second_I)
    if third_I is not None:
        VTKWriterElecs.addScalarData('current_injection_third_last', third_I)
    VTKWriterElecs.write(str(out_path / "admm_electrodes"))

    print('notebook cwd:', os.getcwd())
    print('outputdir abs:', os.path.abspath(outputdir))
    print('files:', os.listdir(outputdir) if os.path.isdir(outputdir) else 'no folder yet')


    evaluated_values = B @ scaled_I
    wrapped_evaluation = [dp.FieldVector3D(evaluated_values[3*i:3*(i+1)]) for i in range(len(evaluated_values)//3)]
    VTKWriterEval = dp.PointVTKWriter3d(wrapped_evaluation_positions, True)
    VTKWriterEval.addVectorData('resulting_current_density',wrapped_evaluation)
    if second_I is not None:
        evaluated_values_second = B @ second_I
        wrapped_evaluation_second = [dp.FieldVector3D(evaluated_values_second[3*i:3*(i+1)]) for i in range(len(evaluated_values_second)//3)]
        VTKWriterEval.addVectorData('resulting_current_density_second_last',wrapped_evaluation_second)
    if third_I is not None:
        evaluated_values_third = B @ third_I
        wrapped_evaluation_third = [dp.FieldVector3D(evaluated_values_third[3*i:3*(i+1)]) for i in range(len(evaluated_values_third)//3)]
        VTKWriterEval.addVectorData('resulting_current_density_third_last',wrapped_evaluation_third)
    if all_targets_data is not None:
        target_indices = np.asarray(all_targets_data[0]).astype(int).ravel()
        radial_vectors = np.asarray(all_targets_data[1])
        tangential_vectors = np.asarray(all_targets_data[2])

        n_elements = B.shape[0] // 3
        zero_vec = np.zeros(3)

        radial_map = {
            int(global_idx): radial_vectors[local_idx]
            for local_idx, global_idx in enumerate(target_indices)
        }

        tangential_map = {
            int(global_idx): tangential_vectors[local_idx]
            for local_idx, global_idx in enumerate(target_indices)
        }

        target_radial = []
        target_tangential = []

        for i in range(n_elements):
            radial_vec = radial_map.get(i, zero_vec)
            tangential_vec = tangential_map.get(i, zero_vec)

            target_radial.append(dp.FieldVector3D(radial_vec))
            target_tangential.append(dp.FieldVector3D(tangential_vec))

        VTKWriterEval.addVectorData("radial_targets", target_radial)
        VTKWriterEval.addVectorData("tangential_targets", target_tangential)
        
    VTKWriterEval.write(str(out_path / "admm_resulting_current_density"))

def _fmt(x):
    """Helper function for stable file directories."""
    if x is None:
        return "None"
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.12g}"
    return str(x).replace(" ", "")

def create_admm_run_directory(outputdir, subdir, target_indices= None, gr_or_ngr=None, abs_tol=None, rel_tol=None, max_iterations=None, alpha=None, beta=None, omega=None, epsilon=None, is_converged=None, update_mu_every=None, seed=None, outer_percentile=None):
    """
    Create ADMM run directory under outputdir/subdir/<run_name>.
    
    Directory name structure:
    [target_indices with - separated]_[GR|NGR]_[abs_tol]_[rel_tol]_[max_iterations]_
    [alpha]_[beta]_[epsilon]_[omega]_[converged_after_iterations]_[mu_update_frequency]_[seed]_[outer_percentile]

    Returns
    run_dir: Absolute path to run directory
    run_name : Directory name only (without path)
    """


    # target_indices -> "1-3-7"
    if target_indices is None:
        target_part = "None"
    else:
        target_part = "-".join(str(int(i)) for i in target_indices)


    if is_converged is not None:
        run_name = (
            f"{target_part}_"
            f"{_fmt(gr_or_ngr)}_"
            f"{_fmt(abs_tol)}_"
            f"{_fmt(rel_tol)}_"
            f"{_fmt(max_iterations)}_"
            f"{_fmt(alpha)}_"
            f"{_fmt(beta)}_"
            f"{_fmt(epsilon)}_"
            f"{_fmt(omega)}_"
            f"{_fmt(is_converged)}"
            f"{'_muUpdate'+_fmt(update_mu_every) if update_mu_every is not None else ''}"
        )
    else:
        if seed is not None and outer_percentile is not None:
            run_name = (
                f"{_fmt(gr_or_ngr)}_"
                f"{_fmt(abs_tol)}_"
                f"{_fmt(rel_tol)}_" 
                f"{_fmt(alpha)}_"
                f"{_fmt(beta)}_"
                f"{_fmt(epsilon)}_"
                f"{_fmt(omega)}_"
                f"{'_muUpdate'+_fmt(update_mu_every) if update_mu_every is not None else ''}_"
                f"{_fmt(seed)}_"
                f"{_fmt(outer_percentile)}_"
            )
        else:
            run_name = (
                f"{target_part}_"
                f"{_fmt(gr_or_ngr)}_"
                f"{_fmt(abs_tol)}_"
                f"{_fmt(rel_tol)}_"
                f"{_fmt(max_iterations)}_"
                f"{_fmt(alpha)}_"
                f"{_fmt(beta)}_"
                f"{_fmt(epsilon)}_"
                f"{_fmt(omega)}_ "
                f"{'_muUpdate'+_fmt(update_mu_every) if update_mu_every is not None else ''}"
            )
    

    run_dir = os.path.join(outputdir, subdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, run_name

def create_B_and_B_run_directory(outputdir, subdir, alpha, beta, omega, epsilon, n_usable, seed = 1, I_ind=None, I_total=None, heruristic=None):
    """
    Create Branch and Bound run directory under outputdir/subdir/<run_name>.
    
    Directory name structure:
    [alpha]_[beta]_[epsilon]_[omega]_[n_usable_electrodes]_[I_ind_constr]_[I_total_constr]_[heuristic]

    Returns
    run_dir: Absolute path to run directory
    run_name : Directory name only (without path)
    """

    run_name = (
            f"{_fmt(alpha)}_"
            f"{_fmt(beta)}_"
            f"{_fmt(epsilon)}_"
            f"{_fmt(omega)}_"
            f"{'_nUsable'+_fmt(n_usable) if n_usable is not None else ''}_"
            f"{'_Iind'+_fmt(I_ind) if I_ind is not None else ''}_"
            f"{'_Itotal'+_fmt(I_total) if I_total is not None else ''}_"
            f"{'_heuristic'+_fmt(heruristic) if heruristic is not None else ''}_"
        )
    
    run_dir = os.path.join(outputdir, subdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, run_name

def create_active_set_run_directory(outputdir,subdir,target_indices=None,abs_tol=None,rel_tol=None,max_iterations=None,alpha=None,beta=None,omega=None,epsilon=None,n_usable_electrodes=None, seed=None, outer_percentile=None, I_ind=None,I_total=None, omega_target_value=None ):
    """
    Create Active Set run directory under outputdir/subdir/<run_name>.
    
    Directory name structure:
    [target_indices with - separated]_[abs_tol]_[rel_tol]_
    [alpha]_[beta]_[epsilon]_[omega]_[seed]_[outer_percentile]_[I_total]_[I_ind]

    Returns
    run_dir: Absolute path to run directory
    run_name : Directory name only (without path)
    """

    # target_indices -> "1-3-7"
    if target_indices is None:
        target_part = "None"
    else:
        target_part = "-".join(str(int(i)) for i in target_indices)

    if seed is not None and outer_percentile is not None:
        if I_total is not None:
            run_name = (
                f"{target_part}_"
                f"{_fmt(abs_tol)}_"
                f"{_fmt(rel_tol)}_" 
                f"{_fmt(alpha)}_"
                f"{_fmt(beta)}_"
                f"{_fmt(epsilon)}_"
                f"{_fmt(omega)}_"
                f"{_fmt(seed)}_"
                f"{_fmt(outer_percentile)}_"
                f"{'_Itotal'+_fmt(I_total)}_"
                f"{_fmt(omega_target_value)}"
            )
            if I_ind is not None:
                run_name += f"{'_Iind'+_fmt(I_ind)}"
        else:
            run_name = (
                f"{target_part}_"
                f"{_fmt(abs_tol)}_"
                f"{_fmt(rel_tol)}_" 
                f"{_fmt(alpha)}_"
                f"{_fmt(beta)}_"
                f"{_fmt(epsilon)}_"
                f"{_fmt(omega)}_"
                f"{_fmt(seed)}_"
                f"{_fmt(outer_percentile)}"
        )
    else:
        run_name = (
                f"{target_part}_"
                f"{_fmt(abs_tol)}_"
                f"{_fmt(rel_tol)}_"
                f"{_fmt(max_iterations)}_"
                f"{_fmt(alpha)}_"
                f"{_fmt(beta)}_"
                f"{_fmt(epsilon)}_"
                f"{_fmt(omega)}_"
                f"{_fmt(n_usable_electrodes)}"
            )

    run_dir = os.path.join(outputdir, subdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, run_name

def plot_admm_convergence(residual_history_1, residual_history_2, residual_history_3=None,
                          Lagrangian_0_history=None, Lagrangian_history=None,
                          wagner_iter=None, residual_iter=None, include_ref_penalty=False,
                          plot_lagrangian=False):
    """
    Plot ADMM convergence history: Lagrangian and residuals.
    
    Parameters
    ----------
    residual_history_1 : list
        Primal and dual residuals for constraint 1 (I - z).
    residual_history_2 : list
        Primal and dual residuals for constraint 2 (B*I - y).
    residual_history_3 : list, optional
        Primal and dual residuals for constraint 3 (1^T I - x).
    Lagrangian_0_history : list, optional
        Lagrangian (without penalty terms) history.
    Lagrangian_history : list, optional
        Lagrangian (with penalty terms) history.
    wagner_iter : int, optional
        Iteration where Wagner criterion was met.
    residual_iter : int, optional
        Iteration where residual criterion was met.
    include_ref_penalty : bool
        Whether reference electrode penalty is included (3rd constraint).
    plot_lagrangian : bool
        Whether to plot Lagrangian history.
    """
    from matplotlib import pyplot as plt
    
    history_1 = np.array(residual_history_1)
    history_2 = np.array(residual_history_2)
    if include_ref_penalty and residual_history_3:
        history_3 = np.array(residual_history_3)

    if plot_lagrangian and Lagrangian_0_history is not None and Lagrangian_history is not None:
        Lagrangian_0_history = np.array(Lagrangian_0_history)
        Lagrangian_history = np.array(Lagrangian_history)

        n_plots = 4 if include_ref_penalty else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

        # Subplot 0: Lagrangian
        ax = axes[0]
        ax.plot(Lagrangian_0_history, label='Lagrangian (without penalty terms)', marker='o', markersize=3)
        ax.plot(Lagrangian_history, label='Lagrangian (with penalty terms)', marker='s', markersize=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Lagrangian value')
        ax.set_title('Lagrangian History')
        ax.legend()
        ax.grid(True, ls='--', lw=0.5)

        # Subplot 1: Constraint 1 residuals (I - z)
        ax = axes[1]
        ax.semilogy(history_1[:, 0], label='primal residual (I-z)', color='C0')
        ax.semilogy(history_1[:, 1], label='dual residual (I-z)', color='C1')
        if wagner_iter is not None:
            ax.axvline(wagner_iter, color='C2', linestyle='--', label=f'Wagner @ {wagner_iter}')
        if residual_iter is not None:
            ax.axvline(residual_iter, color='C3', linestyle='--', label=f'Residual @ {residual_iter}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('residual (log scale)')
        ax.set_title('Constraint 1: I - z')
        ax.legend()
        ax.grid(True, which='both', ls='--', lw=0.5)

        # Subplot 2: Constraint 2 residuals (B*I - y)
        ax = axes[2]
        ax.semilogy(history_2[:, 0], label='primal residual (BI-y)', color='C4')
        ax.semilogy(history_2[:, 1], label='dual residual (BI-y)', color='C5')
        if wagner_iter is not None:
            ax.axvline(wagner_iter, color='C2', linestyle='--', label=f'Wagner @ {wagner_iter}')
        if residual_iter is not None:
            ax.axvline(residual_iter, color='C3', linestyle='--', label=f'Residual @ {residual_iter}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('residual (log scale)')
        ax.set_title('Constraint 2: BI - y')
        ax.legend()
        ax.grid(True, which='both', ls='--', lw=0.5)

        # Subplot 3: Constraint 3 residuals (1^T I - x)
        if include_ref_penalty:
            ax = axes[3]
            ax.semilogy(history_3[:, 0], label='primal residual (1^T I - x)', color='C6')
            ax.semilogy(history_3[:, 1], label='dual residual (1^T I - x)', color='C7')
            if wagner_iter is not None:
                ax.axvline(wagner_iter, color='C2', linestyle='--', label=f'Wagner @ {wagner_iter}')
            if residual_iter is not None:
                ax.axvline(residual_iter, color='C3', linestyle='--', label=f'Residual @ {residual_iter}')
            ax.set_xlabel('iteration')
            ax.set_ylabel('residual (log scale)')
            ax.set_title('Constraint 3: 1^T I - x')
            ax.legend()
            ax.grid(True, which='both', ls='--', lw=0.5)

        plt.tight_layout()
        plt.show()

    else:
        # Plot only residuals (no Lagrangian)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.semilogy(history_1[:, 0], label='primal residual (I-z)', color='C0')
        plt.semilogy(history_1[:, 1], label='dual residual (I-z)', color='C1')
        if wagner_iter is not None:
            plt.axvline(wagner_iter, color='C2', linestyle='--', label=f'Wagner @ {wagner_iter}')
        if residual_iter is not None:
            plt.axvline(residual_iter, color='C3', linestyle='--', label=f'Residual @ {residual_iter}')
        plt.xlabel('iteration')
        plt.ylabel('residual (log scale)')
        plt.title('Constraint 1: I - z')
        plt.legend()
        plt.grid(True, which='both', ls='--', lw=0.5)

        plt.subplot(1, 2, 2)
        plt.semilogy(history_2[:, 0], label='primal residual (BI-y)', color='C4')
        plt.semilogy(history_2[:, 1], label='dual residual (BI-y)', color='C5')
        if wagner_iter is not None:
            plt.axvline(wagner_iter, color='C2', linestyle='--', label=f'Wagner @ {wagner_iter}')
        if residual_iter is not None:
            plt.axvline(residual_iter, color='C3', linestyle='--', label=f'Residual @ {residual_iter}')
        plt.xlabel('iteration')
        plt.ylabel('residual (log scale)')
        plt.title('Constraint 2: BI - y')
        plt.legend()
        plt.grid(True, which='both', ls='--', lw=0.5)

        plt.tight_layout()
        plt.show()

def normalize(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)

def farthest_point_sampling(points, n_samples, rng=None):
    """
    Helper function to get distributed target points.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = points.shape[0]
    selected = np.empty(n_samples, dtype=int)

    selected[0] = rng.integers(n)
    dist2 = np.sum((points - points[selected[0]])**2, axis=1)

    for k in range(1, n_samples):
        selected[k] = np.argmax(dist2)
        new_dist2 = np.sum((points - points[selected[k]])**2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    return selected

def generate_targets_from_element_centers(element_centers, element_labels, target_label=3, n_targets=100, center=None, outer_percentile_within_label=95, inner_percentile_within_label=None, model_type="sphere", seed=42):
    """
    Generates random targets in sphere or head domain. 
    """
    rng = np.random.default_rng(seed)

    element_centers = np.asarray(element_centers)
    element_labels = np.asarray(element_labels).squeeze()

    if center is None:
        center = element_centers.mean(axis=0)

    if model_type not in ["sphere", "head"]:
        raise ValueError(
            f"Unbekannter model_type '{model_type}'. "
            "Erlaubt sind 'sphere' oder 'head'."
        )

    # 1. Nur Elemente mit gewünschtem Label
    label_element_indices = np.where(element_labels == target_label)[0]

    if len(label_element_indices) == 0:
        raise ValueError(f"Keine Elemente mit Label {target_label} gefunden.")

    label_centers = element_centers[label_element_indices]

    # 2. Radien innerhalb des gewünschten Labels berechnen
    label_radii = np.linalg.norm(label_centers - center, axis=1)

    outer_radius_threshold = np.percentile(
        label_radii,
        outer_percentile_within_label,
    )

    # ============================================================
    # Fall 1: sphere
    # Verhalten wie bisher:
    # - radiale und tangentiale Targets sind dieselben Elemente
    # - Kandidaten liegen außen, also >= outer percentile
    # ============================================================
    if model_type == "sphere":
        candidate_local_indices = np.where(
            label_radii >= outer_radius_threshold
        )[0]

        candidate_element_indices = label_element_indices[candidate_local_indices]
        candidate_points = element_centers[candidate_element_indices]

        if len(candidate_element_indices) < n_targets:
            raise ValueError(
                f"Zu wenige Kandidaten: {len(candidate_element_indices)} gefunden, "
                f"aber {n_targets} benötigt. "
                f"Reduziere outer_percentile_within_label, z. B. auf 90 oder 85."
            )

        selected_local = farthest_point_sampling(
            candidate_points,
            n_targets,
            rng=rng,
        )

        target_element_indices = candidate_element_indices[selected_local]
        target_points = element_centers[target_element_indices]

        # Radiale Orientierung
        radial_vectors = normalize(target_points - center)

        # Tangentiale Orientierung
        random_vectors = rng.normal(size=(n_targets, 3))

        tangential_vectors = (
            random_vectors
            - np.sum(random_vectors * radial_vectors, axis=1, keepdims=True)
            * radial_vectors
        )

        tangential_vectors = normalize(tangential_vectors)

        return (
            target_element_indices,
            target_points,
            radial_vectors,
            tangential_vectors,
        )

    # ============================================================
    # Fall 2: head
    # - radiale Targets: wie bisher außen, also >= outer percentile
    # - tangentiale Targets: zwischen inner und outer percentile
    # - separate Rückgabe für radial und tangential
    # ============================================================
    if model_type == "head":
        if inner_percentile_within_label is None:
            raise ValueError(
                "Für model_type='head' muss inner_percentile_within_label "
                "angegeben werden."
            )

        if inner_percentile_within_label >= outer_percentile_within_label:
            raise ValueError(
                "inner_percentile_within_label muss kleiner als "
                "outer_percentile_within_label sein."
            )

        inner_radius_threshold = np.percentile(
            label_radii,
            inner_percentile_within_label,
        )

        # --------------------------------------------------------
        # 2a. Radiale Kandidaten: wie bisher außen
        # --------------------------------------------------------
        radial_candidate_local_indices = np.where(
            label_radii >= outer_radius_threshold
        )[0]

        radial_candidate_element_indices = label_element_indices[
            radial_candidate_local_indices
        ]

        radial_candidate_points = element_centers[
            radial_candidate_element_indices
        ]

        if len(radial_candidate_element_indices) < n_targets:
            raise ValueError(
                f"Zu wenige radiale Kandidaten: "
                f"{len(radial_candidate_element_indices)} gefunden, "
                f"aber {n_targets} benötigt. "
                f"Reduziere outer_percentile_within_label, z. B. auf 90 oder 85."
            )

        selected_radial_local = farthest_point_sampling(
            radial_candidate_points,
            n_targets,
            rng=rng,
        )

        target_element_indices_radial = radial_candidate_element_indices[
            selected_radial_local
        ]

        target_points_radial = element_centers[target_element_indices_radial]

        radial_vectors = normalize(target_points_radial - center)

        # --------------------------------------------------------
        # 2b. Tangentiale Kandidaten: zwischen inner und outer
        # --------------------------------------------------------
        tangential_candidate_local_indices = np.where(
            (label_radii >= inner_radius_threshold)
            & (label_radii <= outer_radius_threshold)
        )[0]

        tangential_candidate_element_indices = label_element_indices[
            tangential_candidate_local_indices
        ]

        tangential_candidate_points = element_centers[
            tangential_candidate_element_indices
        ]

        if len(tangential_candidate_element_indices) < n_targets:
            raise ValueError(
                f"Zu wenige tangentiale Kandidaten: "
                f"{len(tangential_candidate_element_indices)} gefunden, "
                f"aber {n_targets} benötigt. "
                f"Vergrößere den Bereich zwischen "
                f"inner_percentile_within_label und "
                f"outer_percentile_within_label."
            )

        selected_tangential_local = farthest_point_sampling(
            tangential_candidate_points,
            n_targets,
            rng=rng,
        )

        target_element_indices_tangential = tangential_candidate_element_indices[
            selected_tangential_local
        ]

        target_points_tangential = element_centers[
            target_element_indices_tangential
        ]

        # Für tangentiale Orientierung brauchen wir lokale Radialrichtungen
        # an den tangentialen Target-Punkten.
        radial_vectors_for_tangential = normalize(
            target_points_tangential - center
        )

        random_vectors = rng.normal(size=(n_targets, 3))

        tangential_vectors = (
            random_vectors
            - np.sum(
                random_vectors * radial_vectors_for_tangential,
                axis=1,
                keepdims=True,
            )
            * radial_vectors_for_tangential
        )

        tangential_vectors = normalize(tangential_vectors)

        return (
            target_element_indices_radial,
            target_points_radial,
            radial_vectors,
            target_element_indices_tangential,
            target_points_tangential,
            tangential_vectors,
        )
    
def h5_to_np_array(h5_file_path_current_densities, h5_file_path_element_centers, h5_file_path_element_labels):
    """
    Converts hf files to numpy arrays. 
    Inputs are paths to the h5 files for current densities, element centers, and element labels.
    Outputs are the corresponding numpy arrays.
    """

    with h5.File(h5_file_path_current_densities, 'r') as h5_file:
        current_densities_dataset = h5_file['current_densities']
        current_densities_array =  np.array(current_densities_dataset)

    with h5.File(h5_file_path_element_centers, 'r') as h5_file:
        element_centers_dataset = h5_file['element_centers']
        element_centers_array =  np.array(element_centers_dataset)
    
    with h5.File(h5_file_path_element_labels, 'r') as h5_file:
        element_labels_dataset = h5_file['element_labels']
        element_labels_array =  np.array(element_labels_dataset)
    
    return current_densities_array, element_centers_array, element_labels_array

def check_whole_domain_constraints(B,I,omega,epsilon,tolerance):
    """Checks if the omega-epsilon constraint is satisfied at the whole domain."""
    omega = omega.flatten()
    BI = B@I
    check = np.where(np.abs(BI)> epsilon/omega + tolerance)[0]

    if check.shape[0] >=1:
        print(f'Side constraint not passed!!! {check.shape[0]} unvalid nodes!')
        print(f'')
        return False
    else:
        print(f'Side constraint check passed.')
        return True

def compute_stats(B,I,target_indices,target_vector):
    """Ref Wagner et. al. tdcs optimization approach. Computes Intensity focality stats like in Table 3 there. But just for one target here."""
    BI = B@I
    target_vector = target_vector / np.linalg.norm(target_vector)

    BI_without_target = BI
    cda = np.mean(np.linalg.norm(BI[3*target_indices : 3*(target_indices + 1 ),]))
    target_indices_rows = [3*target_indices,3*target_indices+1, 3*target_indices+2]
    BI_without_target = np.delete(BI_without_target,target_indices_rows,axis=0)
    foc = np.linalg.norm(BI_without_target)/BI_without_target.shape[0]
    cdt =  BI[target_indices_rows]@target_vector
    par = cdt/cda

    return cda,foc,cdt,par

