# Detailed studies of numerical methods for optimized multi-channel transcranial direct current stimulation
## Master Thesis: Valentin Adam - University of Münster.

## General
This Git repository contains all the relevant code that was developed for the results in my thesis. 

## Requires
To reproduce the results that were presented in this thesis one would need to: 

- Install the DUNEURO dependencies from https://gitlab.dune-project.org/malte.hoeltershinken/duneuro-optimization/-/tree/a18b19c6d4f9503ad539300dd5e85936398649cd (Commit Hash:a18b19c6)
- Add one Python Binding in that code and some minor code changes according to APPENDIX B in Thesis
- Additionally to actually write the VTK files the code also uses DUNEURO which would need to be installed as well 
- Fill the admm/configs.ini File with the Pathes to the DUNEURO installations/the meshes/and the forward operators (that can be computed via DUNEURO)
- The sphere mesh that was used can be found and downloaded here: https://uni-muenster.sciebo.de/s/DNj9a2NHnGGi6EY
- For the realistic mesh please contact me under vadam@uni-muenster.de because i cannot publish this mesh that was build with personal data

## Stucture and important Files
The structure is as follows:

- The folder admm contains all relevant code for the admm and the pure active_set_approach. Additionally it contains the test_scripts which all data in the thesis can be reproduced
    - admm_implementation: contains all the admm implementation
    - active_set_wrapper: contains all the active set code needed to use the active set approach for the Wagner et al. proposed tdcs Problem
    - admm_test.py: is a script to run the admm on some example parameters and data
    - active_set_test.py: is a script to run the active set on some example parameters and data
    - eps_var.py: calls the active set for varying epsilon values (ref Thesis Section 6.2.1)
    - alpha_beta_var.py: calls the active set for varying alpha beta values (ref Thesis Section 6.2.2)
    - Branch_and_Bound_and_heuristic.py: calls the Branch and Bound algorithm or the heuristic approaches (ref Thesis Section 6.3.2)

- The folder branch_bound contains just the Branch and Bound implementation and the heuristic approaches for the additional constraint

## Usage 

The python files can be executed in the normal way, while you are in the admm folder just call the python files.