FastEBM
============

FastEBM is an algorithm for discrete disease progression modeling. This repository is the Python implementation of FastEBM.

Acknowledgement
================
If you use FastEBM, please cite the following paper:
- [The FastEBM paper](https://doi.org/10.1007/978-3-032-07904-6_19)

Installation
============
## Conda env + Clone repository

1) Create a conda environment:
   ```
   conda create --name fastebm_env python=3.9
   ```

2) Activate your conda environment:
   ```
   conda activate fastebm_env
   ```

3) To use the environment in Jupyter notebook run the following commands:
   ```
   conda install -c anaconda ipykernel
   python -m ipykernel install --user --name=fastebm_env
   ```

4) Clone/download this repo: [cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

5) Navigate to the main FastEBM directory (where you see pyproject.toml, README.txt, LICENSE.txt, and all subfolders), then run:

       pip install .

    Alternatively, you can do `pip install -e .` where the `-e` flag allows you to make edits to the code without reinstalling.
   
6) You should now be able to run FastEBM_example.ipynb or create your own notebook using the fastebm_env environment.

Dependencies
============
- Python >= 3.9.18
- [setuptools](https://pypi.org/project/setuptools/)
- [Pandas](https://github.com/pandas-dev/pandas)
- [NumPy](https://github.com/numpy/numpy) >= 1.26.24
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [ruptures](https://github.com/deepcharles/ruptures/) for changepoint detection
- [Scikit-learn](https://scikit-learn.org)
- [statsmodels](https://github.com/statsmodels/statsmodels/)

FastEBM Tutorial
===============  
See the jupyter notebook (FastEBM_example.ipynb) in the tests folder for a tutorial on how to use FastEBM using simulated and real data.

Relevant Papers
============
Methods:
- The FastEBM algorithm: [Javid et al. 2025](https://doi.org/10.1007/978-3-032-07904-6_19) 
- Diffusion Maps: [Coifman, R.R., Lafon, S. 2006](https://doi.org/10.1016/j.acha.2006.04.006)
- Diffusion Pseudotime [Haghverdi et al. 2016](https://doi.org/10.1038/nmeth.3971)

Funding
================
xyz