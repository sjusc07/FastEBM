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
   conda create --name fastebm_env python=3.12
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

FastEBM Main Functions
============
1) First, you can create a FastEBM object as follows
```
my_fastebm = FastEBM(data = X, ordering_method = 'median_template', zFastEBM = True, n_neighbors = 1, anamoly_model = 'l2', return_dps= True, seed = None)
```
`X` is your z-scored data with the last column being the stage value. Control subjects should have `stage==1`. The stage value for non-controls are redundant and is not used in the FastEBM calculations.

`Ordering method` determines the reference subject. `median_template` uses the median of controls as the reference subject. `mean_template` is another option, which calculates the mean of controls. 

`zFastEBM` is a boolean parameter. If set to `True`, z-score version of FastEBM will run. 

`n_neighbors` determines the number of neighbors to be used in the construction of the similarity matrix (equation 2 of the FastEBM paper). It is recommended that number of neighbors to be smallest number while the resulting similarity matrix remains connected and the condition number of matrix is 'reasonable` in order to get the fine-grained structure of the manifold. 

`anamoly_model` is used to determine which model to use for the anamoly detection algorithm. A list of models can be found on the ruptures website under *Cost functions* (https://centre-borelli.github.io/ruptures-docs/code-reference/)

`return_dps` is a boolean variable. If set to `True` it returns the dynamic progression score (DPS) for each subject in the sorted data. DPS is useful for some post-processing (see the tutorial).

`seed` sets the random seed for calculations.

2) Using `my_fastebm`, we can get the subject ordering
```
X_fastebm_ordered = my_fastebm.get_subject_ordering()
```
This function returns a dataframe with the subjects ordered according to the dynamic progression score (DPS). DPS indicates how far is the subject relative to the reference subject along the disease continuum. 

3) Once subjects are ordered, we can order the features
```
fastebm_ordered_biomarkers, event_index_dict = my_fastebm.get_event_ordering()
```
`get_event_ordering()` returns a list containing the ordered biomarkers (`fastebm_ordered_biomarkers`), and a dictionary (`event_index_dict`) where the keys are feature names and the values are the indices of abnormality as determined by the changepoint detection algorithm. The event-index dictionary can be useful to check where the abnormalities are happening (same index or far apart).

FastEBM Beta Functions
============
FastEBM's package offers two preprocessing functions for detecting and merging correlated features. Having a high number of correlated features can negatively affect the performance of FastEBM.

1) Detecting correlated features
```
best_corr_clusters = detect_correlated_features(df = df.iloc[:,:-1])
```
`detect_correlated_features` takes in the dataframe (without the 'stage' column) as input and returns a dictionary (tuple -> float64) where the keys are correlated feature clusters and the value is the average correlation value. 

2) Merging correlated features
```
df_decorr = merge_correlated_features(df = df.iloc[:,:-1], correlated_clusters = list(best_corr_clusters.keys()))
```
`merge_correlated_features` takes in the data dataframe (without stage column) and a list of tuples, where each tuple contains the correlated feature names. Users can use the output of `detect_correlated_features` or define their own set of features. Grouped features are mapped to the first principal component using PCA algorithm.

FastEBM Tutorial
===============  
See the jupyter notebook (tests/FastEBM_example.ipynb) in the tests folder for a tutorial on how to use FastEBM using simulated and real data.

Relevant Papers
============
Methods:
- The FastEBM algorithm: [Javid et al. 2025](https://doi.org/10.1007/978-3-032-07904-6_19) 
- Diffusion Maps: [Coifman, R.R., Lafon, S. 2006](https://doi.org/10.1016/j.acha.2006.04.006)
- Diffusion Pseudotime [Haghverdi et al. 2016](https://doi.org/10.1038/nmeth.3971)

Funding
================
xyz