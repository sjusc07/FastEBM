FastEBM
============

> [*"Problem solving is often a matter of cooking up an appropriate Markov chain."*](https://math.uchicago.edu/~shmuel/Network-course-readings/Markov_chain_tricks.pdf)

>

> Olle Häggström
---

FastEBM is an algorithm for discrete disease progression modeling. This repository is the Python implementation of FastEBM.

Current package features include deterministic FastEBM event ordering, correlated-feature detection and PCA-based merging, vectorized single-breakpoint L2 changepoint detection, and first-passage sensitivity analysis for propagating Markov-chain variability through subject Disease Progression Scores (DPS).

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

5) Navigate to the main FastEBM directory (where you see pyproject.toml, README.md, LICENSE.txt, and all subfolders), then run:

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

`ordering_method` determines the reference subject. `median_template` uses the median of controls as the reference subject. `mean_template` is another option, which calculates the mean of controls. 

`zFastEBM` is a boolean parameter. If set to `True`, z-score version of FastEBM will run. 

`n_neighbors` determines the number of neighbors to be used in the construction of the similarity matrix. It is recommended that number of neighbors to be smallest number while the resulting similarity matrix remains connected and the condition number of matrix is 'reasonable` in order to get the fine-grained structure of the manifold. 

`anamoly_model` is used to determine which model to use for the anamoly detection algorithm. For the default `anamoly_model='l2'`, FastEBM uses a vectorized one-breakpoint sum-of-squared-errors implementation that matches the single-breakpoint L2 search grid used by `ruptures.Dynp`, while avoiding repeated per-feature `ruptures` calls. Other `anamoly_model` values continue to use the ruptures implementation (https://centre-borelli.github.io/ruptures-docs/code-reference/).

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

First-Passage Sensitivity Analysis
============
FastEBM provides a native first-passage sensitivity analysis for propagating Markov-chain variability through subject DPS and into event ordering. This is exposed through `first_passage_uncertainty`.

```
from fastebm import first_passage_uncertainty

first_passage_result = first_passage_uncertainty(
    data=X,
    n_noise_samples=100,
    ordering_method='median_template',
    n_neighbors=1,
    anamoly_model='l2',
    min_segment_size=2,
    dps_noise_scale=None,
    normalize_subject_variance=True,
    convergence_check_interval=None,
    convergence_min_samples=None,
    event_rank_convergence_tol=None,
    convergence_patience=2,
    seed=None,
)
```

`X` should be formatted the same way as the deterministic FastEBM input: features in all columns except the last column, and stage labels in the last column. Control subjects should have `stage==1`.

`n_noise_samples`: Number of Monte Carlo DPS perturbation samples to draw. If adaptive convergence is not enabled, exactly this many samples are generated. If convergence checking is enabled, this value is treated as the maximum number of samples.

`ordering_method` determines the reference subject. `median_template` uses the median of controls as the reference subject. `mean_template` is another option, which calculates the mean of controls. 

`n_neighbors` determines the number of neighbors to be used in the construction of the similarity matrix.

`min_segment_size`: Minimum number of subjects required on each side of a changepoint.

`dps_noise_scale`: Optional multiplicative scale for the DPS perturbation noise. If `None`, FastEBM automatically estimates a scale from the median local spacing of the normalized DPS values.

`normalize_subject_variance`: Controls how first-passage variances are converted to DPS-scale noise variances. If `True`, each subject’s first-passage variance is divided by the median positive first-passage variance across subjects and then scaled by `dps_noise_scale`. If `False`, the variance is instead converted to the normalized DPS scale using the squared raw DPS range.

`convergence_check_interval`: Number of Monte Carlo samples between adaptive convergence checks. If `None`, convergence is not checked unless a convergence tolerance is supplied. When convergence is enabled and this value is `None`, FastEBM chooses a default checkpoint interval.

`convergence_min_samples`: Minimum number of Monte Carlo samples that must be generated before adaptive stopping is allowed. This prevents early stopping before enough samples have accumulated to estimate stable event-rank summaries.

`event_rank_convergence_tol`: Optional tolerance for adaptive convergence based on mean biomarker event ranks. At each checkpoint, FastEBM compares the current cumulative mean event rank for each biomarker with the previous checkpoint. Sampling can stop once the maximum absolute change across biomarkers is below this tolerance for the required number of consecutive checks. If `None`, event-rank convergence is not used.

`convergence_patience`: Number of consecutive convergence checks that must satisfy the specified tolerance before Monte Carlo sampling stops early. Larger values make early stopping more conservative.

The result is a `FastEBMFirstPassageUncertaintyResult` object with the following commonly used outputs:

- `event_rank_samples`: sampled event ranks for each feature.
- `event_position_samples`: sampled changepoint/DPS positions for each feature.
- `event_summary`: feature-level rank and position summaries, including means, medians, standard deviations, and 95% intervals.
- `subject_rank_samples`: sampled subject ranks after DPS perturbation.
- `subject_rank_summary`: subject-level rank summaries.
- `subject_variance`: first-passage variance and scaled DPS-noise variance for each subject.
- `mean_first_passage_matrix` and `first_passage_variance_matrix`: Kemeny-Snell first-passage moment matrices.
- `sample_metadata`: sampling and convergence metadata.
- `convergence_history`: convergence diagnostics when adaptive stopping is enabled.

Correlated Feature Functions
============
FastEBM offers preprocessing functions for detecting and merging correlated features.

1) Detecting correlated features
```
from fastebm import detect_correlated_features

best_corr_clusters = detect_correlated_features(df = df.iloc[:,:-1])
```
`detect_correlated_features` takes in the dataframe (without the 'stage' column) as input and returns a dictionary (tuple -> float64) where the keys are correlated feature clusters and the value is the average correlation value. 

2) Merging correlated features
```
from fastebm import merge_correlated_features

df_decorr = merge_correlated_features(df = df.iloc[:,:-1], correlated_clusters = best_corr_clusters)
```
`merge_correlated_features` takes in the data dataframe (without stage column) and a dictionary of correlated clusters, such as the output of `detect_correlated_features`. Users can also provide their own cluster dictionary. Grouped features are mapped to the first principal component using PCA.

FastEBM Tutorial
===============  
See the jupyter notebook (tests/FastEBM_example.ipynb) in the tests folder for a tutorial on how to use FastEBM using simulated and real data.

Relevant Papers
============
Methods:
- The FastEBM algorithm: [Javid et al. 2025](https://doi.org/10.1007/978-3-032-07904-6_19) 
- Diffusion Maps: [Coifman, R.R., Lafon, S. 2006](https://doi.org/10.1016/j.acha.2006.04.006)
- Diffusion Pseudotime [Haghverdi et al. 2016](https://doi.org/10.1038/nmeth.3971)
- Offline changepoint detection: [Truong, Oudre, Vayatis 2020](https://doi.org/10.1016/j.sigpro.2019.107299)
- First-passage moments for finite Markov chains: Kemeny and Snell, *Finite Markov Chains*, 1960

Funding
================
This work was supported by grant R01AG087513 from the National Institute on Aging at the National Institutes of Health, and by grant AARG-23-1149996 from US Alzheimer’s Association.
