import numpy as np
import sklearn
import pandas as pd
import ruptures as rpt
from collections import deque
import sys

from ._changepoint import l2_one_breakpoint_indices

class FastEBM:
    _VALID_ORDERING_METHODS = {'mean_template', 'median_template'}

    def __init__(self, data, ordering_method = 'median_template', zFastEBM = True, n_neighbors = 5, anamoly_model = 'l2', return_dps = False, seed = None):
        self._check_input(input_data=data)
        self.data = data.iloc[:, :-1]
        self.stage = data.iloc[:, -1]                       # stage/reference group data
        self.feature_names = self.data.columns              # getting the feature names
        if ordering_method not in self._VALID_ORDERING_METHODS:
            valid_methods = ", ".join(sorted(self._VALID_ORDERING_METHODS))
            raise ValueError(f"Unsupported ordering_method '{ordering_method}'. Expected one of: {valid_methods}.")
        self.ordering_method = ordering_method
        self.n_neighbors = n_neighbors
        self.zFastEBM = zFastEBM
        self.anamoly_model = anamoly_model
        self.sorted_data = None                             # sorted subjects
        self.return_dps = return_dps
        self.seed = seed

    def _check_input(self, input_data):
        try:
            if not isinstance(input_data, pd.DataFrame):             
                raise ValueError("Input data should be a Pandas DataFrame.")
        except Exception as e:
            raise ValueError("The input data type can't be identified")

        try:
            if len(input_data.iloc[:,-1].value_counts().index) > 10:
                raise ValueError("Last column should contain stage information (categorical) in numerical format. Please check your data")
        except Exception as e:
            raise ValueError("Error handling the stage information. Please check your data")
        
    def _is_connected(self, graph):
        visited = np.full(len(graph), False)
        num_vertices = len(graph)
        q = deque()
        q.append(0)         # node 0
        visited[0] = True
        while q:
            u = q.popleft()
            for v in range(num_vertices):
                # If an edge exists and the neighbor hasn't been visited
                if graph[u][v] > 0 and not visited[v]:
                    visited[v] = True
                    q.append(v)
        return all(visited)


    def get_subject_ordering(self):
        """
        Core implementation of FastEBM.

        Returns
        -------
        pandas.DataFrame
            DataFrame with subjects ordered along the pseudotime.
            Rows correspond to subjects and columns represent biomarker
            event positions.
        """
        np.random.seed(self.seed)
        data_shape = self.data.shape
        number_of_subjects = data_shape[0]
        omega = np.ones(shape = (number_of_subjects, number_of_subjects))     # this is the stage matrix -- in the z-score version all values are 1
        pariwise_distance = sklearn.metrics.pairwise.pairwise_distances(self.data, self.data, metric = 'nan_euclidean')**2 #pairwise distance between subjects
        np.fill_diagonal(pariwise_distance, 0)
        neighbor_count = min(self.n_neighbors + 1, number_of_subjects)

        nearest_n_dists, _ = sklearn.neighbors.NearestNeighbors(
            n_neighbors=neighbor_count,
            p=2,
            metric='nan_euclidean'
        ).fit(self.data).kneighbors(n_neighbors=neighbor_count)

        epsilon = nearest_n_dists[:, -1]
        epsilon_squared = epsilon**2                        # squred euclidean distance to the n_neighbor neighbor

        # calculating the sum of distances for each subject x and y (epsilon_squared_sum = epsilon_squared_x + epsilon_squared_y) 
        # gamma = omega / (epsilon_squared_x + epsilon_squared_y)
        epsilon_squared_sum = np.zeros(shape = (len(epsilon), len(epsilon)))
        epsilon_squared_sum = epsilon_squared[:, np.newaxis] + epsilon_squared      # numpy vectorized operation instead of looping twice over epsilon_squared
        gamma = epsilon_squared_sum / omega                         # calculating the weighting factor between subjects distance
        S_xy = np.exp(-pariwise_distance/gamma)                     # complete locally-weighted Gaussian kernel (equation 1)

        # If sampling density needs to be handled. 
        # Refer to Coifman, R.R., Lafon, S.: Diffusion maps. Applied and Computational Harmonic Analysis 21(1), 5–30 (2006) for more info on sampling density in diffusion maps
        D_xy = np.sum(S_xy, 1)                                      # sum of kernel values for each subject x for all y
        W_denom = np.outer(D_xy, D_xy)                              # normalization factor for sampling density removal
        W_xy = S_xy / W_denom                                       # new kernel free of sampling density bias
        W_xy[np.where(pariwise_distance == 0)] = 0                  # cleaning up
        # print(W_xy)
        # converting the kernel W_xy to a symmetric stochastic matrix using Chung's method
        # E_x = np.diag(np.sum(W_xy, 1)**(-0.5))
        # T = np.matmul(np.matmul(E_x, W_xy), E_x)                    # T is a symmetric stochastic matrix corresponding to a Markov chain

        # converting the kernel W_xy to a row-stochastic Markov transition matrix
        degree = np.sum(W_xy, axis=1)
        if np.any(degree <= 0) or not np.all(np.isfinite(degree)):
            raise ValueError("Transition matrix construction failed because at least one row has zero degree.")

        T = W_xy / degree[:, np.newaxis]

        if not np.all(np.isfinite(T)):
            raise ValueError("Transition matrix contains non-finite values.")
    
        system = T.T - np.eye(number_of_subjects)
        system[-1, :] = 1.0

        target = np.zeros(number_of_subjects)
        target[-1] = 1.0

        try:
            stationary_distribution = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eig(T.T)
            stationary_distribution = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
            if stationary_distribution.sum() < 0:
                stationary_distribution = -stationary_distribution

        stationary_distribution = np.clip(np.real(stationary_distribution), 0.0, None)
        stationary_total = stationary_distribution.sum()

        if stationary_total <= np.finfo(float).eps or not np.isfinite(stationary_total):
            raise ValueError("Could not compute a finite stationary distribution.")

        stationary_distribution = stationary_distribution / stationary_total
        stationary_matrix = np.tile(stationary_distribution, (number_of_subjects, 1))
        # calculating the fundamental matrix of the Markov chain
        N = np.linalg.inv(np.eye(number_of_subjects) - T + stationary_matrix) - np.eye(number_of_subjects)
        N = N.clip(min = 0)                                         # cleaning up

        # ordering subjects across pseudotime axis
        if self.ordering_method == 'mean_template':
            N_df = pd.DataFrame(N)      # can be done with numpy probably -- to be optimized
            N_df['stage'] = self.stage.to_numpy()
            centroid = (N_df[N_df.stage == 1].mean())[:-1].values   # assumption is that the reference group is labeled 1 and everyone else is 2

        elif self.ordering_method == 'median_template':
            N_df = pd.DataFrame(N)
            N_df['stage'] = self.stage.to_numpy()
            centroid = (N_df[N_df.stage == 1].median())[:-1].values

        DPS = np.sqrt(np.sum((N - centroid) ** 2, axis=1))
        
        self.sorted_data = self.data.copy(deep = True)
        self.sorted_data['DPS'] = list(DPS)
        self.sorted_data = self.sorted_data.sort_values('DPS', ascending = True)
        if self.return_dps == False:
            self.sorted_data = self.sorted_data.drop(['DPS'], axis = 1)
        return self.sorted_data
    
    def get_event_ordering(self):
        np.random.seed(self.seed)
        if self.sorted_data is None:                    # if the user calls event_ordering first
            print("Getting subject ordering first...")
            self.get_subject_ordering()
        event_order_dict = {}
        if self.anamoly_model == 'l2':
            feature_data = self.sorted_data.loc[:, self.feature_names]
            values = feature_data.to_numpy(dtype=float)

            if not np.isnan(values).any():
                breakpoints = l2_one_breakpoint_indices(values, min_size=2)
                if np.all(breakpoints >= 0):
                    event_order_dict = {
                        col: int(breakpoint)
                        for col, breakpoint in zip(self.feature_names, breakpoints)
                    }
                    self.event_ordering = list(
                        {k: v for k, v in sorted(event_order_dict.items(), key=lambda item: item[1])}.keys()
                    )
                    return self.event_ordering, dict(sorted(event_order_dict.items(), key=lambda item: item[1]))

            for col in self.feature_names:
                valid = self.sorted_data[col].dropna()
                values = valid.to_numpy(dtype=float)
                breakpoint = l2_one_breakpoint_indices(values, min_size=2)[0]
                if breakpoint < 0:
                    anamoly_alg = rpt.Dynp(model=self.anamoly_model)
                    anamoly_alg.fit(values)
                    breakpoint = anamoly_alg.predict(n_bkps=1)[0]
                breakpoint = min(max(int(breakpoint), 0), len(valid) - 1)
                event_order_dict[col] = valid.index[breakpoint]

            self.event_ordering = list(
                {k: v for k, v in sorted(event_order_dict.items(), key=lambda item: item[1])}.keys()
            )
            return self.event_ordering, dict(sorted(event_order_dict.items(), key=lambda item: item[1]))

        anamoly_alg = rpt.Dynp(model = self.anamoly_model)
        for col in self.feature_names:
            valid = self.sorted_data[col].dropna()
            anamoly_alg.fit(valid.values)
            breakpoint = anamoly_alg.predict(n_bkps = 1)[0]
            breakpoint = min(max(int(breakpoint), 0), len(valid) - 1)
            event_order_dict[col] = valid.index[breakpoint]
        self.event_ordering = list({k: v for k, v in sorted(event_order_dict.items(), key=lambda item: item[1])}.keys())
        
        return self.event_ordering, dict(sorted(event_order_dict.items(), key=lambda item: item[1]))