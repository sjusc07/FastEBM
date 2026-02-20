import numpy as np
import sklearn
import pandas as pd
import ruptures as rpt
from collections import deque
import sys

class FastEBM:
    def __init__(self, data, ordering_method = 'mean_anchor', zFastEBM = True, n_neighbors = 5, anamoly_model = 'l2', return_dps = False, seed = None):
        self._check_input(input_data=data)
        self.data = data.iloc[:, :-1]
        self.stage = data.iloc[:, -1]                       # stage/reference group data
        self.feature_names = self.data.columns              # getting the feature names
        self.ordering_method = ordering_method
        self.n_neighbors = n_neighbors
        self.zFastEBM = zFastEBM
        self.anamoly_model = anamoly_model
        self.sorted_data = None                             # sorted subjects
        self.return_dps = return_dps
        self.seed = seed

    def _check_input(self, input_data):      # probably need a function to check the input data
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


# TODO: 
# DONE: graph connectivity
# DONE: Condition number
# DONE: Median anchor
# Support for NAs
# Add curves in the tutorial notebook
# Return abnormality indices in the main function for curve plotting
# DONE: Correlated features -- this is in Beta mode. Will need to be better in the future
## should the check for correlated features be automated or should the user be asked

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
        # print(pariwise_distance)
        nearest_n_dists, _ = sklearn.neighbors.NearestNeighbors(n_neighbors = self.n_neighbors, p = 2, metric = 'nan_euclidean')\
            .fit(self.data)\
                .kneighbors(n_neighbors = self.n_neighbors)         # calculating the euclidean distance of each subject to its first n_neighbor (default = 5) neighbors
        epsilon = nearest_n_dists[:, self.n_neighbors-1]            # getting the distance to the n_neighbor neighbor (e.g. 5th nearest neighbor)
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
        E_x = np.diag(np.sum(W_xy, 1)**(-0.5))
        T = np.matmul(np.matmul(E_x, W_xy), E_x)                    # T is a symmetric stochastic matrix corresponding to a Markov chain
        if not self._is_connected(T):
            print(f"WARNING: The transition matrix is not connected (connected: {self._is_connected(T)}.")
            print("Running FastEBM with one more nearest neighbor.")
            print("If you keep seeing this warning, consider stopping the program and checking your input.")
            return 

        # print(T)
        print(f"Condition number of transition matrix is: {np.linalg.cond(T)}")
        print("If this number is too high for your application, consider increasing the number of nearest neighbors.")
        print("--------------------------------------")

        # calculating the stationary distribution of the Markov chain
        stationary_distribution = np.array([1/number_of_subjects for _ in range(number_of_subjects)])
        stationary_matrix = np.tile(stationary_distribution, (len(stationary_distribution), 1))     # creating a matrix with all rows being the stationary distribution

        # calculating the fundamental matrix of the Markov chain
        N = np.linalg.inv(np.eye(number_of_subjects) - T + stationary_matrix) - np.eye(number_of_subjects)
        N = N.clip(min = 0)                                         # cleaning up

        # ordering subjects across pseudotime axis
        if self.ordering_method == 'mean_template':
            N_df = pd.DataFrame(N)      # can be done with numpy probably -- to be optimized
            N_df['stage'] = self.stage
            centroid = (N_df[N_df.stage == 1].mean())[:-1].values   # assumption is that the reference group is labeled 1 and everyone else is 2
            
            # calculating the disease progression score (DPS)
            DPS = np.zeros((1, number_of_subjects))[0]
            for x in range(number_of_subjects):
                centroid_dist = (centroid- N[x,:])**2
                DPS[x] = np.sqrt(np.sum(centroid_dist))

        elif self.ordering_method == 'median_template':
            N_df = pd.DataFrame(N)
            N_df['stage'] = self.stage
            centroid = (N_df[N_df.stage == 1].median())[:-1].values
            DPS = np.zeros((1, number_of_subjects))[0]
            for x in range(number_of_subjects):
                D2 = (centroid - N[x,:])**2
                DPS[x] = np.sqrt(np.sum(D2))
        
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
        anamoly_alg = rpt.Dynp(model = self.anamoly_model)
        for col in self.feature_names:
            anamoly_alg.fit(self.sorted_data[col].dropna().values)
            event_order_dict[col] = anamoly_alg.predict(n_bkps = 1)[0]
        self.event_ordering = list({k: v for k, v in sorted(event_order_dict.items(), key=lambda item: item[1])}.keys())
        
        return self.event_ordering, dict(sorted(event_order_dict.items(), key=lambda item: item[1]))

    def get_biomarker_curve(self):
        pass

    def insert_features(self, df_ordered, df_data, features):
        pass