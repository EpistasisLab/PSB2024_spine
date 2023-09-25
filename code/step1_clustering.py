import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets
import openensembles as oe
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore')
input_file = 'path_to_file.csv'
norm_file = 'path_to_norm_file.csv'
n_df = pd.read_csv(input_file)  # rawdata_all4-4.csv')
cwd = '/path/'
consensus_result_path = os.path.join(cwd, "consensus_")
clustering_result_path = os.path.join(cwd, "clustering_")

##
if not os.path.exists(consensus_result_path):
    os.mkdir(consensus_result_path)
if not os.path.exists(clustering_result_path):
    os.mkdir(clustering_result_path)


df = pd.DataFrame(MinMaxScaler().fit_transform(n_df))
df.to_csv(norm_file)


###############


np.random.seed(0)
d = oe.data(df, [i for i in range(1, len(df.columns) + 1)])
c = oe.cluster(d)  # instantiate an object so we can get all available algorithms
a = c.algorithms_available()
paramsC = c.clustering_algorithm_parameters()  # here we will rely on walking through

algorithmsToRemove = ['DBSCAN', 'HDBSCAN', 'MeanShift', 'AffinityPropagation']

for algToRemove in algorithmsToRemove:
    del a[algToRemove]

takesLinkages = paramsC['linkage']

takesDistances = paramsC['distance']

takesK = paramsC['K']

K = list(range(2, 11, 1))
linkages = ['ward']
distances = ['euclidean', 'l1']



c = oe.cluster(d)
for data_source in d.D.keys():  # if there were transformations in d.D
    for algorithm in list(
            a.keys()):  # linkage is only for agglomerative, which also accepts K and distances, so handle that here
        # check if the algorithm takes K
        if algorithm in takesK:
            # Loop through all the K's we want.That is 2,3,4,5,6,7,8,9,10
            for k in K:
                # Check if the algorithms takes distance
                if algorithm in takesDistances:
                    # Check if the algorithm takes linkages
                    if algorithm in takesLinkages:
                        # Go through all the linkages
                        for linkage in linkages:
                            if linkage == 'ward':

                                # out_name = '_'.join([data_source, algorithm, linkage, str(k)])
                                out_name = '_'.join([algorithm, linkage, str(k)])
                                c.cluster(data_source, algorithm, out_name, K=k, random_state=0, Require_Unique=True,
                                            linkage=linkage)


                            # check if linkage is not ward
                            else:
                                # go through the distances [euclidean,L1,l2]
                                for dist in distances:
                                    out_name = '_'.join([algorithm, dist, linkage, str(k)])

                                    # Create the cluster with the data source,algorithm,output name,number of K's
                                    c.cluster(data_source, algorithm, out_name, K=k, random_state=0, Require_Unique=True,
                                                linkage=linkage, distance=dist)


                    # Algorithm does not take linkages
                    else:
                        # Go through all the distances
                        for dist in distances:
                            out_name = '_'.join([algorithm, dist, str(k)])
                            c.cluster(data_source, algorithm, out_name, K=k, random_state=0, Require_Unique=True, distance=dist)
                # Algorithm does not take distance
                else:
                    out_name = '_'.join([algorithm, str(k)])
                    c.cluster(data_source, algorithm, out_name, K=k, random_state=0, Require_Unique=True)

        # Algorithm does not take K
        else:  # does not take K
            # Check if algorithm  takes distance
            if algorithm in takesDistances:
                for dist in distances:
                    out_name = '_'.join([algorithm, dist])
                    c.cluster(data_source, algorithm, out_name, random_state=0, Require_Unique=True, distance=dist)
            # Algorithm that does not take distance
            else:
                out_name = '_'.join([algorithm])
                c.cluster(data_source, algorithm, out_name, random_state=0, Require_Unique=True)
result_df = pd.DataFrame()
# Plot and save each of the clustering solutions (Each clustering solution brings us labels) using the original dataset
cluster_solution_names = c.labels.keys()
for name in cluster_solution_names:
    mini_df = pd.DataFrame()
    mini_df[""] = c.labels[name]
    mini_df[""] = mini_df[""] + 1
    mini_df.to_csv(clustering_result_path + "/" + name + ".csv", index=False, header=False)
    result_df[name] = c.labels[name]
    result_df[name] = result_df[name] + 1

import multiprocessing

def clustering_and_consensus(c,d,i):
    total_mixture = pd.DataFrame()
    mixture_model = c.mixture_model(i, iterations=10)
    mixture_model_labels = mixture_model.labels["mixture_model"]
    # For mixture model solution (Mixture model already clusters starting from 1.So we dont need to add)
    mini_df = pd.DataFrame()
    mini_df["mixture_model_" + str(i)] = mixture_model_labels
    mini_df.to_csv(consensus_result_path + "/mixture_model_{}.csv".format(str(i)), index=False, header=False)
    print("Mixture Model " + str(i) + "done")
    result_df["mixture_model_" + str(i)] = mixture_model_labels
    total_mixture['K' + str(i)] = mixture_model_labels

    
    
if __name__ == '__main__':
    pool = multiprocessing.Pool(9)
    results = pool.starmap(clustering_and_consensus, [(c, d, i) for i in range(2, 11)])
    pool.close()
