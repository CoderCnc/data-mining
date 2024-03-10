# Part 2: Cluster Analysis

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df.drop(columns = ["Channel", "Region"], inplace = True)
	return df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	return round(df.describe().loc[['mean', 'std', 'min', 'max']])


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	standart_df = (df - df.mean()) / df.std()
	return standart_df

	scaler = StandardScaler()
	return pd.DataFrame(scaler.fit_transform(df), columns = df.columns)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	kmeans = KMeans(n_clusters = k, random_state = 0)
	y = kmeans.fit_predict(df)
	y = pd.Series(y)
	return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state=0)
	y = kmeans.fit_predict(df)
	y = pd.Series(y)
	return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	agglomerative = AgglomerativeClustering(n_clusters = k)
	y = agglomerative.fit_predict(df)
	y = pd.Series(y)
	return y


# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	score = silhouette_score(X, y)
	return score


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.

def cluster_evaluation(df):
	k_values = [3, 5, 10]
	algorithms = ["Kmeans", 'Agglomerative']
	data_types = ["Original", "Standardized"]
	cluster_evaluations_list = []


	for data_type in data_types:
		for algorithm in algorithms:
			for k in k_values:
				if data_type == "Standardized":
					X = standardize(df)
				else:
					X = df
				
				if algorithm == "Kmeans":
					for i in range(10):
						y = kmeans_plus(X, k)
						silhouette_avg = clustering_score(X, y)
						cluster_evaluations_list.append({'Algorithm': algorithm, 'data': data_type, 'k': k, 'run': i+1, 'Silhouette Score': silhouette_avg})

				else:
					y = agglomerative(X, k)
					silhouette_avg = clustering_score(X, y)
					cluster_evaluations_list.append({'Algorithm': algorithm, 'data': data_type, 'k': k, 'Silhouette Score': silhouette_avg})
	
	cluster_evaluations = pd.DataFrame(cluster_evaluations_list)
	return cluster_evaluations


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()


# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	pass

