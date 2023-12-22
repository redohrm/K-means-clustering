#!/usr/bin/env python3

# PROGRAMMER: Ruth Dohrmann
# PROGRAM: kmeans.py
# 
# Description: This program performs K-means clustering on labeled classification data.
# The progam calculates a K-means clustering of a provided set of input training data,
# assigns classification labels to each cluster using a majority vote, and then reports
# the classification performance on a separate set of input validation data.

import sys
import numpy as np

# The program takes 3 command-line arguments: (integer: the number of clusters, 
# string: input training data filename, string: input validation data filename)
K = int(sys.argv[1])

def main():

    # get training data
    fileName = sys.argv[2]
    training_data = np.loadtxt(fileName)

    # perform k-means clustering
    clusters = k_means_clustering(training_data)
    # separate return value into center locations and classifications
    # print(K)
    if K > 1 and len(training_data.shape) > 1:
        classification = [item[-1] for item in clusters]
        location = np.delete(clusters, -1, axis=1)
    else:
        classification = clusters[0][-1]
        location = np.delete(clusters, -1)

    # get validation data
    newFileName = sys.argv[3]
    val_data = np.loadtxt(newFileName)
    num_correct = 0

    # check if the validation data is only one row long
    if len(val_data.shape) < 2:
        num_correct += test_kmeans(val_data, location, classification)
    else:
        # find the number of correctly classified objects
        for i in range(val_data.shape[0]):
            num_correct += test_kmeans(val_data[i], location, classification)
    print(num_correct)


# This function performs k-means clustering. It follows the general algorithm:
# 1. Given a set of N training examples each consisting of a vector of continuous attributes, 
# select K training examples to be a set of initial cluster means (centers)
# 2. Determine the distance between each training example and each of the K cluster means
# 3. Assign each training example to the closest cluster mean
# 4. Calculate the average of the training examples assigned to each cluster mean (creates a
# new mean)
# 5. Go back to step 2 until the cluster means do not change (i.e. all training examples are 
# assigned to the same cluster mean as on the previous iteration) 
def k_means_clustering(data):

    global K

    # check if there is only one row of data
    if len(data.shape) < 2:
        return [list(data)]

    # check if the number of rows in the training data is less than K
    if data.shape[0] < K:
        K = data.shape[0]

    saved_data = data
    data = np.delete(data, -1, axis=1)
    converged = False
    first = True
    indices_previous = []
    centers = data[:K, :]
    first = True

    # check if there is only one center (center numbers of less than 1 are treated as one)
    if K <= 1:
        # find the center
        center = np.average(data, axis=0)
        # find the classification with the maximum number of occurrences in the data
        arr = np.array(saved_data[:, -1]).astype(int)
        counts = np.bincount(arr)
        max_counts = np.argmax(counts)
        center = np.append(center, max_counts).tolist()
        K = 1
        return [center]

    # repeat this process until the center no longer change
    while not converged:

        # calculate the distances between each point and the centers
        dist = [(np.linalg.norm(data - centers[i], axis=1)) for i in range(centers.shape[0])]

        # Find the assignments (which cluster center will each point be closest to?)
        min_vals = np.minimum.reduce(np.array(dist))
        dist2 = np.reshape(np.array(dist), (len(dist), dist[1].shape[0]), order='F').T
        indices = [list(dist2[i]).index(min_vals[i]) for i in range(min_vals.shape[0])]
        
        # Find the new centers
        averages = np.zeros((K, data.shape[1]))
        for i in range(K):
            if data[np.array(indices[:]) == i, :].size == 0:
                averages[i] = centers[i]
            else:
                averages[i] += np.average(data[np.array(indices[:]) == i, :], axis=0).tolist()

        centers = averages

        # Check if the vector assignment stayed the same
        if not first and indices == indices_previous:
            converged = True
        else:
            indices_previous = indices
            first = False

    updated_centers = [None] * K
    
    # Assign class label to each cluster (represented by each center) by taking
    # a majority vote amongst it's assigned examples from the training set
    for i in range(K):
        arr = np.array(saved_data[np.array(indices[:]) == i, -1]).astype(int)
        if arr.size != 0:
            counts = np.bincount(arr)
            max_counts = np.argmax(counts)
            updated_centers[i] = np.append(centers[i], max_counts).tolist()

    null_count = updated_centers.count(None)
    updated_centers = [i for i in updated_centers if i is not None]
    K -= null_count
    
    return updated_centers


# This function tests whether or not the k-means clustering classified a validation
# example correctly.
def test_kmeans(data_single, locations, classification):
    saved_data = data_single
    data_single = np.delete(data_single, -1)
    # find which cluster the validation example is associated with
    dist = [(np.linalg.norm(data_single - locations[i])) for i in range(locations.shape[0])]
    my_index = dist.index(min(dist))

    # there are more than one classifications, compare the validation example 
    # classification with the correct cluster classification.
    if type(classification) == list:
        if saved_data[-1] == classification[my_index]:
            return 1
    # otherwise, compare the validation example classification with the only 
    # cluster classificiton
    else:
        if saved_data[-1] == classification:
            return 1
    return 0


if __name__ == "__main__":
    main()
