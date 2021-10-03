import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def compute_rmse(arr1: list, arr2: list) -> int:
    """
    Simple computation of Root Mean Squared Error
    Takes in two arrays: arr1, arr2
    
    Return rmse: integer
    """
    rmse = np.square(np.sum((arr1 - arr2)**2))
    return rmse



def simple_kmeans(input_data: np.array, cluster_count: int, max_interations: int=100, 
                  visualize: bool=True) -> (list, list):
    """
    A Simple computation of the K means based on cluster centroid RMSE
    Measure the RMSE for each iteration, once there is no imporvement,
    break the loop to get the values.
    
    
    Param input_data: it can be a nd-array; it's good if its a numpy array
        It can handle n-dimensions.
    Param cluster_count: it is the count of number clusters to fit in the data
    Param max_iterations: integer value; a hard stop to the loop when there is
        a little or no convergance in error rates
    Param visulaize: boolean value, if set to True will create a final cluster
        chart only if the input_data dimension is 2
        
    Returns: Tuple containing array of centroid_ids, 
        array of centroid coordinates for each iteration
        (LIST, LIST(array))
    
    """
    # first thing is to randomly initialize the cluster centroids:
    # lets create a dataframe for visualizations and better axis handling
    input_rows, input_columns = input_data.shape
    
    if input_rows <  cluster_count:
        raise ValueError(" Minimus sample size should "
                         "match input_data.shape[1]")
    
    dataset = pd.DataFrame(input_data)
    dataset_min = dataset.min().min()
    dataset_max = dataset.max().max()
    
    centroids = []

    for center in range(cluster_count):
        centroid = np.random.uniform(dataset_min, dataset_max, input_columns)
        centroids.append(centroid)
    # for better accessibilty
    centroids = np.array(centroids)
    
    master_errors = []
    continue_the_loop = True
    counter = 0
    centroid_coordinations_for_iteration = []
    while(continue_the_loop):
        all_centeriods_assigned = []
        all_centeroids_errors = []
        for input_point in range(input_rows):
            # compute error with each initialized centeroid
            errors = np.array([])
            for center in range(cluster_count):
                # now this is dataframe not np array so use iloc
                if centroids.shape[0] < cluster_count:
                    continue
                err = compute_rmse(centroids[center, :2], 
                                   dataset.iloc[input_point, :2])
                errors = np.append(errors, err)

            all_centeriods_assigned.append(np.argmin(errors))
            all_centeroids_errors.append(np.amin(errors))
            
        # use the all here
        # lets put the centroids in the dataframe to each point
        dataset['centroid_id'] = all_centeriods_assigned
        current_iteration_error = np.sum(all_centeroids_errors)
        master_errors.append(current_iteration_error)
        
        # after the errors, labels are assigned, we also need to update
        # centroid values based on the mean values corresponding to center ids
        
        centroids = dataset.groupby('centroid_id')\
            .agg('mean').reset_index(drop=True).to_numpy()
        
        centroid_coordinations_for_iteration.append(centroids)
        
        # Lets put in the closing logic
        # we want to quit once the there is no more improvement in the errors
        # or max iteration is reached (sklearn)
        
        if (len(master_errors) > 3):
            # breaking the two checks for better readabilty:
            #             if (counter <= max_interations):
            #                 print("Max iteration reached to:", max_interations)
            #                 continue_the_loop = False
            # check improvement up to 4 decimal places
            if (np.round(master_errors[counter - 1], 4) == \
                  np.round(master_errors[counter]), 4):
                print("No Improvement in Error Rate")
                continue_the_loop = False
            else:
                continue_the_loop = True
                
        counter += 1

    # final update is a must
    centroids = dataset.groupby('centroid_id')\
        .agg('mean').reset_index(drop = True).to_numpy()
    
    centroid_coordinations_for_iteration.append(centroids)
    
    # visualize data upto 2 dims
    if visualize and input_columns == 2:
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], 
                    marker = 'o',alpha = 0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                    marker = 'o', s=300)
    
    return dataset.centroid_id.values, centroid_coordinations_for_iteration
