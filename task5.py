import numpy as np
import csv
from task2 import ts_greedy
from task3 import dtw
from task4 import approach_2
import random
import matplotlib.pyplot as plt

"""
    high-level:
    * trajectories must be simplified using method from task 2 to lower runtime
    
    1.  seed centers:
        use k-centers alg by picking random initial trajectory
        update trajectories to closest center
        mark trajectory with farthest distance as next center
        repeat
    2.  partition trajectories into their closest respective centers
    3.  compute new centers for each partition
    4.  for each trajectory, use new centers to group them into new partitions
    5.  repeat until centers do not change after one iteration
"""


def seed_centers(T, k):
    # choose random initial center
    centers = [random.choice(T)]
    for _ in range(1, k):
        max_dist = 0
        # for each trajectory, calculate dist from nearest center
        for j, t in enumerate(T):
            curr_dist = min([dtw(t, c)[0] for c in centers])
            if curr_dist > max_dist:
                max_dist = curr_dist
                max_dist_idx = j
        # choose trajectory with farthest dist from nearest center as new center
        centers.append(T[max_dist_idx])

    return centers


def lloyds_algorithm(T, k, tmax, sm):
    partitions = []
    # Randomly initialize partitions but change this to account for seeding method
    if sm == "random":
        n = len(T)
        indices = np.arange(n)
        np.random.shuffle(indices)
        partition_size = n // k
        partitions = []
        used_indices = set()
        for i in range(k): 
            p = []
            for j in indices[i*partition_size : (i+1)*partition_size]: 
                used_indices.add(j)
                p.append(T[j])
            partitions.append(p)
        
        last_partition = []
        for i in range(n):
            if i not in used_indices:
                last_partition.append(T[i])

        partitions[-1] = np.concatenate([partitions[-1], last_partition])

    # k-centers seeding method initial partition
    if sm == "kcenters":
        sc = seed_centers(T, k)
        for t in T:
            min_dist = float('inf')
            min_index = -1
            for k0 in range(k):
                dist = dtw(tr, sc[k0])[0]
                if dist < min_dist:
                    min_dist = dist
                    min_index = k0
            partitions[min_index].append(t)

    for _ in range(tmax):
        # calculate centers
        centers = [approach_2(partitions[j]) for j in range(k)]
        new_partitions = [[] for _ in range(k)]

        # calculate dtw distances for each trajectory to each center, pick smallest and add to new partitions
        for tr in T:
            min_dist = float('inf')
            min_index = -1
            for k0 in range(k):
                dist = dtw(tr, centers[k0])[0]
                if dist < min_dist:
                    min_dist = dist
                    min_index = k0
            new_partitions[min_index].append(tr)

        # Check for convergence
        if partitions == new_partitions:
            break

        partitions = new_partitions

    return partitions

def calc_cost(T, k_list, tmax, sm):
    cost_list = []
    for k in k_list:
        clusters = lloyds_algorithm(T, k, tmax, sm)
        centers = [approach_2(clusters[j]) for j in range(k)]
        # calculate cost 
        costs = []
        for i in range(k): 
            costs_i = np.sum([dtw(t, centers[i])[0] for t in clusters[i]])
            costs.append(costs_i)
        cost = np.sum(costs)
        cost_list.append(cost)
    return cost_list

if __name__ == "__main__":
    k_list = [4, 6, 8, 10, 12]

    # Read in a list of trajectory IDs from a text file
    with open('trajectory-ids.txt', 'r') as file:
        trajectory_ids = [line.rstrip() for line in file]
    # Create a dictionary to hold the trajectories for each ID
    trajectories = {}
    for trajectory_id in trajectory_ids:
        trajectories[trajectory_id] = []
    # Read in the trajectory data from a CSV file and add each point to the appropriate trajectory
    with open('geolife-cars-upd8.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            id = row[0]
            if id in trajectories:
                trajectories[id].append((float(row[1]), (float(row[2]))))

    # Convert the dictionary of trajectories to a list of trajectory lists
    T = []
    for trajectory in trajectories:
        T.append(trajectories[trajectory])
    # Simplify the trajectory using ts_greedy
    for i, trajectory in enumerate(T):
        T[i] = ts_greedy(T[i], 0.1)
        
    tmax = 1
    random_costs = calc_cost(T, k_list, tmax, "random")
    kcenters_costs = calc_cost(T, k_list, tmax, "kcenters")

    plt.figure(1)
    plt.plot(k_list, random_costs, color="red", label="random", linestyle='dashed', marker='.', markersize=2)
    plt.plot(k_list, kcenters_costs, color="black", label="kcenters", linestyle='dashed', marker='.', markersize=2)
    
    plt.show()


