import numpy as np
from task3 import dtw
from task4 import approach_2
import random

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
    for i in range(1, k):
        max_dist = 0
        # for each trajectory, calculate dist from nearest center
        for j, t in enumerate(T):
            curr_dist = min([dtw(t, c) for c in centers])
            if curr_dist > max_dist:
                max_dist = curr_dist
                max_dist_idx = j
        # choose trajectory with farthest dist from nearest center as new center
        centers.append(T[max_dist_idx])
    
    return centers

if __name__ == "__main__":
    k_list = [4,6,8,10,12]
    k = 4
    for i in range(1, k):
        print(i)

def lloyds_algorithm(T, k, tmax, sm):
    partitions = []
    # Randomly initialize partitions but change this to account for seeding method
    if sm == "random":    
        n = len(T)
        indices = np.arange(n)
        np.random.shuffle(indices)
        partition_size = n // k
        partitions = [T[indices[i*partition_size:(i+1)*partition_size]] for i in range(k)]
        partitions[-1] = np.concatenate([partitions[-1], T[indices[k*partition_size:]]]) 
    
    # k-centers seeding method initial partition
    if sm == "k-centers": 
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

    t = 0 
    while t <= tmax: 
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
        t+=1

    return partitions
