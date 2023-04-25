import numpy as np
import csv
from task2 import ts_greedy
from task3 import dtw
from task4 import approach_2
# from task4 import plot_original_trajectories
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
        for i, t in enumerate(T):
            curr_dist = min([dtw(t, c)[0] for c in centers])
            if curr_dist > max_dist:
                max_dist = curr_dist
                max_dist_idx = i
        # choose trajectory with farthest dist from nearest center as new center
        centers.append(T[max_dist_idx])

    return centers


def reassign_trajectory_partition(T, partitions):
    for t in T:
        min_dist = float('inf')
        # find closest partition center to t
        for i, partition in enumerate(partitions):
            dist = dtw(t, partition['center'])[0]
            if dist < min_dist:
                min_dist = dist
                closest_partition_idx = i
        partitions[closest_partition_idx]['new_trajectories'].append(t)


def chungus_lloyds(T, k, tmax):
    centers = seed_centers(T, k)
    partitions = []
    for c in centers:
        partitions.append({
            'center': c,
            'trajectories': [],
            'new_trajectories': []
        })
    reassign_trajectory_partition(T, partitions)
        
    for _ in range(tmax):
        # recalculate centers & update trajectories for next iteration
        for partition in partitions:
            if len(partition['new_trajectories']):
                partition['center'] = approach_2(partition['new_trajectories'])
            partition['trajectories'] = partition['new_trajectories']
            partition['new_trajectories'] = []

        reassign_trajectory_partition(T, partitions)

        trajectories_changed = True
        for partition in partitions:
            if partition['trajectories'] == partition['new_trajectories']:
                trajectories_changed = False
        if not trajectories_changed:
            break
    
    return partitions


def plot_trajectories(T, color_value):
    # This function takes a list of trajectories as input and plots them
    # Each trajectory is represented as a list of points
    for trajectory in T:
        # Extract the X and Y coordinates for each point in the trajectory
        x_trajectory = [point[0] for point in trajectory]
        y_trajectory = [point[1] for point in trajectory]
        # Set the title, x-axis label, and y-axis label for the plot
        plt.title('Trajectories')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        # Plot the trajectory using a black line with small dot markers
        plt.plot(x_trajectory, y_trajectory, color=color_value, label='Original Trajectory', marker='.', markersize=2)


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

    k = 4
    colors = ['red', 'green', 'blue', 'yellow']
    tmax = 15
    plot_trajectories(T, 'black')

    partitions = chungus_lloyds(T, k, tmax)
    centers = []
    cluster_trajectories = []
    for i, partition in enumerate(partitions):
        centers.append(partition['center'])
        plot_trajectories(partition['new_trajectories'], colors[i])
        print('partition #{}: {}'.format(i, len(partition['new_trajectories'])))
    plot_trajectories(centers, 'magenta')
    
    # for k in k_list: 
    #     print(lloyds_algorithm(T, k, 10, "kcenters"))
    # tmax = 2
    # # random_costs = calc_cost(T, k_list, tmax, "random")
    # kcenters_costs = calc_cost(T, k_list, tmax, "kcenters")

    # plt.figure(1)
    # # plt.plot(k_list, random_costs, color="red", label="random", linestyle='dashed', marker='.', markersize=2)
    # plt.plot(k_list, kcenters_costs, color="black", label="kcenters", linestyle='dashed', marker='.', markersize=2)
    
    plt.show()