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
"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣧⠀⠀⠀⢰⡿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡟⡆⠀⠀⣿⡇⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⣿⠀⢰⣿⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡄⢸⠀⢸⣿⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡇⢸⡄⠸⣿⡇⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⢸⡅⠀⣿⢠⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣥⣾⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡿⡿⣿⣿⡿⡅⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠉⠀⠉⡙⢔⠛⣟⢋⠦⢵⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣄⠀⠀⠁⣿⣯⡥⠃⠀⢳⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡇⠀⠀⠀⠐⠠⠊⢀⠀⢸⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⡿⠀⠀⠀⠀⠀⠈⠁⠀⠀⠘⣿⣄⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣠⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣷⡀⠀⠀⠀
⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣧⠀⠀
⠀⠀⠀⡜⣭⠤⢍⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢛⢭⣗⠀
⠀⠀⠀⠁⠈⠀⠀⣀⠝⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠠⠀⠀⠰⡅
⠀⠀⠀⢀⠀⠀⡀⠡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠔⠠⡕⠀
⠀⠀⠀⠀⣿⣷⣶⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀
⠀⠀⠀⠀⠘⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠉⢆⠀⠀⠀⠀
⠀⢀⠤⠀⠀⢤⣤⣽⣿⣿⣦⣀⢀⡠⢤⡤⠄⠀⠒⠀⠁⠀⠀⠀⢘⠔⠀⠀⠀⠀
⠀⠀⠀⡐⠈⠁⠈⠛⣛⠿⠟⠑⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠉⠑⠒⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
BIG CHUNGUS
"""

def randomized_seed_centers(T, k):
    centers = []
    t_indices = random.sample(range(len(T)), k)
    for idx in t_indices:
        centers.append(T[idx])
        
    return centers

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
    cost = 0
    for t in T:
        min_dist = float('inf')
        # find closest partition center to t
        for i, partition in enumerate(partitions):
            dist = dtw(t, partition['center'])[0]
            if dist < min_dist:
                min_dist = dist
                closest_partition_idx = i
        partitions[closest_partition_idx]['new_trajectories'].append(t)
        cost += min_dist
    
    return cost


def lloyds(T, k, tmax, seeding_mode):
    centers = seed_centers(T, k) if seeding_mode else randomized_seed_centers(T, k)
    
    partitions = []
    for c in centers:
        partitions.append({
            'center': c,
            'trajectories': [],
            'new_trajectories': []
        })
    reassign_trajectory_partition(T, partitions)
    
    cost = [0 for _ in range(tmax)]
    for i in range(tmax):
        # recalculate centers & update trajectories for next iteration
        for partition in partitions:
            if len(partition['new_trajectories']):
                partition['center'] = approach_2(partition['new_trajectories'])
            partition['trajectories'] = partition['new_trajectories']
            partition['new_trajectories'] = []

        cost[i] = reassign_trajectory_partition(T, partitions)

        trajectories_changed = True
        for partition in partitions:
            if partition['trajectories'] == partition['new_trajectories']:
                trajectories_changed = False
        if not trajectories_changed:
            print("broke after {} iterations".format(i + 1))
            break
    
    return partitions, cost

def calc_cost(partitions):       
    cost = 0
    for partition in partitions: 
        if partition['new_trajectories']:   
            center = partition['center']
            cost += np.sum([dtw(t, center)[0] for t in partition['new_trajectories']])
    return cost

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
    # Create a dictionary to hold the trajectories for each ID
    trajectories = {}

    # Read in the trajectory data from a CSV file and add each point to the appropriate trajectory
    with open('geolife-cars-upd8.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            id = row[0]
            if id not in trajectories:
                trajectories[id] = []
            trajectories[id].append((float(row[1]), (float(row[2]))))

    # Convert the dictionary of trajectories to a list of trajectory lists
    T = []
    for trajectory in trajectories:
        T.append(trajectories[trajectory])

    # Simplify the trajectory using ts_greedy
    for i, trajectory in enumerate(T):
        T[i] = ts_greedy(T[i], 0.1)

    # initialize variables
    k_list = [4, 6, 8, 10, 12]
    tmax = 15
    colors = [
        'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'grey', 'black', 'turquoise', 'magenta'
    ]

    # TASK 5.3:
    random_costs = []
    kcenters_costs = []
    for k in k_list:
        r = 3
        random_clusters = [lloyds(T, k, tmax, 0) for _ in range(r)]
        rc = np.sum([calc_cost(random_cluster[0]) for random_cluster in random_clusters])/r
        k_center_clusters = [lloyds(T, k, tmax, 1) for _ in range(r)]
        kc = np.sum([calc_cost(k_center_cluster[0]) for k_center_cluster in k_center_clusters])/r
        random_costs.append(rc)
        kcenters_costs.append(kc)

    plt.figure(1)
    plt.plot(k_list, random_costs, color="red", label='Random Seeding')
    plt.plot(k_list, kcenters_costs, color="blue", label='Proposed Seeding')
    plt.title("Average Cost of Clustering vs. k-value in Lloyd's algorithm")
    plt.xlabel('k')
    plt.ylabel('Average Cost')
    plt.legend()

    # TASK 5.4:
    # optimal_k = 8
    # r = 3
    # random_clusters = [lloyds(T, optimal_k, tmax, 0) for _ in range(r)]
    # k_center_clusters = [lloyds(T, optimal_k, tmax, 1) for _ in range(r)]
    # random_cluster_costs = [cluster[1] for cluster in random_clusters]
    # k_center_cluster_costs = [cluster[1] for cluster in k_center_clusters]
    
    # max_len = max(len(l) for l in random_cluster_costs)
    # random_average = []
    # for i in range(max_len):
    #     total = sum(l[i] for l in random_cluster_costs if i < len(l))
    #     avg = total / len([l for l in random_cluster_costs if i < len(l)])
    #     random_average.append(avg)
    
    # max_len = max(len(l) for l in k_center_cluster_costs)
    # k_center_average = []
    # for i in range(max_len):
    #     total = sum(l[i] for l in k_center_cluster_costs if i < len(l))
    #     avg = total / len([l for l in k_center_cluster_costs if i < len(l)])
    #     k_center_average.append(avg)
    
    # while random_average and random_average[-1] == 0:
    #     random_average.pop()

    # while k_center_average and k_center_average[-1] == 0:
    #     k_center_average.pop()
    
    # plt.figure(2)
    # plt.scatter([i for i in range(1, len(random_average) + 1)], random_average, color='red', label='Random Seeding')
    # plt.scatter([i for i in range(1, len(k_center_average) + 1)], k_center_average, color='blue', label = 'Proposed Seeding')
    # plt.title("Average Cost of Clustering vs. Iterations for k = 8")
    # plt.xlabel('Iteration')
    # plt.ylabel('Average Cost')
    # plt.legend()
    

    # TASK 5.5:
    # optimal_k = 8
    # partitions = lloyds(T, optimal_k, tmax, 0)[0]
    # centers = []
    # plt.figure(3)
    # # Set the title, x-axis label, and y-axis label for the plot
    # plt.title('Center Trajectories of Clusters for k = 8')
    # plt.xlabel('X (km)')
    # plt.ylabel('Y (km)')
    # for i, partition in enumerate(partitions):
    #     x_trajectory = [point[0] for point in partition['center']]
    #     y_trajectory = [point[1] for point in partition['center']]
    #     plt.plot(x_trajectory, y_trajectory, color=colors[i], label='Center Trajectory {}'.format(i + 1), marker='.', markersize=2)
    # plt.legend()

    plt.show()
