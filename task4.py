import csv
import matplotlib.pyplot as plt
import numpy as np
from task2 import ts_greedy
from task3 import dtw

def approach_1(trajectories):
    # Initialize a table to store the pairwise distances between trajectories
    table = [[0 for j in range(len(trajectories))] for i in range(len(trajectories))]
    
    # Compute the pairwise distances between trajectories using DTW and store them in the table
    for i in range(len(trajectories)):
        for j in range(len(trajectories)):
            # Only compute the distance if the trajectories are different and the distance has not been computed before
            if i != j and not table[i][j]:
                # Compute the DTW distance between the two trajectories and store it in the table
                distance = dtw(trajectories[i], trajectories[j])[0]
                table[i][j] = distance
                table[j][i] = distance
    
    # Find the trajectory with the minimum sum of distances to all other trajectories
    min_sum = float('inf')
    min_trajectory = -1
    for trajectory in range(len(trajectories)):
        distance_sum = sum(table[trajectory])
        if distance_sum < min_sum:
            min_sum = distance_sum
            min_trajectory = trajectory
    
    # Return the trajectory with the minimum sum of distances as the center trajectory
    return trajectories[min_trajectory]

def approach_2(trajectories):
    # Find the maximum length of all trajectories
    max_len = max(len(traj) for traj in trajectories)
    
    # Resample all trajectories to have the same number of points using linear interpolation
    resampled_trajectories = []
    for traj in trajectories:
        traj = np.array(traj)
        resampled_traj = np.zeros((max_len, traj.shape[1]))
        for dim in range(traj.shape[1]):
            resampled_traj[:, dim] = np.interp(np.linspace(0, max_len, num=max_len), np.arange(len(traj)), traj[:, dim])
        resampled_trajectories.append(resampled_traj)

    # Compute the mean trajectory
    mean_trajectory = np.mean(resampled_trajectories, axis=0)

    # Return the mean trajectory
    return mean_trajectory


def plot_original_trajectories(t):
    for trajectory in t:
        x_trajectory = [point[0] for point in trajectory]
        y_trajectory = [point[1] for point in trajectory]
        plt.title('Trajectories')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.plot(x_trajectory, y_trajectory, color='black', marker='.', markersize=2)
        

def main():
    with open('trajectory-ids.txt', 'r') as file:
        trajectory_ids = [line.rstrip() for line in file]
    trajectories = {}
    for trajectory_id in trajectory_ids:
        trajectories[trajectory_id] = []
    with open('geolife-cars-upd8.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                id = row[0]
                if id in trajectories:
                    trajectories[id].append((float(row[1]), (float(row[2]))))
    
    t = []
    for trajectory in trajectories:
        t.append(trajectories[trajectory])

    plt.figure(1)
    plot_original_trajectories(t)
    approach_2_center = approach_2(t)
    x_trajectory = [point[0] for point in approach_2_center]
    y_trajectory = [point[1] for point in approach_2_center]
    plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=2)

    plt.figure(2)
    plot_original_trajectories(t)
    approach_1_center = approach_1(t)
    x_trajectory = [point[0] for point in approach_1_center]
    y_trajectory = [point[1] for point in approach_1_center]
    plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=2)

    for idx, error in enumerate([0.03, 0.1, 0.3]):
        for i, trajectory in enumerate(trajectories):
            t[i] = ts_greedy(trajectories[trajectory], error)
        plt.figure(idx + 3)
        plot_original_trajectories(t)
        approach_1_center = approach_1(t)
        x_trajectory = [point[0] for point in approach_1_center]
        y_trajectory = [point[1] for point in approach_1_center]
        plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=2)
    
    plt.show()


if __name__ == '__main__':
    main()
