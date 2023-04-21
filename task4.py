import csv
import matplotlib.pyplot as plt
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
    # Initialize an empty list to store the average trajectory
    avg_trajectory = []
    
    # Initialize a list to keep track of the number of updates for each point in the average trajectory
    point_counts = []
    
    # Iterate over all the trajectories
    for t in trajectories:
        # Iterate over all the points in the current trajectory
        for i, p in enumerate(t):
            # If the current index is greater than or equal to the length of the average trajectory,
            # append a new point to the average trajectory with the same coordinates as the current point
            if i >= len(avg_trajectory):
                avg_trajectory.append(p)
                point_counts.append(1)
            # Otherwise, update the i-th point in the average trajectory by computing the weighted average
            # of its current coordinates and the coordinates of the current point, and increasing the count
            # of updates for the point by 1
            else:
                count = point_counts[i]
                avg_trajectory[i] = [((count * avg_trajectory[i][0]) + p[0]) / (count + 1), ((count * avg_trajectory[i][1]) + p[1]) / (count + 1)]
                point_counts[i] += 1
    
    # Return the average trajectory as the center trajectory
    return avg_trajectory


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
    
    plt.figure(1)
    for trajectory in trajectories:
        x_trajectory = [point[0] for point in trajectories[trajectory]]
        y_trajectory = [point[1] for point in trajectories[trajectory]]
        plt.title('Trajectories')
        plt.xlabel('X in km')
        plt.ylabel('Y in km')
        plt.plot(x_trajectory, y_trajectory, color='black', marker='.', markersize=8)
    
    t = []
    for trajectory in trajectories:
        t.append(trajectories[trajectory])

    approach_2_center = approach_2(t)
    x_trajectory = [point[0] for point in approach_2_center]
    y_trajectory = [point[1] for point in approach_2_center]
    plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=8)

    plt.show()

    # approach_1_center = approach_1(t)
    # x_trajectory = [point[0] for point in approach_1_center]
    # y_trajectory = [point[1] for point in approach_1_center]
    # plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=8)

    for error in [0.03, 0.1, 0.3]:
        for i, trajectory in enumerate(trajectories):
            t[i] = ts_greedy(trajectories[trajectory], error)
        # approach_1_center = approach_1(t)
        # x_trajectory = [point[0] for point in approach_1_center]
        # y_trajectory = [point[1] for point in approach_1_center]
        # plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=8)

if __name__ == '__main__':
    main()
