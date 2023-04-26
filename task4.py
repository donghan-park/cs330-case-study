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
    max_len = max(len(traj) for traj in trajectories)
    # Initialize a list to store the interpolated trajectories
    interpolated_trajectories = [[[0, 0] for _ in range(max_len)] for _ in range(len(trajectories))]

    # Compute the cumulative distances for each trajectory
    cumulative_distances_list = []
    for traj in trajectories:
        distances = []
        for j in range(1, len(traj)):
            dx = traj[j][0] - traj[j-1][0]
            dy = traj[j][1] - traj[j-1][1]
            distances.append((dx ** 2 + dy ** 2) ** 0.5)

        cumulative_distances = [0] + [sum(distances[:k+1]) for k in range(len(distances))]
        cumulative_distances_list.append(cumulative_distances)

    # Compute the interpolated trajectories
    for i, traj in enumerate(trajectories):
        cumulative_distances = cumulative_distances_list[i]
        # Compute the target distances for interpolation
        target_distances = [t * cumulative_distances[-1] / (max_len - 1) for t in range(max_len)]

        # Interpolate the trajectory using target distances
        interpolated_points = [[0, 0] for _ in range(max_len)]
        for dim in range(2):
            interpolated_dim = []
            k = 0
            for j in range(max_len):
                while k < len(cumulative_distances) - 2 and target_distances[j] > cumulative_distances[k + 1]:
                    k += 1
                if cumulative_distances[k+1] == cumulative_distances[k]:
                    t = 0
                else:
                    t = (target_distances[j] - cumulative_distances[k]) / (cumulative_distances[k + 1] - cumulative_distances[k])
                interpolated_point_dim = (1 - t) * traj[k][dim] + t * traj[k+1][dim]
                interpolated_dim.append(interpolated_point_dim)
            for j in range(max_len):
                interpolated_points[j][dim] = interpolated_dim[j]

        # Store interpolated trajectory in list
        for j in range(max_len):
            interpolated_trajectories[i][j][0] = interpolated_points[j][0]
            interpolated_trajectories[i][j][1] = interpolated_points[j][1]

    # Compute the mean trajectory
    mean_trajectory = [[0, 0] for _ in range(max_len)]
    for j in range(max_len):
        for i in range(len(trajectories)):
            mean_trajectory[j][0] += interpolated_trajectories[i][j][0]
            mean_trajectory[j][1] += interpolated_trajectories[i][j][1]
        mean_trajectory[j][0] /= len(trajectories)
        mean_trajectory[j][1] /= len(trajectories)

    # Return the mean trajectory
    return mean_trajectory


def plot_original_trajectories(t):
    # This function takes a list of trajectories as input and plots them
    # Each trajectory is represented as a list of points
    for trajectory in t:
        # Extract the X and Y coordinates for each point in the trajectory
        x_trajectory = [point[0] for point in trajectory]
        y_trajectory = [point[1] for point in trajectory]
        # Set the title, x-axis label, and y-axis label for the plot
        plt.title('Trajectories')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        # Plot the trajectory using a black line with small dot markers
        plt.plot(x_trajectory, y_trajectory, color='black', label='Original Trajectory', marker='.', markersize=2)


def plot_legend():
    # This function generates a legend for the plot based on the labels used in plt.plot() calls
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
        

def main():
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
    t = []
    for trajectory in trajectories:
        t.append(trajectories[trajectory])

    # Plot the original trajectories
    plt.figure(1)
    plot_original_trajectories(t)

    # Apply approach 2 to find the center trajectory and calculate the average distance between each trajectory and the center
    approach_2_center = approach_2(t)
    distance_sum = 0
    for trajectory in t:
        distance_sum += dtw(approach_2_center, trajectory)[0]
    print("Average distance for approach 2: {}".format(distance_sum / len(t)))
    # Plot the center trajectory in red with a dashed line
    x_trajectory = [point[0] for point in approach_2_center]
    y_trajectory = [point[1] for point in approach_2_center]
    plt.plot(x_trajectory, y_trajectory, color='red', label='Center Trajectory', linestyle='dashed', marker='.', markersize=2)
    # Add a legend to the plot
    plot_legend()

    # Repeat the process for approach 1, then plot the results
    plt.figure(2)
    plot_original_trajectories(t)
    approach_1_center = approach_1(t)
    distance_sum = 0
    for trajectory in t:
        distance_sum += dtw(approach_1_center, trajectory)[0]
    print("Average distance for approach 1: {}".format(distance_sum / len(t)))
    # Plot the center trajectory in red with a dashed line
    x_trajectory = [point[0] for point in approach_1_center]
    y_trajectory = [point[1] for point in approach_1_center]
    plt.plot(x_trajectory, y_trajectory, color='red', label='Center Trajectory', linestyle='dashed', marker='.', markersize=2)
    # Add a legend to the plot
    plot_legend()

    # Repeat the process for approach 1 after simplifying the trajectory, then plot the results 
    for idx, error in enumerate([0.03, 0.1, 0.3]):
        # Simplify the trajectory using ts_greedy
        for i, trajectory in enumerate(trajectories):
            t[i] = ts_greedy(trajectories[trajectory], error)
        plt.figure(idx + 3)
        plot_original_trajectories(t)
        approach_1_center = approach_1(t)
        distance_sum = 0
        for trajectory in trajectories:
            distance_sum += dtw(approach_1_center, trajectories[trajectory])[0]
        print("Average distance for approach 1 with error {}: {}".format(error, distance_sum / len(t)))
        x_trajectory = [point[0] for point in approach_1_center]
        y_trajectory = [point[1] for point in approach_1_center]
        plt.plot(x_trajectory, y_trajectory, color='red', label='Center Trajectory', linestyle='dashed', marker='.', markersize=2)
        plot_legend()
    
    plt.show()


if __name__ == '__main__':
    main()