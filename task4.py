import csv
import matplotlib.pyplot as plt
from task3 import dtw

def pairwise_centering(trajectories):
    table = [[0 for j in range(len(trajectories))] for i in range(len(trajectories))]
    for i in range(len(trajectories)):
        for j in range(len(trajectories)):
            if i != j and not table[i][j]:
                distance = dtw(trajectories[i], trajectories[j])[0]
                table[i][j] = distance
                table[j][i] = distance
    
    min_sum = float('inf')
    min_trajectory = -1
    for trajectory in range(len(trajectories)):
        distance_sum = sum(table[trajectory])
        if distance_sum < min_sum:
            min_sum = distance_sum
            min_trajectory = trajectory
    
    return trajectories[min_trajectory]

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
    for trajectory in trajectories:
        x_trajectory = [point[0] for point in trajectories[trajectory]]
        y_trajectory = [point[1] for point in trajectories[trajectory]]
        plt.title('Trajectories')
        plt.xlabel('X in km')
        plt.ylabel('Y in km')
        plt.plot(x_trajectory, y_trajectory, color='black', marker='.', markersize=8)
    
    # pairwise_center = pairwise_centering(t)
    # x_trajectory = [point[0] for point in pairwise_center]
    # y_trajectory = [point[1] for point in pairwise_center]
    # plt.plot(x_trajectory, y_trajectory, color='red', linestyle='dashed', marker='.', markersize=8)

    plt.show()

if __name__ == '__main__':
    main()
