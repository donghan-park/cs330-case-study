import csv
import matplotlib.pyplot as plt


def ts_greedy(trajectory, max_error):
    # Current estimate is leftmost and rightmost points
    estimate = [trajectory[0], trajectory[-1]]

    # For each point in the trajectory, find the one with the highest error
    error, error_index = 0, 0
    for i, point in enumerate(trajectory):
        current_error = compute(point, estimate)
        if current_error > error:
            error = current_error
            error_index = i
    
    # If error is less than max error, estimate is good
    if error <= max_error:
        return estimate
    
    # Otherwise, split the trajectory into two estimates with the error point being the middle point
    else:
        return ts_greedy(trajectory[0:error_index + 1], max_error) + ts_greedy(trajectory[error_index:], max_error)[1:]


def compute(q, e):
    # http://paulbourke.net/geometry/pointlineplane/ #
    # Find the x and y distances of the two points on the line
    x_distance = e[1][0] - e[0][0]  # x2 - x1
    y_distance = e[1][1] - e[0][1]  # y2 - y1

    # Calculate norm of the line
    norm = x_distance*x_distance + y_distance*y_distance

    # Testing if point does not lay outside the line
    u = ((q[0] - e[0][0]) * x_distance +
         (q[1] - e[0][1]) * y_distance) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # Calculate minimum distance of point from line
    x = e[0][0] + u * x_distance
    y = e[0][1] + u * y_distance
    dx = x - q[0]
    dy = y - q[1]
    dist = (dx*dx + dy*dy)**.5

    return dist


if __name__ == "__main__":
    # Read in first trajectory
    trajectory = []
    with open('geolife-cars.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[1] == '128-20080503104400':
                trajectory.append((float(row[2]), (float(row[3]))))

    # Convert to separate x and y arrays
    x_trajectory = [point[0] for point in trajectory]
    y_trajectory = [point[1] for point in trajectory]

    # Estimate path and plot
    for idx, error in enumerate([0.03, 0.1, 0.3]):
        estimate = ts_greedy(trajectory, error)
        x_estimate = [point[0] for point in estimate]
        y_estimate = [point[1] for point in estimate]
        plt.figure(idx + 1)
        plt.plot(x_trajectory, y_trajectory, color='black', marker='.', markersize=8)
        plt.plot(x_estimate, y_estimate, color='red', linestyle='dashed', marker='.', markersize=8)

    # Read in next four trajectories
    for file in ['128-20080503104400', '010-20081016113953', '115-20080520225850', '115-20080615225707']:
        trajectory = []
        with open('geolife-cars.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[1] == file:
                    trajectory.append((float(row[2]), (float(row[3]))))

        # For each trajectory, report its compression results
        estimate = ts_greedy(trajectory, 0.03)
        print('The compression ratio for trajectory {} is {}'.format(
        file, len(trajectory) / len(estimate)))
    
    plt.show()
