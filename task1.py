import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import time

# Function to store data from CSV file into array of tuples that represent points
def get_csv_data(file_name):
    with open(file_name) as file:
        next(file)
        data = [(float(i[2]), float(i[3])) for i in csv.reader(file)]
    return data

# Function to return euclidean distance between two points
def get_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Class definition of a node
class Node:
    def __init__(self, point, axis, left, right):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

# Class definition of KD-tree
class KDTree:
    # Class constructor parameters include the set of points P and preset radius
    # that is used by the density function
    def __init__(self, P, r):
        self.root = self.build_tree(P)
        self.radius = r

    # Build KD-tree by recursively subdividing the points of P, alternating by the
    # x- and y-axis. 
    # Runtime: O(nlogn) to build a KD-tree of n points in calligraphic P
    def build_tree(self, points, depth=0):
        if not points:
            return None
        
        # Alterate between the x- and y-axis
        axis = depth % 2
        # Sort subtree of points along the specified axis
        sorted_points = sorted(points, key = lambda p: p[axis])

        # Split points at the middle to create two subtrees
        mid_index = len(sorted_points) // 2
        point = sorted_points[mid_index]
        left = self.build_tree(sorted_points[:mid_index], depth + 1)
        right = self.build_tree(sorted_points[mid_index + 1:], depth + 1)

        # Return the root of the tree
        return Node(point, axis, left, right)
    
    # Calculate density of a point by traversing through a subset of all points in 
    # the KD-tree that is within a distance r from query point p.
    # Runtime: O(1) because density is computed by traversing through a neighborhood 
    # of points with radius r centered at point p. Instead of depending on the total
    # number of points in calligraphic P, it instead traverses through a preprocessed
    # KD-Tree, meaning its complexity only depends on the size of a small neighborhood.
    def density(self, p, root, count=0):
        if root is None:
            return count
        
        # Compute distance between query point p and the point of current node
        dist = get_dist(p[0], p[1], root.point[0], root.point[1])
        # Increment density counter if current node is within distance
        if dist <= self.radius:
            count += 1
        axis = root.axis
        # Visit left/right child nodes depending on whether or not the left/right
        # subtree is within the neighborhood
        if root.left and p[axis] - self.radius <= root.point[axis]:
            count = self.density(p, root.left, count)
        if root.right and p[axis] + self.radius >= root.point[axis]:
            count = self.density(p, root.right, count)

        # Return density, or the total number of points within distance r of point p
        return count
    
    # Compute points of k hubs by looping through all points in P
    # Runtime: O(n) because it linearly traverses through all points in calligraphic P.
    # For every point, the density() function is called, but this is an O(1) operation.
    def hubs(self, P, k, r):
        # Initialize empty list to store hubs
        hubs = []

        # Loop through all points
        for p in P:
            # Check if point is within a previously found hub
            if any(get_dist(hub[0], hub[1], p[0], p[1]) < r for hub in hubs):
                continue

            # If not, calculate density
            density = self.density(p, self.root)

            # Add to hubs if density is at least k (definition of valid hub)
            if density >= k:
                hubs.append(p)

            # Terminate loop if k hubs are found
            if len(hubs) == k:
                return hubs
            
        # Return list of hubs
        return hubs
    
# Plot scatterplot of all points in calligraphic P and all hubs
def plot_hubs(P, hubs, radius, title):
    p_x, p_y = np.array([P]).T
    hubs_x, hubs_y = np.array([hubs]).T

    fig, ax = plt.subplots()
    plt.scatter(p_x, p_y, s=0.1, color='blue', label='points of $\mathcal{P}$')
    plt.scatter(hubs_x, hubs_y, s=20, color='red', label='hubs')

    for hub in hubs:
        circle = plt.Circle((hub[0], hub[1]), radius, color='black', fill=False, linestyle='dashed')
        ax.add_artist(circle)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Main method to obtain experimental results
if __name__ == "__main__":
    P = get_csv_data('geolife-cars.csv')
    P_10 = get_csv_data('geolife-cars-ten-percent.csv')
    P_30 = get_csv_data('geolife-cars-thirty-percent.csv')
    P_60 = get_csv_data('geolife-cars-sixty-percent.csv')

    # Declare radius that density function uses
    density_r = 1

    # == Bullet 1: scatterplot ==
    k = 10
    r = 8
    start_time = time.time()
    tree = KDTree(P, density_r)
    hubs = tree.hubs(P, k, r)
    run_time = time.time() - start_time
    print('[k = {}, r = {}]'.format(k, r))
    print("Runtime: {:.0f} ms".format(run_time * 1000))
    plot_hubs(P, hubs, r, 'Plot of hubs and points of $\mathcal{P}$ (k=10, r=8)')

    # == Bullet 2: run hubs() with varying k ==
    ks = [5, 10, 20, 40]
    r = 2
    for k in ks:
        print("[k = {}, r = {}]".format(k, r))
        for i in range(3):
            start_time = time.time()
            tree = KDTree(P, density_r)
            hubs = tree.hubs(P, k, r)
            run_time = time.time() - start_time
            print("{}) Runtime: {:.0f} ms".format(i + 1, run_time * 1000))
    
    # == Bullet 3: run hubs() with subsamples ==
    k = 10
    r = 8
    subsets = [P_10, P_30, P_60, P]
    for subset in subsets:
        start_time = time.time()
        tree = KDTree(subset, density_r)
        hubs = tree.hubs(subset, k, r)
        run_time = time.time() - start_time
        print("Runtime: {:.0f} ms".format(run_time * 1000))