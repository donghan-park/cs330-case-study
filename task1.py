import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import time

def init_P():
    with open('geolife-cars.csv') as file:
        next(file)
        P = [(float(x), float(y)) for date, id, x, y in csv.reader(file)]
    return P

def get_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class Node:
    def __init__(self, point, axis, left, right):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, P, r):
        self.tree = self.build_tree(P)
        self.radius = r

    def build_tree(self, points, depth=0):
        if not points:
            return None
        
        axis = depth % 2
        sorted_points = sorted(points, key = lambda p: p[axis])

        mid_index = len(sorted_points) // 2
        point = sorted_points[mid_index]
        left = self.build_tree(sorted_points[:mid_index], depth + 1)
        right = self.build_tree(sorted_points[mid_index + 1:], depth + 1)

        return Node(point, axis, left, right)

    def query_density(self, p, tree, r, count=0):
        if tree is None:
            return count
        dist = get_dist(p[0], p[1], tree.point[0], tree.point[1])
        if dist <= r:
            count += 1
        axis = tree.axis
        if tree.left and p[axis] - r <= tree.point[axis]:
            count = self.query_density(p, tree.left, r, count)
        if tree.right and p[axis] + r >= tree.point[axis]:
            count = self.query_density(p, tree.right, r, count)
        return count

    def density(self, p):
        return self.query_density(p, self.tree, self.radius)
    
    def hubs(self, P, k, r):
        hubs = []
        for p in P:
            if any(get_dist(hub[0], hub[1], p[0], p[1]) < r for hub in hubs):
                continue
            density = self.density(p)
            if density >= k:
                hubs.append(p)
            if len(hubs) == k:
                return hubs
        return hubs

# for testing
if __name__ == "__main__":
    start_time = time.time()
    P = init_P()
    k = 40
    r = 2
    tree = KDTree(P, r)
    hubs = tree.hubs(P, k, r)
    print("runtime: {:.4f} sec".format(time.time() - start_time))

    # P_np = np.array([P])
    # hubs_np = np.array([hubs])
    # p_x, p_y = P_np.T
    # hubs_x, hubs_y = hubs_np.T

    # fig, ax = plt.subplots()
    # plt.scatter(p_x, p_y, s=1, color = 'blue')
    # plt.scatter(hubs_x, hubs_y, s=20, color='red')
    # for x in hubs:
    #     circle = plt.Circle((x[0], x[1]), r, color='black', fill=False, linestyle='dashed')
    #     ax.add_artist(circle)

    # ax.set_aspect('equal', adjustable='box')
    # plt.show()