import math
import csv
import numpy as np
import matplotlib.pyplot as plt

def dtw(seriesA, seriesB):
    len_a = len(seriesA)
    len_b = len(seriesB)

    # make dp table
    dp = [[(float('inf'), 0) for j in range(len_b + 1)] for i in range(len_a + 1)]
    dp[0][0] = (0, 0)

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # print(seriesA[i-1], seriesB[j-1])
            dist = find_dist(seriesA[i - 1][0], seriesB[j - 1][0], seriesA[i - 1][1], seriesB[j - 1][1]) ** 2
            
            size1 = dp[i - 1][j - 1][1] + 1
            temp1 = (dist + dp[i - 1][j - 1][1] * dp[i - 1][j - 1][0]) / size1
            if math.isnan(temp1):
                temp1 = float('inf')
                
            size2 = dp[i - 1][j][1] + 1
            temp2 = (dist + dp[i - 1][j][1] * dp[i - 1][j][0]) / size2
            if math.isnan(temp2):
                temp2 = float('inf')
                
            size3 = dp[i][j - 1][1] + 1
            temp3 = (dist + dp[i][j - 1][1] * dp[i][j - 1][0]) / size3
            if math.isnan(temp3):
                temp3 = float('inf')
                
            # print(i, j, temp1, temp2, temp3, size1, size2, size3)
            new_size = 0
            if temp2 <= temp3 and temp2 <= temp1:
                new_size = size2
            elif temp3 <= temp2 and temp3 <= temp1:
                new_size = size3
            else:
                new_size = size1 
                
            dp[i][j] = (min(temp1, temp2, temp3), new_size)
            # print(dp[i][j])
            # dp[i][j] = min(dist, min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]))
    
    # for i in range(len_a, 0, -1): 
    #     for j in range(len_b, 0, -1): 
    #         dp[]

    # for x in dp: 
    #     print(x)

    return dp[len_a][len_b]

def frechet(seriesA, seriesB):
    len_a = len(seriesA)
    len_b = len(seriesB)

    # make dp table
    dp = [[float('inf') for j in range(len_b + 1)] for i in range(len_a + 1)]
    dp[0][0] = 0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dist = find_dist(seriesA[i-1][0], seriesB[j-1][0], seriesA[i-1][1], seriesB[j-1][1])
            dp[i][j] = max(dist, min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]))
        
    assign_e = [(seriesA[len_a - 1], seriesB[len_b - 1])]
    i = len_a - 1
    j = len_b - 1
    while i != 0 and j != 0:

        if min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) == dp[i-1][j-1]: #diagonal
            assign_e.append((seriesA[i-1], seriesB[j-1]))
            i -= 1
            j -=1
        elif min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) == dp[i-1][j]: #left
            assign_e.append((seriesA[i-1], seriesB[j]))
            i -= 1
        elif min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) == dp[i][j-1]: #right  
            assign_e.append((seriesA[i], seriesB[j-1]))
            j -= 1
        else:
            break

            # assign_e.append(min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]))
    
    for x in dp: 
        print(x)

    return (dp[len_a][len_b], assign_e[::-1])

def find_dist(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_dist_points(p, q):
    return math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)

def main():
    # seriesA = [(2, 0), (2, 1), (4, 1)]
    # seriesB = [(1, 2), (3, 3), (5, 5), (7, 5)]
    # print(frechet(seriesA, seriesB))
    for idx, (file1, file2) in enumerate([('128-20080503104400', '128-20080509135846'), ('010-20081016113953', '010-20080923124453'), ('115-20080520225850', '115-20080615225707')]):
        trajectory_1, trajectory_2 = [], []
        with open('geolife-cars.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[1] == file1:
                    trajectory_1.append((float(row[2]), (float(row[3]))))
                if row[1] == file2:
                    trajectory_2.append((float(row[2]), (float(row[3]))))
   
        frechet_result = frechet(trajectory_1, trajectory_2)
        frechet_distance = [find_dist_points(p,q) for p,q in frechet_result[1]]
        
        plt.figure(idx + 1)
        counts, bins = np.histogram(frechet_distance)
        plt.hist(bins[:-1], bins, weights=counts)
        print(frechet_distance[0])
        # dtw(trajectory_1, trajectory_2)
    plt.show()

if __name__ == "__main__":
    main()
