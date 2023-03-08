import math
import csv
import numpy as np
import matplotlib.pyplot as plt

# Runtime: O(mn). To construct the dp table, iterate through all the points
# in trajectory p, and for each iteration, iterate through all the points
# in trajectory q, calculating dtw score and updating size and pointers.
def dtw(seriesA, seriesB):
    len_a = len(seriesA)
    len_b = len(seriesB)

    # Initialize edges of the dp table 
    dp = [[[float('inf'), 0, float('inf')]  for j in range(len_b + 1)] for i in range(len_a + 1)]
    dp[0][0] = [0, 0, float('inf')]

    # Construct dp table
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate distance between current two points
            dist = find_dist_points(seriesA[i - 1], seriesB[j - 1]) ** 2
            
            # Calculate scores
            size1 = dp[i - 1][j - 1][1] + 1
            score1 = (dist + dp[i - 1][j - 1][1] * dp[i - 1][j - 1][0]) / size1
            if math.isnan(score1):
                score1 = float('inf')
                
            size2 = dp[i - 1][j][1] + 1
            score2 = (dist + dp[i - 1][j][1] * dp[i - 1][j][0]) / size2
            if math.isnan(score2):
                score2 = float('inf')
                
            size3 = dp[i][j - 1][1] + 1
            score3 = (dist + dp[i][j - 1][1] * dp[i][j - 1][0]) / size3
            if math.isnan(score3):
                score3 = float('inf')
            
            # Update size and pointer
            new_size = 0
            pointer = float('inf')
            if score2 <= score3 and score2 <= score1:
                new_size = size2
                pointer = 1
            elif score3 <= score2 and score3 <= score1:
                new_size = size3
                pointer = -1
            else:
                new_size = size1 
                pointer = 0
                
            dp[i][j] = [min(score1, score2, score3), new_size, pointer]

    # Find traceback assignemnt E_avg
    assign_e = [(seriesA[len_a - 1], seriesB[len_b - 1])]
    i = len_a
    j = len_b
    while i != 0 and j != 0:
        if dp[i][j][2] == 0: 
            assign_e.append((seriesA[i-2], seriesB[j-2]))
            i -= 1
            j -=1
        elif dp[i][j][2] == 1:
            assign_e.append((seriesA[i-2], seriesB[j-1]))
            i -= 1
        elif dp[i][j][2] == -1:
            assign_e.append((seriesA[i-1], seriesB[j-2]))
            j -= 1
        else: 
            break

    return (dp[len_a][len_b][0], assign_e[::-1])

# Runtime: O(mn). To construct the dp table, iterate through all the points
# in trajectory p, and for each iteration, iterate through all the points
# in trajectory q, calculating frechet score and updating pointers.
def frechet(seriesA, seriesB):
    len_a = len(seriesA)
    len_b = len(seriesB)

    # Initialize edges of the dp table 
    dp = [[[float('inf'), float('inf')] for j in range(len_b + 1)] for i in range(len_a + 1)]
    dp[0][0] = [0, float('inf')]

    # Construct dp table 
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Calculate distance between current two points
            dist = find_dist_points(seriesA[i-1], seriesB[j-1])
            dp_min = min(dp[i-1][j-1][0], dp[i-1][j][0], dp[i][j-1][0])

            # Update pointer 
            pointer = float('inf')
            if dp_min == dp[i - 1][j - 1][0]:     
                pointer = 0
            elif dp_min == dp[i - 1][j][0]:       
                pointer = 1
            else:       
                pointer = -1
            dp[i][j] = [max(dist, dp_min), pointer]

    # Find traceback assignment E_max
    assign_e = [(seriesA[len_a - 1], seriesB[len_b - 1])]
    i = len_a
    j = len_b
    while i != 0 and j != 0:
        if dp[i][j][1] == 0: 
            assign_e.append((seriesA[i-2], seriesB[j-2]))
            i -= 1
            j -=1
        elif dp[i][j][1] == 1:
            assign_e.append((seriesA[i-2], seriesB[j-1]))
            i -= 1
        elif dp[i][j][1] == -1:
            assign_e.append((seriesA[i-1], seriesB[j-2]))
            j -= 1
        else: 
            break
    
    return (dp[-1][-1][0], assign_e[::-1])

# Calculate euclidean distance between two points
def find_dist_points(p, q):
    return math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)

def main():
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
        print(frechet_result[0])

           
        dtw_result = dtw(trajectory_1, trajectory_2)
        dtw_distance = [find_dist_points(p,q) for p,q in dtw_result[1]]
        
        plt.figure(idx + 4)
        counts, bins = np.histogram(dtw_distance)
        plt.hist(bins[:-1], bins, weights=counts)
        print(dtw_result[0])
    
    file1, file2 = '115-20080520225850', '115-20080615225707'
    plt.show()

if __name__ == "__main__":
    main()
