import csv
from task2 import ts_greedy
from task3 import dtw
from task4 import approach_2
import random
import matplotlib.pyplot as plt

"""
Before we begin with the code, let us improve our well-being 
by appreciating the following animal friends :)

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

                                       /;    ;\
                                   __  \\____//
                                  /{_\_/   `'\____
                                  \___   (o)  (o  }
       _____________________________/          :--'  
   ,-,'`@@@@@@@@       @@@@@@         \_    `__\
  ;:(  @@@@@@@@@        @@@             \___(o'o)
  :: )  @@@@          @@@@@@        ,'@@(  `===='       
  :: : @@@@@:          @@@@         `@@@:
  :: \  @@@@@:       @@@@@@@)    (  '@@@'
  ;; /\      /`,    @@@@@@@@@\   :@@@@@)
  ::/  )    {_----------------:  :~`,~~;
 ;;'`; :   )                  :  / `; ;
;;;; : :   ;                  :  ;  ; :              
`'`' / :  :                   :  :  : :
    )_ \__;      ";"          :_ ;  \_\       `,','
    :__\  \    * `,'*         \  \  :  \   *  8`;'*  *
        `^'     \ :/           `^'  `-^-'   \v/ :  \/ 

Larry the Cow

                             .-----.
                            /7  .  (
                           /   .-.  \
                          /   /   \  \
                         / `  )   (   )
                        / `   )   ).  \
                      .'  _.   \_/  . |
     .--.           .' _.' )`.        |
    (    `---...._.'   `---.'_)    ..  \
     \            `----....___    `. \  |
      `.           _ ----- _   `._  )/  |
        `.       /"  \   /"  \`.  `._   |
          `.    ((O)` ) ((O)` ) `.   `._\
            `-- '`---'   `---' )  `.    `-.
               /                  ` \      `-.
             .'                      `.       `.
            /                     `  ` `.       `-.
     .--.   \ ===._____.======. `    `   `. .___.--`     .''''.
    ' .` `-. `.                )`. `   ` ` \          .' . '  8)
   (8  .  ` `-.`.               ( .  ` `  .`\      .'  '    ' /
    \  `. `    `-.               ) ` .   ` ` \  .'   ' .  '  /
     \ ` `.  ` . \`.    .--.     |  ` ) `   .``/   '  // .  /
      `.  ``. .   \ \   .-- `.  (  ` /_   ` . / ' .  '/   .'
        `. ` \  `  \ \  '-.   `-'  .'  `-.  `   .  .'/  .'
          \ `.`.  ` \ \    ) /`._.`       `.  ` .  .'  /
    LGB    |  `.`. . \ \  (.'               `.   .'  .'
        __/  .. \ \ ` ) \                     \.' .. \__
 .-._.-'     '"  ) .-'   `.                   (  '"     `-._.--.
(_________.-====' / .' /\_)`--..__________..-- `====-. _________)
                 (.'(.'

kathyl the Wizard Toad
"""


"""
Initialize centers using randomized seeding
"""
def randomized_seed_centers(T, k):
    centers = []
    t_indices = random.sample(range(len(T)), k)
    for idx in t_indices:
        centers.append(T[idx])
        
    return centers


"""
Initialize centers using k-centers algorithm
"""
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


"""
Reassign each trajectory t in T to its nearest partition
"""
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
    
    # return cost for computing cost
    return cost


"""
Runs Lloyd's algorithm to compute k clusters. The seeding mode can be 
specified by changing the fourth parameter, seeding_mode:

0: run with random seeding
1: run with proposed seeding (k-centers algorithm)

Runtime: O(tknm^2)
- Depending on the seeding method used:
    - Random seeding iterates through all trajectories, in O(n)
    - K-centers iterates through all trajectories k times, in O(nk)
- Initializing k partitions takes O(km)
- Iterating through all trajectories to find its nearest partition takes O(nkm)
- This is done for each iteration; hence, each iteration takes O(nkm) with a max of O(tknm)

Hence, total runtime is bounded by O(tknm^2) where:
t: number of iterations to termination
k: number of desired clusters
n: number of trajectories
m: maximum size of trajectory
"""
def lloyds(T, k, tmax, seeding_mode):
    centers = seed_centers(T, k) if seeding_mode else randomized_seed_centers(T, k)
    
    # create list to store lists of partitions
    partitions = []
    for c in centers:
        # each partition is represented as a dictionary with the following keys
        partitions.append({
            'center': c,
            'trajectories': [],
            'new_trajectories': []
        })
    reassign_trajectory_partition(T, partitions)
    
    # initialize list for storing iteration costs
    cost = [0 for _ in range(tmax)]

    # run iterations until no change in trajectory data is found or tmax is reached
    for i in range(tmax):
        # recalculate centers & update trajectories for next iteration
        for partition in partitions:
            if len(partition['new_trajectories']):
                partition['center'] = approach_2(partition['new_trajectories'])
            partition['trajectories'] = partition['new_trajectories']
            partition['new_trajectories'] = []

        # reassign all trajectories to nearest partitions
        cost[i] = reassign_trajectory_partition(T, partitions)

        # break loop if no change is found
        trajectories_changed = True
        for partition in partitions:
            if partition['trajectories'] == partition['new_trajectories']:
                trajectories_changed = False
        if not trajectories_changed:
            break
    
    return partitions, cost


"""
Calculate average cost from partitions created using Lloyd's
"""
def calc_cost(partitions):       
    cost = 0
    for partition in partitions: 
        if partition['new_trajectories']:   
            center = partition['center']
            cost += sum([dtw(t, center)[0] for t in partition['new_trajectories']])
    return cost


"""
Helper function to plot list of trajectories
"""
def plot_trajectories(T, color_value):
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

    # TASK 4.2.3:
    random_costs = []
    kcenters_costs = []
    for k in k_list:
        r = 3
        random_clusters = [lloyds(T, k, tmax, 0) for _ in range(r)]
        rc = sum([calc_cost(random_cluster[0]) for random_cluster in random_clusters])/r
        k_center_clusters = [lloyds(T, k, tmax, 1) for _ in range(r)]
        kc = sum([calc_cost(k_center_cluster[0]) for k_center_cluster in k_center_clusters])/r
        print("Average cost with random seeding for k = {} is {}".format(k, rc))
        random_costs.append(rc)
        print("Average cost with k-centers seeding for k = {} is {}".format(k, kc))
        kcenters_costs.append(kc)

    plt.figure(1)
    plt.plot(k_list, random_costs, color="red", label='Random Seeding')
    plt.plot(k_list, kcenters_costs, color="blue", label='Proposed Seeding')
    plt.title("Average Cost of Clustering vs. k-value in Lloyd's algorithm")
    plt.xlabel('k')
    plt.ylabel('Average Cost')
    plt.legend()

    # TASK 4.2.4:
    optimal_k = 8
    r = 3
    random_clusters = [lloyds(T, optimal_k, tmax, 0) for _ in range(r)]
    k_center_clusters = [lloyds(T, optimal_k, tmax, 1) for _ in range(r)]
    random_cluster_costs = [cluster[1] for cluster in random_clusters]
    k_center_cluster_costs = [cluster[1] for cluster in k_center_clusters]
    
    max_len = max(len(l) for l in random_cluster_costs)
    random_average = []
    for i in range(max_len):
        total = sum(l[i] for l in random_cluster_costs if i < len(l))
        avg = total / len([l for l in random_cluster_costs if i < len(l)])
        random_average.append(avg)
    
    max_len = max(len(l) for l in k_center_cluster_costs)
    k_center_average = []
    for i in range(max_len):
        total = sum(l[i] for l in k_center_cluster_costs if i < len(l))
        avg = total / len([l for l in k_center_cluster_costs if i < len(l)])
        k_center_average.append(avg)
    
    while random_average and random_average[-1] == 0:
        random_average.pop()

    while k_center_average and k_center_average[-1] == 0:
        k_center_average.pop()
    
    plt.figure(2)
    plt.scatter([i for i in range(1, len(random_average) + 1)], random_average, color='red', label='Random Seeding')
    plt.scatter([i for i in range(1, len(k_center_average) + 1)], k_center_average, color='blue', label = 'Proposed Seeding')
    plt.title("Average Cost of Clustering vs. Iterations for k = 8")
    plt.xlabel('Iteration')
    plt.ylabel('Average Cost')
    plt.legend()
    

    # TASK 4.2.5:
    optimal_k = 8
    partitions = lloyds(T, optimal_k, tmax, 1)[0]
    centers = []
    plt.figure(3)
    # Set the title, x-axis label, and y-axis label for the plot
    plt.title('Center Trajectories of Clusters for k = 8')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    for i, partition in enumerate(partitions):
        x_trajectory = [point[0] for point in partition['center']]
        y_trajectory = [point[1] for point in partition['center']]
        plt.plot(x_trajectory, y_trajectory, color=colors[i], label='Center Trajectory {}'.format(i + 1), marker='.', markersize=2)
    plt.legend()

    plt.show()
