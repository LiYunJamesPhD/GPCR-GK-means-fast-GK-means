import numpy as np
import sys

file_path = sys.argv[1]
data = []
small_num = int(sys.argv[2])  #5000
big_num = int(sys.argv[3])  #5500
random_index = np.random.choice(range(big_num), small_num, replace=False)

with open(file_path) as f:
    for line in f:
        data.append(line.strip().split(' '))

for index in range(random_index.shape[0]):
    tmp_line = data[random_index[index]]
    print(tmp_line[0])

