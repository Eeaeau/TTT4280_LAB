import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.signal as signal


# ---------- constants ----------------

path = "export/finger_test.txt"
fps = 40

# ----------- functions ------------- 
def import_and_format(path, fps):
    with open(path, 'r') as fid:
        lines = fid.readlines()
        i=0
        data = np.empty([len(lines),3], dtype=float)
        total_images = len(lines)
        for line in lines:
            line = [float(e) for e in line.split()]
            # print(line)
            data[i] = line
            i+=1
        # print(len(lines))

        # data = np.fromfile(fid)
        # # data = data.reshape((-1, 3))
        t = np.linspace(0,30, total_images)
    return t, data




# ------------------- main 

# t, data = import_and_format(path, fps)
t, data = import_and_format(path, fps)

print(data.shape)
# plt.plot(t, data[0])