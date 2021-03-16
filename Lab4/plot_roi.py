import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.signal as signal


# ---------- constants ----------------

path = "export/finger_rec2.txt"
fps = 40

# ----------- functions ------------- 
def import_and_format(path, fps):
    with open(path, 'r') as fid:
        lines = fid.readlines()
        i=0
        total_images = len(lines)

        data = np.empty([3, total_images], dtype=float)
        r = np.empty(total_images, dtype=float)
        g = np.empty(total_images, dtype=float)
        b = np.empty(total_images, dtype=float)
        for line in lines:
            line = [float(e) for e in line.split()]
            # print(line)
            r[i] = line[0]
            g[i] = line[1]
            b[i] = line[2]
            i+=1

        # print(len(lines))
        data[0] = r
        data[1] = g
        data[2] = b
        # data = np.fromfile(fid)
        # # data = data.reshape((-1, 3))
        t = np.linspace(0,30, total_images)
    return t, data




# ------------------- main 

# t, data = import_and_format(path, fps)
t, data = import_and_format(path, fps)

print(data[0])
plt.plot(t, data[2])
plt.show()