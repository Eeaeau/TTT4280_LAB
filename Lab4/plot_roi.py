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
        data = np.fromfile(fid, dtype=float)
        # data = data.reshape((-1, 3))
        t = np.linspace(0,30, 2000)
    return t, data




# ------------------- main 

t, data = import_and_format(path, fps)

print(data[:2])