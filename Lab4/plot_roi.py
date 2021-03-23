import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import measure
import scipy.signal as signal
import statistics

# ---------- constants ----------------

path = "export/asta_puls.txt"
fps = 40

# ----------- functions -------------


def import_and_format(path, fps):
    with open(path, 'r') as fid:
        lines = fid.readlines()
        i = 0
        total_images = len(lines)

        data = np.empty([3, total_images], dtype=float)
        r = np.empty(total_images, dtype=float)
        g = np.empty(total_images, dtype=float)
        b = np.empty(total_images, dtype=float)
        for newline in lines:
            line = [float(e) for e in newline.split()]

            # print(line)
            r[i] = line[0]
            g[i] = line[1]
            b[i] = line[2]
            i += 1

        # print(len(lines))
        data[0] = r
        data[1] = g
        data[2] = b
        # data = np.fromfile(fid)
        # # data = data.reshape((-1, 3))
        t = np.linspace(0, 30, total_images)
    return t, data


# ------------------- main
# plt.subplot(2, 1, 1)
# t, data = import_and_format(path, fps)
n_mesurements = 5
# pulse_rgb = np.empty([3, n_mesurements])
pulse_rgb = [[],[],[]]
colors = ["r", "g", "b"]

for n in range(n_mesurements):
    t, data = import_and_format(
        "export/seb_puls_transmitans_ref_pekefing"+str(n+1)+".txt", fps)
    # print(data[0])
    # plt.plot(t, data[1], label='data')
    # plt.show()

    fc = 2  # Cut-off frequency of the filter
    w = fc / (fps / 2)  # Normalize the frequency
    b, a = signal.butter(20, w, 'low')
    output = signal.filtfilt(b, a, data)
    # plt.plot(t, output, label='filtered_lp')

    fc_hp = 0.5  # Cut-off frequency of the filter
    w = fc_hp / (fps / 2)  # Normalize the frequency
    b, a = signal.butter(2, w, 'highpass')
    output_hp = signal.filtfilt(b, a, output)
    # print("hp", output_hp)
    plt.plot(t, output_hp[1], label='filtered_lphp')
    plt.legend()
    plt.show()

    # freq_max = []
    # plt.subplots(2, 1, 2)
    # i = 0
    for i in range(3):
        spectrum= plt.magnitude_spectrum(output_hp[i], fps*60, window=np.hamming(
            len(output_hp[i])), pad_to=len(output_hp[i])+100, scale='dB', color=colors[i])

        pulse = spectrum[1][np.argmax(spectrum[0])]
        print("pulse: ", pulse)

        # pulse_rgb[i][n] = pulse
        pulse_rgb[i].append(pulse)

    plt.show()

print(pulse_rgb)

for channel in pulse_rgb:
    print("Standard Deviation of sample is % s "
          % (statistics.stdev(channel)))

    print("Mean of sample is % s "
          % (statistics.mean(channel)))
