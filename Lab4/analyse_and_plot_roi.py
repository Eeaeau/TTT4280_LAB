import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import measure
import scipy.signal as signal
import statistics

# ---------- global constants ---------------- #

path = "export/asta_puls.txt"
fps = 40
colors = ["r", "g", "b"]

# ----------- functions ------------- #


def import_and_format(path, fps):
    with open(path, 'r') as fid:
        # reads lines from file to list
        lines = fid.readlines()

        i = 0
        total_images = len(lines)

        # creates numpy arrays to fill with data
        data = np.empty([3, total_images], dtype=float)
        r = np.empty(total_images, dtype=float)
        g = np.empty(total_images, dtype=float)
        b = np.empty(total_images, dtype=float)

        # assigns data from each average pixel value to their corresponding arrays
        for line in lines:
            line = [float(e) for e in line.split()]

            r[i] = line[0]
            g[i] = line[1]
            b[i] = line[2]
            i += 1

        data[0] = r
        data[1] = g
        data[2] = b

        t = np.linspace(0, total_images/fps, total_images)

    return t, data


def analyse_accuracy(pulse_rgb):
    for ch in range(3):
        SD = statistics.stdev(pulse_rgb[ch])
        Mean = statistics.mean(pulse_rgb[ch])

        print("Standard Deviation of channel " +
              colors[ch] + " is % s " % (SD))
        print("Mean of channel " + colors[ch] + " is % s " % (Mean))


def find_SNR(ch, pulse, spectrum):

    # ------------- SNR calculation ------------- #
    interestSignal = 0
    noise = 0

    # Find the lower and upper frequency limit of the interest signal
    lF = np.floor(pulse)
    uF = np.ceil(pulse)

    freqs = spectrum[1]
    magnitude = spectrum[0]

    # Everything except pulse signal is noise
    index = np.argwhere((lF <= freqs) & (uF >= freqs))
    for j in range(len(index)):
        interestSignal += magnitude[index[j]]
    interestSignalAvg = interestSignal / len(index)

    # Everything except pulse signal in expected pulse area, 40-230bpm, is noise
    index = np.argwhere(((lF > freqs) & (freqs >= 40))
                        | ((freqs > uF) & (freqs <= 230)))
    for j in range(len(index)):
        noise += magnitude[index[j]]
    noiseAvg = noise / len(index)

    # Bruker ikke-skalerte amplitudeverdier (så de er ikke i dB)
    SNR = (interestSignalAvg)/(noiseAvg)

    # ------------- print results ------------- #

    print("Interessesnitt for fargekanal",
          colors[ch], ":", interestSignalAvg)
    print("Snitt av støy", noiseAvg)
    print("SNR (snitt av interesse/snitt av noise) for fargekanal",
          colors[ch], ":", SNR)

    return SNR


def find_pulse(n_mesurements, output_SNR=True, plot=False):

    # create list to store pulse values for each mesurement, seperated by pixel channel
    pulse_rgb = [[], [], []]

    for n in range(n_mesurements):
        t, data = import_and_format(
            "Lab4\export\seb_puls_transmitans_kald_pekefing"+str(n+1)+".txt", fps)

        fc = 2  # Cut-off frequency of the filter
        w = fc / (fps / 2)  # Normalize the frequency
        b, a = signal.butter(20, w, 'low')
        output = signal.filtfilt(b, a, data)

        fc_hp = 0.5  # Cut-off frequency of the filter
        w = fc_hp / (fps / 2)  # Normalize the frequency
        b, a = signal.butter(2, w, 'highpass')
        output_hp = signal.filtfilt(b, a, output)

        if(plot):
            plt.plot(t, output_hp[1], label='filtered_lphp')
            plt.legend()
            plt.show()

        # calculate pulse from magnitude spectrum per channel
        for ch in range(3):
            spectrum = plt.magnitude_spectrum(output_hp[ch], fps*60, window=np.hamming(
                len(output_hp[ch])), pad_to=len(output_hp[ch])+100, scale='dB', color=colors[ch])

            pulse = spectrum[1][np.argmax(spectrum[0])]

            if (output_SNR):
                find_SNR(ch, pulse, spectrum)

            pulse_rgb[ch].append(pulse)

    if (plot):
        plt.show()

    return pulse_rgb

    # ------------------- main ------------------- #


pulse_rgb = find_pulse(n_mesurements=5)

analyse_accuracy(pulse_rgb)
