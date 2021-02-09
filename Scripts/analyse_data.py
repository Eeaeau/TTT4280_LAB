import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.signal as signal

channels = 4


def raspi_import(path, channels=4):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


def bandpass_transferfunc(low_freq, high_freq, f_s, order):
    #nyquist = fs/2

    b, a = signal.butter(order, [low_freq, high_freq],
                         'bandpass', analog=True, output='ba', fs=f_s)

    return b, a


def bandpass_filtering(data, low_cutfreq, high_cutfreq, fs, order):

    b, a = bandpass_transferfunc(low_cutfreq, high_cutfreq, fs, order)

    filtered_data = signal.lfilter(b, a, data)


# Import data from bin file
sample_period, data = raspi_import('export/measurement.bin', channels)



sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

print(t.shape)

# define new constants
elements_removed = 3000
num_interp_samples = 2**12
num_of_samples_fixed = num_of_samples - elements_removed


sample_period_interp = (1-sample_period*elements_removed)/num_interp_samples

t_interp = np.linspace(start=0, stop=num_interp_samples*sample_period_interp, num=num_interp_samples)

data_fixed = np.empty([channels, num_of_samples - elements_removed])
data_interp = np.empty([channels, num_interp_samples])
# x = np.linspace(0, 1, num_interp_samples)

for i in range(channels):
    data_fixed[i] = data[:, i][elements_removed:]
    data_interp[i] = np.interp(t_interp, t[elements_removed:], data_fixed[i])

    # data_fixed[i] = i
print("Fixed", data_fixed.shape)
print("interp", data_interp.shape)

data_interp = signal.detrend(data_interp, axis=0)  # removes DC component for each channel

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_interp_samples, d=sample_period_interp)
spectrum = np.fft.fft(data_interp, axis=0)  # takes FFT of all channels

# print(freq.shape, spectrum.shape )

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
# fig = plt.figure(figsize=(16/2.5, 9/2.5))


# print(data[:, 0])

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.grid(True)
plt.xlim(0.2, .3)
# plt.yticks(np.arange(min(data[:,0]), max(data[:,0])+1, 500))
plt.plot(t_interp, data_interp[0])
# 1VA+1V 2.54Vdd, 500Hz
plt.legend(["Ch1", "Ch2", "Ch3"])


plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.xlim(-1000, 1000)
plt.plot(freq, 20*np.log(np.abs(spectrum[0])))  # get the power spectrum
plt.legend(["Ch1$", "Ch2$", "Ch3$"])
plt.tight_layout()
plt.show()
