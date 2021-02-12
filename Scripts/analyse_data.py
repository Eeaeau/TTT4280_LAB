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


# Generates a normalized frequency
def convertFrequency(T_sample, f):
    #nyquist = fs/2
    omega = 2*f*T_sample

    return omega


def bandpass_transferfunc(low_freq, high_freq, T_sample, order):

    omega = []
    omega.append(convertFrequency(T_sample, low_freq))
    omega.append(convertFrequency(T_sample, high_freq))

    b, a = signal.butter(order, omega, btype='band', analog=False, output='ba')

    return b, a

def bandpass_filtering(data, low_cutfreq, high_cutfreq, T_sample, order):
    
    b, a = bandpass_transferfunc(low_cutfreq, high_cutfreq, T_sample, order)

    filtered_data = signal.filtfilt(b, a, data)
    #filtered_data = signal.lfilter(b, a, data)

    return filtered_data

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

# Code for plotting the measured angles on a unit circle
def Circle(x,y):
    return (x*x+y*y)

# Converts angle from rad to deg
def rad_to_deg(rad):
    return (360*rad)/(2*(np.pi))


# Function calculates the angle (innfallsvinkel) of the sound signal, 
# based on the measured delay/lag between the different microphones 
def find_angle(x21, x31, x32):
    return np.arctan2(np.sqrt(3)*(x21+x31), x31-x21+2*x32)

# This function finds the lag (in number of samples) between two arrays a and b
# This is done by taking the cross correlation of a and b 
# (then flipping the array around 0, since np.correlate gives the reverse array of what we are used to)
# Then finding the index of the largest argument (abs value) in the cross correlation array
# This index, when adjusted for the fact that the correlation is centered around zero, is the lag in samples between a and b
def find_lag (a, b):
    cross_corr = np.correlate(a, b, "full")
    cross_corr = np.flip(cross_corr, 0)
    plt.show()
    cross_corr_max = np.argmax(np.abs(cross_corr), axis=0)
    return cross_corr_max-len(a)+1


# Import data from bin file
sample_period, data = raspi_import('export/sample_brown_270degree_1.bin', channels)

sample_period *= 1e-6  # change unit to micro seconds

print("Sampling frequency:", 1/sample_period)

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)


# Unwanted noise is filtered from the signals
#for i in range(channels):
 #   data[:,i] = bandpass_filtering(data[:,i], 400, 600, sample_period, 3)

# define new constants
elements_removed = 4000
num_interp_samples = 2**15
num_of_samples_fixed = num_of_samples - elements_removed
sample_period_interp = (1-sample_period*elements_removed)/num_interp_samples

print("Sample frequency interp", 1/sample_period_interp)

# define new time array 
t_interp = np.linspace(start=0, stop=num_interp_samples*sample_period_interp, num=num_interp_samples)

print(len(t))
print(t_interp[-1])

# Defining a new variable data_fixed, which will contain 
data_fixed = np.empty([channels, num_of_samples_fixed])
data_interp = np.empty([channels, num_interp_samples])
# x = np.linspace(0, 1, num_interp_samples)

for i in range(channels):
    data_fixed[i] = data[:, i][elements_removed:]
    data_interp[i] = np.interp(t_interp, t[:num_of_samples_fixed], data_fixed[i])

print(t[:num_of_samples_fixed].shape)

print("Fixed", data_fixed.shape)
print("interp", data_interp.shape)

for i in range(channels):
    data_interp[i] = signal.detrend(data_interp[i], axis=0)  # removes DC component for each channel

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_interp_samples, d=sample_period_interp)

spectrum = np.empty([channels, len(freq)])

for i in range(channels):
    spectrum[i] = np.fft.fft(data_interp[i], axis=0)  # takes FFT of all channels

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
# plt.xlim(0.2, .3)
# plt.yticks(np.arange(min(data[:,0]), max(data[:,0])+1, 500))
for i in range(channels):
    plt.plot(t_interp, data_interp[i])
# 1VA+1V 2.54Vdd, 500Hz
plt.legend(["Ch1", "Ch2", "Ch3"])

# ---------------------- auto corr
crosscor_12 = abs(np.correlate(data_interp[0], data_interp[1], mode="full"))
print(crosscor_12)

plt.subplot(2, 1, 2)
plt.title("Cross correlation")
plt.xlabel("n")
plt.ylabel("Cross correlation")
plt.grid(True)
plt.stem(range(-int(len(crosscor_12)/2), int(len(crosscor_12)/2)+1), crosscor_12)  # get the power spectrum
plt.legend(["krysskorr12"])
plt.tight_layout()
plt.show()


# ----------------- find angle 

n = {}

testa = [1, 0.2, 0, 0, 0, 0, 0, 0]
testb = [0, 0, 1, 0.2, 0, 0, 0, 0]

print("Delaytest", find_lag(testa,testb))

# Finding the lag in samples for all combinations of microphones, and saving these to n
for i in range(channels):
    for j in range(channels):
        n[str(i+1)+str(j+1)]=find_lag(data_interp[i], data_interp[j])


#plt.subplot(2, 1, 2)
#plt.title("Power spectrum of signal")
#plt.xlabel("Frequency [Hz]")
#plt.ylabel("Power [dB]")
#plt.xlim(-1000, 1000)
#for i in range(channels-1):
 #   plt.plot(freq, 20*np.log(np.abs(spectrum[i])))  # get the power spectrum
#plt.legend(["Ch1", "Ch2", "Ch3"])
#plt.tight_layout()
#plt.show()





xx=np.linspace(-2,2,400)
yy=np.linspace(-2,2,400)
[X,Y]=np.meshgrid(xx,yy)

Z=Circle(X,Y)

# Arrays for storing the coordinates of angle- and microphone points
angx = []
angy = []
micx = []
micy = []

microphones = [np.pi/2, 7/6*(np.pi), 11/6*(np.pi)]
for i in range (len(microphones)):
    micx.append(np.cos(microphones[i]))
    micy.append(np.sin(microphones[i]))

angles = []
# Itererer gjennom alle filene, og finne angle i hver av disse
#angle = find_angle(n["21"], n["31"], n["32"])
#angles.append(angle)

for i in range (len(angles)):
    angx.append(np.cos(angles[i]))
    angy.append(np.sin(angles[i]))

plt.figure()
plt.grid(True)
plt.contour(X,Y,Z,[1])
plt.scatter(angx,angy)
plt.scatter(micx,micy)

i = 0
for x,y in zip(micx, micy):
    label = "Mic"+str(i)
    i += 1
    plt.annotate(label, (x,y), xytext=(0.0,10.0), textcoords="offset points", ha='center')

i = 0
for x,y in zip(angx, angy):
    label = str(round(rad_to_deg(angles[i])))
    i += 1
    plt.annotate(label, (x,y), xytext=(0.0,10.0), textcoords="offset points", ha='center')


plt.show()


print("Angle:", rad_to_deg(find_angle(n["21"], n["31"], n["32"])))
