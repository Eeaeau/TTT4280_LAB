import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.signal as signal


# ---------- constants
channels = 5
max_voltage = 2.7
adc_res = 4096


def raspi_import(path, channels=5):
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


def Circle(x, y):
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


def find_lag(a, b):
    cross_corr = np.correlate(a, b, "full")
    autocorr = np.correlate(a, a, "full")
    cross_corr = cross_corr-0.5*autocorr
    # cross_corr = cross_corr-autocorr(a)
    # plt.stem(cross_corr)
    # plt.show()
    cross_corr_max = np.argmax(np.abs(cross_corr), axis=0)
    return cross_corr_max-len(a)+1


def find_timeshift(A, B, periode):
    nsamples = len(A)
    try:

        xcorr = np.correlate(A, B)
        # delta time array to match xcorr
        dt = np.arange(1-nsamples, nsamples)
        return dt[xcorr.argmax()]*periode
    except:
        print("unequal lenght")


def read_and_format(string):
    # Import data from bin file
    sample_period, data = raspi_import(
        string, channels)

    sample_period *= 1e-6  # change unit to micro seconds

    print("Sampling frequency:", 1/sample_period)

    # Generate time axis
    num_of_samples = data.shape[0]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples *
                    sample_period, num=num_of_samples)

    # Unwanted noise is filtered from the signals
    # for i in range(channels):
    #   data[:,i] = bandpass_filtering(data[:,i], 400, 600, sample_period, 3)

    # define new constants
    elements_removed = 31250*2
    sample_duration = num_of_samples/31250

    # num_interp_samples = 2**18
    num_of_samples_fixed = num_of_samples - elements_removed
    num_interp_samples = int((num_of_samples_fixed)*2)
    sample_period_interp = (sample_duration-sample_period *
                            elements_removed)/num_interp_samples

    print("Sample frequency interp", 1/sample_period_interp)

    # define new time array
    t_interp = np.linspace(start=0, stop=num_interp_samples *
                           sample_period_interp, num=num_interp_samples)

    print(len(t))
    print(t_interp[-1])

    # Defining a new variable data_fixed, which will contain
    data_fixed = np.empty([channels, num_of_samples_fixed])
    data_interp = np.empty([channels, num_interp_samples])
    # x = np.linspace(0, 1, num_interp_samples)

    for i in range(channels):
        data_fixed[i] = data[:, i][elements_removed:]
        data_interp[i] = np.interp(
            t_interp, t[:num_of_samples_fixed], data_fixed[i])

    print(t[:num_of_samples_fixed].shape)

    print("Fixed", data_fixed.shape)
    print("interp", data_interp.shape)

    for i in range(channels):
        # removes DC component for each channel
        data_interp[i] = signal.detrend(data_interp[i], axis=0)


#lab3 - radar

# Diverse vindusfunksjoner som kan multipliseres med signalet vårt.
    for i in range(3, channels):
        data_interp[i] = data_interp[i] * np.hamming(num_interp_samples)
    # data_interp[i] = data_interp[i] * np.hanning(num_interp_samples)
    # siste argument gir formen på vinduet
    # data_interp[i] = data_interp[i] * np.kaiser(num_interp_samples, 1.5)


# def doppler_find_average_velocity(data_interp, sample_period_interp):

    pad_width = 2**12

    # find timeshift ----------------
    time_shift_val = find_timeshift(
        data_interp[3], data_interp[4], sample_period_interp)/np.pi
    print("phaseshift: ", time_shift_val, "pi")
    # print(int(np.pi-abs(time_shift_val)/sample_period_interp))

    # data_interp[4] = np.roll(data_interp[4], -1000000)

    # time_shift_val2 = find_timeshift(data_interp[3], data_interp[4], sample_period_interp)/np.pi

    # print("timeshift2", time_shift_val2)

    # create fft -------------------------------------
    combined_IQ = data_interp[3] + 1j * data_interp[4]  # ADC4 = 3, ADC5= 4
    combined_IQ = np.pad(combined_IQ, pad_width)  # zeropad

    IQ_freq = np.fft.fftfreq(n=len(combined_IQ), d=sample_period_interp)
    IQ_freq_shifted = np.fft.fftshift(IQ_freq)

    print("lengths", len(combined_IQ), len(IQ_freq))

    # versjon 2
    doppler_spectrum_v2 = abs(np.fft.fft(combined_IQ))

    spectrum_max_freq = IQ_freq[np.argmax(abs(doppler_spectrum_v2))]
    Average_velocity = spectrum_max_freq/160.87

    fig = plt.figure(figsize=(16/2.5, 9/2.5))

    # # ---------------- prossesed
    plt.subplot(2, 1, 1)
    plt.title("Tidsdomene signal")
    plt.xlabel("Tid [us]")
    plt.ylabel("Spenning")
    plt.grid(True)
    # plt.xlim(0.2, .3)
    # plt.yticks(np.arange(min(data[:,0]), max(data[:,0])+1, 500))
    for i in range(3, channels):
        plt.plot(t_interp, data_interp[i]/adc_res*max_voltage)

    # 1VA+1V 2.54Vdd, 500Hz
    plt.legend(["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"])

    plt.subplot(2, 1, 2)
    plt.title("Doppler spectrum av signal")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Effekt [dB]")

    # Average speed
    print("freq: ", spectrum_max_freq)
    print("max power: ", max(doppler_spectrum_v2))

    # print("Average speed is: ", spectrum_max_freq/160.87)
    print(IQ_freq)
    plt.xlim(-1000, 1000)
    # get the power spectrum
    #plt.plot(IQ_freq, 20*np.log(np.abs(doppler_spectrum[0])))
    plt.plot(IQ_freq, doppler_spectrum_v2)  # in kHz
    plt.legend(["Doppler spectrum"])
    plt.tight_layout()
    plt.show()

    return Average_velocity


measured_avarage_velocity = []

# for i in range(2):
#     for j in range(10):
#         try:
#             measured_avarage_velocity.append(read_and_format("Scripts/export/4sec_radar_away_sample"+str(i)+str(j)+".bin"))
#         except:
#             # exit()
#             print(measured_avarage_velocity)
#         # doppler_find_average_velocity(, sample_period_interp)

read_and_format("Scripts/export/4sec_radar_away_sample"+str(10)+".bin")

print(measured_avarage_velocity)

# # Generate frequency axis and take FFT
# freq = np.fft.fftfreq(n=num_interp_samples, d=sample_period_interp)

# spectrum = np.empty([channels, len(freq)], dtype=complex)


# for i in range(channels):
#     # takes FFT of all channels
#     spectrum[i] = np.fft.fft(data_interp[i], axis=0)

# print(freq.shape, spectrum.shape )

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
# fig = plt.figure(figsize=(16/2.5, 9/2.5))


#plt.subplot(2, 1, 1)
#plt.title("Time domain signal")
#plt.xlabel("Time [us]")
# plt.ylabel("Voltage")
# plt.grid(True)
# plt.xlim(0.2, .3)
# plt.yticks(np.arange(min(data[:,0]), max(data[:,0])+1, 500))
# for i in range(3, channels):
#   plt.plot(t_interp, data_interp[i])
# 1VA+1V 2.54Vdd, 500Hz
#plt.legend(["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"])

# ---------------------- auto corr

# corr_12 = np.correlate(data_interp[0], data_interp[1], "full")

# plt.subplot(2, 1, 2)
# plt.title("Cross correlation")
# plt.xlabel("n")
# plt.ylabel("Cross correlation")
# plt.grid(True)
# plt.stem(range(-int(len(corr_12)/2), int(len(corr_12)/2)+1),
#          corr_12)  # get the power spectrum
# plt.legend(["krysskorr12"])
# plt.tight_layout()
# #plt.show()

#plt.subplot(2, 1, 2)
#plt.title("Power spectrum of signal")
#plt.xlabel("Frequency [Hz]")
#plt.ylabel("Power [dB]")
#plt.xlim(-1000, 1000)
# for i in range(channels-1):
#   plt.plot(freq, 20*np.log(np.abs(spectrum[i])))  # get the power spectrum
#plt.legend(["Ch1", "Ch2", "Ch3"])
# plt.tight_layout()
# plt.show()


# ----------------------- doppler spektrum - LAB3

# # ------------------ raw data
# plt.subplot(2, 1, 1)
# plt.title("Time domain signal")
# plt.xlabel("Time [us]")
# plt.ylabel("Voltage")
# plt.grid(True)
# # plt.xlim(0.2, .3)
# # plt.yticks(np.arange(min(data[:,0]), max(data[:,0])+1, 500))
# for i in range(3, channels):
#     plt.plot(t, data)
# # 1VA+1V 2.54Vdd, 500Hz
# plt.legend(["Ch1", "Ch2", "Ch3"])


# ------------------------------------- find angle ----------------------------------------


# # testa = [1, 0.2, 0, 0, 0, 0, 0, 0]
# # testb = [0, 0, 1, 0.2, 0, 0, 0, 0]

# # print("Delaytest", find_lag(testa,testb))


# # dictonary to hold relative crosscorelations
# n = {}

# # Finding the lag in samples for all combinations of microphones, and saving these to n
# for i in range(channels):
#     for j in range(channels):
#         n[str(i+1)+str(j+1)] = find_lag(data_interp[i], data_interp[j])


# xx = np.linspace(-2, 2, 400)
# yy = np.linspace(-2, 2, 400)
# [X, Y] = np.meshgrid(xx, yy)

# Z = Circle(X, Y)

# # Arrays for storing the coordinates of angle- and microphone points
# angx = []
# angy = []
# micx = []
# micy = []

# microphones = [np.pi/2, 11/6*(np.pi), 7/6*(np.pi)]
# for i in range(len(microphones)):
#     micx.append(np.cos(microphones[i]))
#     micy.append(np.sin(microphones[i]))

# angles = []
# # Itererer gjennom alle filene, og finne angle i hver av disse
# # obs 2 er 3 og 3 er 2 (basert på formelen, for vi har flyttet plass på mikrofon 2 og 3)
# angle = find_angle(n["21"], n["31"], n["32"])
# angles.append(angle)

# for i in range(len(angles)):
#     angx.append(np.cos(angles[i]))
#     angy.append(np.sin(angles[i]))

# plt.figure(figsize=(10, 10))
# plt.grid(True)
# plt.contour(X, Y, Z, [1])
# plt.scatter(angx, angy)
# plt.scatter(micx, micy)

# i = 1
# for x, y in zip(micx, micy):
#     label = "Mic"+str(i)
#     i += 1
#     plt.annotate(label, (x, y), xytext=(0.0, 10.0),
#                  textcoords="offset points", ha='center')

# i = 0
# for x, y in zip(angx, angy):
#     label = str(round(rad_to_deg(angles[i])))
#     i += 1
#     plt.annotate(label, (x, y), xytext=(0.0, 10.0),
#                  textcoords="offset points", ha='center')

# plt.show()
