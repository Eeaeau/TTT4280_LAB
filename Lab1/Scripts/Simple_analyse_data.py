import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.signal as signal


# ---------- constants
channels = 4
max_voltage = 1
adc_res = 4096

# ----------- functions


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


def format_and_plot(string):
    # Import data from bin file
    sample_period, data = raspi_import(
        string, channels)

    sample_period *= 1e-6  # change unit to micro seconds

    print("Sampling frequency:", 1/sample_period)

    # Generate time axis
    num_of_samples = data.shape[0]  # returns shape of matrix
    t = np.linspace(start=0, stop=num_of_samples *
                    sample_period, num=num_of_samples)

    # define new constants
    elements_removed = int(31250*.5)

    sample_duration = num_of_samples/31250  # original sample periode

    # number of samples after removing faulty samples
    num_of_samples_fixed = num_of_samples - elements_removed

    # number of samples to upscale to
    num_interp_samples = int((num_of_samples_fixed)*1)

    # new sample periode for interperated data
    sample_period_interp = (sample_duration-sample_period *
                            elements_removed)/num_interp_samples

    # ----- Defining a new data arrays ----

    # fixed array by removing first samples
    data_fixed = np.empty([channels, num_of_samples_fixed])
    # interperated array for upscaling
    data_interp = np.empty([channels, num_interp_samples])
    # define new time array
    t_interp = np.linspace(start=0, stop=num_interp_samples*sample_period_interp,
                           num=num_interp_samples)

    # assign data to new data arrays
    for i in range(channels):
        data_fixed[i] = data[:, i][elements_removed:]
        data_interp[i] = np.interp(
            t_interp, t[:num_of_samples_fixed], data_fixed[i])

    # removes DC component for each channel
    for i in range(channels):
        data_interp[i] = signal.detrend(data_interp[i], axis=0)

    # ------------- Plotting - ---------------
    fig = plt.figure(figsize=(16/1.8, 9/1.8))

    pad_width = num_interp_samples+2*16

    # channel_names = ["Ch1@$0.993V$", "Ch2@$0.689V$","Ch3@$0.386V$", "Ch4@$11.6mV$"]
    channel_names = ["Ch1", "Ch2", "Ch3", "Ch4"]

    # time
    plt.subplot(2, 1, 1)
    plt.title("Tidsdomene av signal")
    plt.xlabel("Tid [us]")
    plt.ylabel("Spenning")
    plt.grid(True)
    plt.xlim(0, .001)
    for i in range(0, channels):
        plt.plot(t_interp, data_interp[i]/adc_res*max_voltage)

    # 1V amplitude + 1V offset, 2.54Vdd
    plt.legend(channel_names, loc="center left")

    # power spectrum
    plt.subplot(2, 1, 2)
    plt.title("Doppler spektrum av signal")

    for i in range(channels):
        plt.magnitude_spectrum(
            data_interp[i]/adc_res*max_voltage, Fs=1/sample_period_interp, pad_to=pad_width, scale="dB")  # in kHz

    plt.legend(channel_names, loc="center left")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.xlim(11*10**3, 11.5*10**3)
    plt.tight_layout()
    plt.show()


# ---------------- run

format_and_plot('Lab1\Scripts\export\measurement@20kHz.bin')
