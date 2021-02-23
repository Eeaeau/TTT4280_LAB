import csv
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import text
import numpy as np
from matplotlib import rc

fig = plt.figure(figsize=(16/2, 9/2))

# from matplotlib import rc
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# # for Palatino and other serif fonts use:
# # rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
header1 = []
data1 = []
header2 = []
data2 = []

filename1 = "Plot\Lab3_bandpass_fr.csv"
# filename2 = "graf_v1_v_3.csv"
# filename2 = "v_1 og v_2 1kHz.csv"

with open(filename1) as csvfile1:
    csvreader1 = csv.reader(csvfile1)

    header1 = next(csvreader1)

    for datapoint1 in csvreader1:
        values1 = [float(value1) for value1 in datapoint1]

        data1.append(values1)

# with open(filename2) as csvfile2:
#     csvreader2 = csv.reader(csvfile2)

#     header2 = next(csvreader2)

#     for datapoint2 in csvreader2:
#         values2 = [float(value2) for value2 in datapoint2]

#         data2.append(values2)

print(header1)
print(data1[0])

# print(header2)
# print(data2[0])

f = [p[0] for p in data1]
ch11 = [(p[1]) for p in data1]
ch21 = [(p[2]) for p in data1]

# time2 = [n[0]*1000 for n in data2]
# ch22 = [(n[2]-4) for n in data2]

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# plt.plot(f, ch11, f, ch21, time2, ch22, linewidth="0.5")
plt.semilogx(f, ch21)
plt.semilogx(f, ch11)
plt.grid(True)

plt.xlabel("Frequency (f)")
plt.ylabel("Amplitude (dB)")
# plt.legend(['Channel 1 (V)', 'Channel 2 (V)', 'Channel 3 (V)'])
plt.legend(['$v_{in}(t)$', '$v_{out}(t)$'], loc="lower left")
# plt.legend(['$v_1 (t)$', '$v_2 (t)$'])
plt.axhline(y=max(ch11)-3, color='red')
plt.axvline(x=3.4315, color='red')
plt.axvline(x=20912.79105182546, color='red')

bbox = dict(boxstyle ="round", fc ="0.8") 
arrowprops = dict( 
    arrowstyle = "->", 
    connectionstyle = "angle, angleA = 0, angleB = 90, rad = 10") 
offset = 30

plt.annotate("$-3dB$", xy=(ch11.index(max(ch11)), max(ch11)-3), xytext = (2*offset, -offset/2), textcoords ='offset points', 
            bbox = bbox, arrowprops = arrowprops)

plt.annotate("$3.43$Hz", xy=(3.4315,0), xytext = (2*offset, -offset/2), textcoords ='offset points', 
            bbox = bbox, arrowprops = arrowprops)
plt.annotate("$20.91$kHz", xy=(20912.79,0), xytext = (-3*offset, -offset/2), textcoords ='offset points', 
            bbox = bbox, arrowprops = arrowprops)
plt.xscale('log')
plt.xlim(0,10**5)
plt.ylim(-3)
# plt.rcParams["legend.shadow"]

plt.show()