import csv
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import text
import numpy as np
from matplotlib import rc

fig = plt.figure(figsize=(16/2, 9/2))

header1 = []
data1 = []
header2 = []
data2 = []

filename1 = "Lab1\Scripts\export\Lab1_Lavpassfilter_frekvensrespons.csv"

with open(filename1) as csvfile1:
    csvreader1 = csv.reader(csvfile1)

    header1 = next(csvreader1)

    for datapoint1 in csvreader1:
        values1 = [float(value1) for value1 in datapoint1]

        data1.append(values1)

f = [p[0] for p in data1]
ch11 = [(p[1]) for p in data1]
ch21 = [(p[2]) for p in data1]

plt.semilogx(f, ch21)
plt.semilogx(f, ch11)
plt.grid(True)

plt.xlabel("Frekvens (f)")
plt.ylabel("Amplitude (dB)")

plt.legend(['$|H_{in}(f)|$', '$|H_{out}(f)|$'], loc="best")

plt.axhline(y=max(ch11)-3, color='red')
plt.axvline(x=6.28, color='red')

bbox = dict(boxstyle="round", fc="0.8", edgecolor="white", facecolor="none")
arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle, angleA = 0, angleB = 90, rad = 10")
offset = 30

plt.annotate("$-3$dB, $6.28$Hz", xy=(6.28, -3), xytext=(2*offset, -offset/1), textcoords='offset points',
             bbox=bbox, arrowprops=arrowprops)
plt.xscale('log')

plt.show()
