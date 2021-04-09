import numpy as np
import scipy.integrate as integrate


# ---------------------------- constants ----------------------------

muabo = np.genfromtxt("Scripts\Blod_Absorbtion\muabo.txt", delimiter=",")
muabd = np.genfromtxt("Scripts\Blod_Absorbtion\muabd.txt", delimiter=",")

red_wavelength = 600  # Replace with wavelength in nanometres
green_wavelength = 520  # Replace with wavelength in nanometres
blue_wavelength = 460  # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

bvf = 0.01  # Blood volume fraction, average blood amount in tissue
bvf_bloodvein = 1.00  # Blood volume fraction, average blood amount in pure blood
oxy = 0.8  # Blood oxygenation


# ---------------------------- functions ----------------------------

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])


def penetration_depth(mus, mua):
    delta = np.sqrt(1/(3*(mus+mua)*mua))
    return delta


def light_transmission(delta, thickness):
    steps = thickness/delta
    # return 0.368**steps*100
    return np.e**(-1/delta*thickness)


def light_reflection_contribution(d, C):
    return np.e**(-2*d*C)


def total_light_reflected(C, finger_thickness):
    return integrate.quad(light_reflection_contribution, 0, finger_thickness, args=(C))


def contrast(Thigh, Tlow):
    return np.abs(Thigh-Tlow)/Tlow

# ---------------------------- calculations ----------------------------


# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25  # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy  # Absorption due to
             + mua_blood_deoxy(wavelength)*(1-oxy))  # pure blood
mua = mua_blood*bvf + mua_other  # mua for tissue
mua_bloodvein = mua_blood*bvf_bloodvein + mua_other  # mua for bloodvein

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

delta = penetration_depth(musr, mua)
delta_bloodvein = penetration_depth(musr, mua_bloodvein)
print(delta)


print("prosentage of transmission through finger",
      light_transmission(delta, 0.015))

print("total reflected", total_light_reflected(1/delta[0], 0.015)[0])

print("prosentage of transmission in 300mu bloodvein",
      light_transmission(delta_bloodvein, 300*10**-6))
print("prosentage of transmission in 300mu slice of tissue",
      light_transmission(delta, 300*10**-6))

K = contrast(light_transmission(delta_bloodvein, 300*10**-6),
             light_transmission(delta, 300*10**-6))

print("kontrast", K)
