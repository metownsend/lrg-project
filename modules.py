from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pylab as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from pythonds.basic.stack import Stack
from math import *
from sklearn.neighbors import KDTree
from lrg_plot_functions import *
from lrg_sum_functions import *
from cosmo_Calc import *
from divideByTwo import *
from readData import *
from nearNeighbors import *
from localBKG import *
from scipy import stats
from bestBkg import *
from astropy import stats
import healpy as hp
from nearNeighborsHEALPix import *
from localBKG_and_interlopersHEALPix import *
import healpy as hpy
import mpl_scatter_density

# Reads in data files for use in readData.py

hdulist = fits.open('/Users/mtownsend/anaconda/Data/survey-dr7-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
hdulist2 = fits.open('/Users/mtownsend/anaconda/Data/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
hdulist3 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p005-250p010-dr7.fits') # this is one sweep file of the DECaLS data
hdulist4 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p010-250p015-dr7.fits') # this is one sweep file of the DECaLS data

# hdulist = fits.open('/Users/mindy/Research/Data/lrgProjectData/survey-dr5-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
# hdulist2 = fits.open('/Users/mindy/Research/Data/lrgProjectData/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
# hdulist3 = fits.open('/Users/mindy/Research/Data/lrgProjectData/sweep-240p005-250p010.fits') # this is one sweep file of the DECaLS data


SpecObj_data = hdulist[1].data
SDSS_data = hdulist2[1].data
DECaLS_data1 = hdulist3[1].data
DECaLS_data2 = hdulist4[1].data

id_ALL1, ra_LRG1, dec_LRG1, ra_BKG1, dec_BKG1, z_LRG1, gdepth_LRG1, rdepth_LRG1, zdepth_LRG1, gdepth_BKG1, rdepth_BKG1, zdepth_BKG1, gobs_LRG1, robs_LRG1, zobs_LRG1, gobs_BKG1, robs_BKG1, zobs_BKG1, gflux_LRG1, rflux_LRG1, zflux_LRG1, gflux_BKG1, rflux_BKG1, zflux_BKG1, w1flux_LRG1, w2flux_LRG1, w3flux_LRG1, w4flux_LRG1, w1flux_BKG1, w2flux_BKG1, w3flux_BKG1, w4flux_BKG1, plate_LRG1, tile_LRG1, specobjid_LRG1, objid_LRG1, brickid_LRG1 = readData(SpecObj_data, SDSS_data, DECaLS_data1)

id_ALL2, ra_LRG2, dec_LRG2, ra_BKG2, dec_BKG2, z_LRG2, gdepth_LRG2, rdepth_LRG2, zdepth_LRG2, gdepth_BKG2, rdepth_BKG2, zdepth_BKG2, gobs_LRG2, robs_LRG2, zobs_LRG2, gobs_BKG2, robs_BKG2, zobs_BKG2, gflux_LRG2, rflux_LRG2, zflux_LRG2, gflux_BKG2, rflux_BKG2, zflux_BKG2, w1flux_LRG2, w2flux_LRG2, w3flux_LRG2, w4flux_LRG2, w1flux_BKG2, w2flux_BKG2, w3flux_BKG2, w4flux_BKG2, plate_LRG2, tile_LRG2, specobjid_LRG2, objid_LRG2, brickid_LRG2 = readData(SpecObj_data, SDSS_data, DECaLS_data2)

# calculate LRG magnitudes

gmag_LRG1 = 22.5 - 2.5 * np.log10(gflux_LRG1)
gmag_LRG2 = 22.5 - 2.5 * np.log10(gflux_LRG2)
rmag_LRG1 = 22.5 - 2.5 * np.log10(rflux_LRG1)
rmag_LRG2 = 22.5 - 2.5 * np.log10(rflux_LRG2)
zmag_LRG1 = 22.5 - 2.5 * np.log10(zflux_LRG1)
zmag_LRG2 = 22.5 - 2.5 * np.log10(zflux_LRG2)

gmag_LRG = np.concatenate([gmag_LRG1, gmag_LRG2])
rmag_LRG = np.concatenate([rmag_LRG1, rmag_LRG2])
zmag_LRG = np.concatenate([zmag_LRG1, zmag_LRG2])

# calculate BKG magnitudes

gmag_BKG1 = 22.5 - 2.5 * np.log10(gflux_BKG1)
gmag_BKG2 = 22.5 - 2.5 * np.log10(gflux_BKG2)
rmag_BKG1 = 22.5 - 2.5 * np.log10(rflux_BKG1)
rmag_BKG2 = 22.5 - 2.5 * np.log10(rflux_BKG2)
zmag_BKG1 = 22.5 - 2.5 * np.log10(zflux_BKG1)
zmag_BKG2 = 22.5 - 2.5 * np.log10(zflux_BKG2)

gmag_BKG = np.concatenate([gmag_BKG1, gmag_BKG2])
rmag_BKG = np.concatenate([rmag_BKG1, rmag_BKG2])
zmag_BKG = np.concatenate([zmag_BKG1, zmag_BKG2])

# calculate LRG (g-r), (r-z), and (z-w1) colors

grcolor_LRG = gmag_LRG - rmag_LRG
rzcolor_LRG = rmag_LRG - zmag_LRG

# calculate BKG (g-r), (r-z), and (z-w1) colors

grcolor_BKG = gmag_BKG - rmag_BKG
rzcolor_BKG = rmag_BKG - zmag_BKG

ra_LRG = np.concatenate([ra_LRG1, ra_LRG2])
ra_BKG = np.concatenate([ra_BKG1, ra_BKG2])
dec_LRG = np.concatenate([dec_LRG1, dec_LRG2])
dec_BKG = np.concatenate([dec_BKG1, dec_BKG2])
z_LRG = np.concatenate([z_LRG1, z_LRG2])

# cut for zmag brighter than 21.5

ra_LRG = ra_LRG[np.where(zmag_LRG <= 21.5)]
dec_LRG = dec_LRG[np.where(zmag_LRG <= 21.5)]

ra_BKG = ra_BKG[np.where(zmag_BKG <= 21.5)]
dec_BKG = dec_BKG[np.where(zmag_BKG <= 21.5)]

z_LRG = z_LRG[np.where(zmag_LRG <= 21.5)]

gmag_LRG = gmag_LRG[np.where(zmag_LRG <= 21.5)]
rmag_LRG = rmag_LRG[np.where(zmag_LRG <= 21.5)]
zmag_LRG = zmag_LRG[np.where(zmag_LRG <= 21.5)]
grcolor_LRG = grcolor_LRG[np.where(zmag_LRG <= 21.5)]
rzcolor_LRG = rzcolor_LRG[np.where(zmag_LRG <= 21.5)]

gmag_BKG = gmag_BKG[np.where(zmag_BKG <= 21.5)]
rmag_BKG = rmag_BKG[np.where(zmag_BKG <= 21.5)]
zmag_BKG = zmag_BKG[np.where(zmag_BKG <= 21.5)]
grcolor_BKG = grcolor_BKG[np.where(zmag_BKG <= 21.5)]
rzcolor_BKG = rzcolor_BKG[np.where(zmag_BKG <= 21.5)]

ra = np.concatenate([ra_LRG, ra_BKG])
dec = np.concatenate([dec_LRG, dec_BKG])

gmag = np.concatenate([gmag_LRG, gmag_BKG])
rmag = np.concatenate([rmag_LRG, rmag_BKG])
zmag = np.concatenate([zmag_LRG, zmag_BKG])

grcolor = np.concatenate([grcolor_LRG, grcolor_BKG])
rzcolor = np.concatenate([rzcolor_LRG, rzcolor_BKG])

print("end readdata")

DTT_Gyr, age_Gyr, zage_Gyr, DCMR_Mpc, DCMR_Gyr, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr, V_Gpc = cosmoCalcfunc(z_LRG)

print("end cosmoCalc")

row = 10
column = 10
# zmag = np.concatenate([zmag_LRG, zmag_BKG])
# color = np.concatenate([color_LRG, color_BKG])
# creates histogram for survey sources; excludes LRGs
# H, xedges, yedges = np.histogram2d(rmag, color, normed=False)
H1, edges = np.histogramdd((zmag, rzcolor, grcolor), bins=10, normed=False)
# print("xedges: ", xedges)
# print("yedges: ", yedges)

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area
sd = H1/(25.) # * (3600.**2.)) # converts square degrees to square arcseconds

# Make HEALPix map using just RA/Dec; end up with a density plot

nside = 1024
npixel = hpy.nside2npix(nside)

# Convert ra/dec into theta/phi
theta = []
phi = []

for i in range(len(ra)):
    theta.append(np.radians(90. - dec[i]))
    phi.append(np.radians(ra[i]))

# Convert angles theta and phi to pixel numbers

pixnums = hpy.ang2pix(nside, theta, phi, nest=True)


# distance = 0.5 # must be in Mpc
#
# distance_kpc, near, gal_tree, dist, index, num = nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag, color, xedges, yedges)
#
# print('end nearNeighbor')

distance = 0.5 # must be in Mpc

distance_kpc, near, gal_tree, dist, index, num = nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, zmag, rzcolor, grcolor, edges)

print('end nearNeighbor')

# inner_dist = 0.4 # in deg
# outer_dist = 0.5 # in deg
#
# Nbkg, indices, localBKG = localBKG_and_interlopersHEALPix(nside, inner_dist, outer_dist, ra_LRG, dec_LRG, pixnums, rmag, color, xedges, yedges, distance_kpc, kpc_DA)
#
# print("end bkg")

inner_dist = 0.4 # in deg
outer_dist = 0.5 # in deg

Nbkg, indices, omega, localBKG = localBKG_and_interlopersHEALPix(nside, inner_dist, outer_dist, ra_LRG, dec_LRG, pixnums, zmag, rzcolor, grcolor, edges, distance_kpc, kpc_DA)

omega = np.array(omega)

print("end bkg")

Nsat = np.array(near) - np.array(Nbkg)

print(len(Nsat))
print(Nsat.shape)

print("end Nsat")

totalNear(near)

plt.show()

totalNbkg(Nbkg)

plt.show()

# totalNsat(Nsat)
from scipy import stats

sumsat = []

# Sum up number of satellite galaxies for every LRG
for i in range(len(Nsat)):
    sumsat.append(np.sum(Nsat[i]))

meansat = np.mean(sumsat)
print("mean number of satellites is", meansat)

mediansat = np.median(sumsat)
print("median number of satellites is", mediansat)

sdsat = np.std(sumsat)
print("standard deviation of satellites is", sdsat)

sterr = stats.sem(sumsat)
print("standard error is", sterr)

plt.rcParams["figure.figsize"] = [10, 8]
plt.title("Histogram of the Number of Satellite Galaxies", fontsize=15)
plt.hist(sumsat, bins=100)
plt.axvline(linewidth=1, color='r')
plt.xlabel(r'$Number$ $of$ $Satellite$ $Galaxies$', fontsize=15)
plt.ylabel(r'$counts$', fontsize=15)
plt.show()