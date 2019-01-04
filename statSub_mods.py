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

# ---------------------------------------------------------------------------------------------------------------------

# Reads in data files for use in readData.py

hdulist = fits.open('/Users/mtownsend/anaconda/Data/survey-dr7-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
hdulist2 = fits.open('/Users/mtownsend/anaconda/Data/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
hdulist3 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p005-250p010-dr7.fits') # this is one sweep file of the DECaLS data

# hdulist = fits.open('/Users/mindy/Research/Data/lrgProjectData/survey-dr5-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
# hdulist2 = fits.open('/Users/mindy/Research/Data/lrgProjectData/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
# hdulist3 = fits.open('/Users/mindy/Research/Data/lrgProjectData/sweep-240p005-250p010.fits') # this is one sweep file of the DECaLS data


SpecObj_data = hdulist[1].data
SDSS_data = hdulist2[1].data
DECaLS_data = hdulist3[1].data

# ---------------------------------------------------------------------------------------------------------------------

id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG = readData(SpecObj_data, SDSS_data, DECaLS_data)

ra_cut_LRG = ra_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
dec_cut_LRG = dec_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
gdepth_cut_LRG = gdepth_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]

print("end readdata")

# ---------------------------------------------------------------------------------------------------------------------

# plt.scatter(ra_BKG, dec_BKG, s=1, color='blue')
# plt.scatter(ra_LRG, dec_LRG, s=1, color='red')
# plt.rcParams["figure.figsize"] = [15, 15]
# plt.show()

# ---------------------------------------------------------------------------------------------------------------------

DTT_Gyr, age_Gyr, zage_Gyr, DCMR_Mpc, DCMR_Gyr, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr, V_Gpc = cosmoCalcfunc(z_LRG)

print("end cosmoCalc")

# ---------------------------------------------------------------------------------------------------------------------

row = 10
column = 10
# creates histogram for survey sources; excludes LRGs
H, xedges, yedges = np.histogram2d(rmag_BKG, color_BKG, normed=False)
# print("xedges: ", xedges)
# print("yedges: ", yedges)

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area
sd = H/(17.5) # * (3600.**2.)) # converts square degrees to square arcseconds

distance = 0.4

distance_kpc, near, gal_tree = nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, color_BKG, xedges, yedges)

print('end nearNeighbor')

# ---------------------------------------------------------------------------------------------------------------------

# Make HEALPix map

min_radius = 0.4
max_radius = 1.0
nside = 1024
npixel = hp.nside2npix(nside)
ra = np.concatenate([ra_LRG, ra_BKG])
dec = np.concatenate([dec_LRG, dec_BKG])
galdepth_g = np.concatenate([gdepth_cut_LRG, gdepth_BKG])
print('galdepth: ', galdepth_g)
print('length galdepth: ', len(galdepth_g))

# Make HEALPix map
# Convert ra/dec into theta/phi
theta_cut_LRG = []
phi_cut_LRG = []

for i in range(len(ra_cut_LRG)):
    theta_cut_LRG.append(np.radians(90. - dec_cut_LRG[i]))
    phi_cut_LRG.append(np.radians(ra_cut_LRG[i]))

theta = []
phi = []

for i in range(len(ra)):
    theta.append(np.radians(90 - dec[i]))
    phi.append(np.radians(ra[i]))

print('length phi: ', len(phi))
# Convert angles theta and phi to pixel numbers
pixnums = hp.ang2pix(nside, theta, phi, nest=True)
print('pixnums: ', pixnums)
print(pixnums[0])
print('length pixnums: ', len(pixnums))
# print(len(pix))
# print(type(pix))
# print(pix.shape)
# print(pix)
# print(len(ra))

# Create a HEALPix map from pix
mapp = np.bincount(pixnums, minlength=npixel)
print('map where ne 0: ', mapp[np.where(mapp > 0)])
print('length map ne 0: ', len(mapp[np.where(mapp > 0)]))
print('length map: ', len(mapp))

# Plot mapp
# hp.gnomview(mapp, xsize=225, rot=(-116.5, 8.25), flip='geo', nest=True)

# plt.show()

pixorder = np.argsort(pixnums)
print('length pixorder: ', len(pixorder))
pixels, pixcnts = np.unique(pixnums, return_counts=True)
print('length pixels: ', len(pixels))
print('length pixcnts: ', len(pixcnts))
pixcnts = np.insert(pixcnts, 0, 0)
pixcnts = np.cumsum(pixcnts)

# print(pixels)
# print(pixcnts)

hpxinfo = [0] * len(pixorder)
for i in range(len(pixcnts)-1):
    inds = pixorder[pixcnts[i]:pixcnts[i+1]] # try making this an array to see if it works then
    print(inds)
    pix = pixnums[inds]
    hpxinfo[inds] = (np.median(galdepth_g[inds]))

print('length hpxinfo: ', len(hpxinfo))
print('length hpxinfo ne 0: ', len(hpxinfo[np.where(hpxinfo > 0)]))
print('hpxinfo: ', hpxinfo)

hp.gnomview(hpxinfo, xsize=225, rot=(-116.5, 8.25), flip='geo', nest=True)

plt.show()

print('end program')
