from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pylab as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from pythonds.basic.stack import Stack
from math import *
from sklearn.neighbors import KDTree
import healpy as hp
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

ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG = readData(SpecObj_data, SDSS_data, DECaLS_data)

print("end readdata")

row = 10
column = 10
# creates histogram for survey sources; excludes LRGs
H, xedges, yedges = np.histogram2d(rmag_BKG, color_BKG, normed=False)

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area
sd = H/(17.5 * (3600.**2.)) # converts 25 square degrees to square arcseconds

print("end surface density calculation")

# cmd(rmag_BKG, color_BKG, rmag_LRG, color_LRG, xedges, yedges)

# plt.savefig("/Users/mtownsend/anaconda/GitHub/lrg-project/Plots/LRG_science_plots/cmd.pdf")

# print('end CMD')

# plt.scatter(ra_BKG, dec_BKG, s=1, color='blue')
# plt.scatter(ra_LRG, dec_LRG, s=1, color='red')
# plt.rcParams["figure.figsize"] = [15, 15]
# plt.show()

healpix(ra_BKG, dec_BKG, ra_LRG, dec_LRG, gmag_BKG, rmag_BKG, zmag_BKG, 1)
plt.show()

# plt.savefig("/Users/mtownsend/anaconda/GitHub/lrg-project/Plots/LRG_science_plots/healpix.pdf")
# # plt.savefig("/Users/mindy/Research/Plots/LRG_Project_Plots/healpix.pdf")

print('end healpix')

# plt.show()

# x = 0
# for i in range(10):
#     for j in range(10):
#         x += 1
#         bin_LRG = ((rmag_LRG > xedges[i]) & (rmag_LRG < xedges[i+1]) & (color_LRG > yedges[j]) & (color_LRG < yedges[j+1]))
#         bin_BKG = ((rmag_BKG > xedges[i]) & (rmag_BKG < xedges[i+1]) & (color_BKG > yedges[j]) & (color_BKG < yedges[j+1]))
#         ra_bin_LRG = ra_LRG[np.where(bin_LRG)]
#         dec_bin_LRG = dec_LRG[np.where(bin_LRG)]
#         ra_bin_BKG = ra_BKG[np.where(bin_BKG)]
#         dec_bin_BKG = dec_BKG[np.where(bin_BKG)]
#         healpix(ra_bin_BKG, dec_bin_BKG, ra_bin_LRG, dec_bin_LRG, gmag_BKG, rmag_BKG, zmag_BKG, x)
# #         plt.show()
# #         print('xedges: ', xedges[i], xedges[i+1])
# #         print('yedges: ', yedges[j], yedges[j+1])
#         plt.savefig('/Users/mtownsend/anaconda/GitHub/lrg-project/Plots/HEALPix_Plots/HEALPix_bins_{}.pdf'.format(str(x)))
#         # plt.show()
#
# print('end binned healpix')

print("end")