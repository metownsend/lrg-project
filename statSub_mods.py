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

id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG, gobs_LRG, robs_LRG, zobs_LRG, gobs_BKG, robs_BKG, zobs_BKG = readData(SpecObj_data, SDSS_data, DECaLS_data)

ra_cut_LRG = ra_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
dec_cut_LRG = dec_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
gdepth_cut_LRG = gdepth_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
rdepth_cut_LRG = rdepth_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]
zdepth_cut_LRG = zdepth_LRG[np.where((ra_LRG > 242.) & (ra_LRG < 245.) & (dec_LRG > 7.5) & (dec_LRG < 9.))]

# five_sig_g_flux_LRG = 5. / np.sqrt(gdepth_LRG)
# five_sig_g_flux_BKG = 5. / np.sqrt(gdepth_BKG)
#
# five_sig_r_flux_LRG = 5. / np.sqrt(rdepth_LRG)
# five_sig_r_flux_BKG = 5. / np.sqrt(rdepth_BKG)
#
# five_sig_z_flux_LRG = 5. / np.sqrt(zdepth_LRG)
# five_sig_z_flux_BKG = 5. / np.sqrt(zdepth_BKG)
# #
# five_sig_g_mag_LRG = -2.5*(np.log10(5. / np.sqrt(gdepth_LRG[np.where(gdepth_LRG > 0.)])) - 9.)
# five_sig_g_mag_BKG = -2.5*(np.log10(5. / np.sqrt(gdepth_BKG[np.where(gdepth_BKG > 0.)])) - 9.)
#
# five_sig_r_mag_LRG = -2.5*(np.log10(5. / np.sqrt(rdepth_LRG[np.where(rdepth_LRG > 0.)])) - 9.)
# five_sig_r_mag_BKG = -2.5*(np.log10(5. / np.sqrt(rdepth_BKG[np.where(rdepth_BKG > 0.)])) - 9.)
#
# five_sig_z_mag_LRG = -2.5*(np.log10(5. / np.sqrt(zdepth_LRG[np.where(zdepth_LRG > 0.)])) - 9.)
# five_sig_z_mag_BKG = -2.5*(np.log10(5. / np.sqrt(zdepth_BKG[np.where(zdepth_BKG > 0.)])) - 9.)

# five_sig_g_mag_LRG = -2.5*(np.log10(5. / np.sqrt(gdepth_LRG)) - 9.)
# five_sig_g_mag_BKG = -2.5*(np.log10(5. / np.sqrt(gdepth_BKG)) - 9.)
#
# five_sig_r_mag_LRG = -2.5*(np.log10(5. / np.sqrt(rdepth_LRG)) - 9.)
# five_sig_r_mag_BKG = -2.5*(np.log10(5. / np.sqrt(rdepth_BKG)) - 9.)
#
# five_sig_z_mag_LRG = -2.5*(np.log10(5. / np.sqrt(zdepth_LRG)) - 9.)
# five_sig_z_mag_BKG = -2.5*(np.log10(5. / np.sqrt(zdepth_BKG)) - 9.)

print("end readdata")

# ---------------------------------------------------------------------------------------------------------------------

# plt.scatter(ra_BKG, dec_BKG, s=0.5, color='blue')
# plt.scatter(ra_LRG, dec_LRG, s=0.5, color='red')
# plt.rcParams["figure.figsize"] = [15, 15]
# plt.xlabel(r'$RA$')
# plt.ylabel(r'Dec')
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
print('npixel: ', npixel)
ra = np.concatenate([ra_LRG, ra_BKG])
dec = np.concatenate([dec_LRG, dec_BKG])
galdepth_g = np.concatenate([gdepth_LRG, gdepth_BKG])
galdepth_r = np.concatenate([rdepth_LRG, rdepth_BKG])
galdepth_z = np.concatenate([zdepth_LRG, zdepth_BKG])
# five_sig_flux_g = np.concatenate([five_sig_g_flux_LRG, five_sig_g_flux_BKG])
# five_sig_mag_g = np.concatenate([five_sig_g_mag_LRG, five_sig_g_mag_BKG])
# five_sig_flux_r = np.concatenate([five_sig_r_flux_LRG, five_sig_r_flux_BKG])
# five_sig_mag_r = np.concatenate([five_sig_r_mag_LRG, five_sig_r_mag_BKG])
# five_sig_flux_z = np.concatenate([five_sig_z_flux_LRG, five_sig_z_flux_BKG])
# five_sig_mag_z = np.concatenate([five_sig_z_mag_LRG, five_sig_z_mag_BKG])
gobs = np.concatenate([gobs_LRG, gobs_BKG])
robs = np.concatenate([robs_LRG, robs_BKG])
zobs = np.concatenate([zobs_LRG, zobs_BKG])


# plt.title("Nobs Distribution")
# plt.xlabel(r'$nobs$')
# plt.ylabel(r'$counts$')
# plt.hist(gobs, color='green', alpha=0.5, label='gmag', bins=50)
# plt.hist(robs, color='red', alpha=0.5, label='rmag', bins=50)
# plt.hist(zobs, color='blue', alpha=0.5, label='zmag', bins=50)
# plt.legend(loc="upper right", fontsize = 15)
# plt.show()


# print('galdepth: ', galdepth_g)
print('length galdepth_g: ', len(galdepth_g))
print('length galdepth_g ne 0: ', len(galdepth_g[np.where(galdepth_g > 0.)]))
print('length galdepth_r: ', len(galdepth_r))
print('length galdepth_r ne 0: ', len(galdepth_r[np.where(galdepth_r > 0.)]))
print('length galdepth_z: ', len(galdepth_z))
print('length galdepth_z ne 0: ', len(galdepth_z[np.where(galdepth_z > 0.)]))
# print('length ra: ', len(ra))
# print('length dec: ', len(dec))
# print('type of array: ', type(galdepth_g))

# Make HEALPix map
# Convert ra/dec into theta/phi
# theta_cut_LRG = []
# phi_cut_LRG = []
#
# for i in range(len(ra_cut_LRG)):
#     theta_cut_LRG.append(np.radians(90. - dec_cut_LRG[i]))
#     phi_cut_LRG.append(np.radians(ra_cut_LRG[i]))

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


# Create a HEALPix map from pix
mapp = np.bincount(pixnums, minlength=npixel)
print('map where ne 0: ', mapp[np.where(mapp > 0)])
print('length map == 0: ', len(mapp[np.where(mapp == 0)]))
print('length map: ', len(mapp))

# Plot mapp
hp.gnomview(mapp, xsize=200, ysize=150, rot=(-116.5, 8.25), flip='geo', nest=True, title='Density Map (all)')
plt.show()

pixorder = np.argsort(pixnums)
print('length pixorder: ', len(pixorder))
pixels, pixinverse, pixcnts = np.unique(pixnums, return_inverse=True, return_counts=True)
print('length pixels: ', len(pixels))
print('length pixinverse: ', len(pixinverse))
print('length pixcnts: ', len(pixcnts))
pixcnts = np.insert(pixcnts, 0, 0)
pixcnts = np.cumsum(pixcnts)

# print(pixels)
# print(pixcnts)

depth_g = np.full(npixel, -1.)
depth_r = np.full(npixel, -1.)
depth_z = np.full(npixel, -1.)
array_g = np.full(npixel, -1.)
array_r = np.full(npixel, -1.)
array_z = np.full(npixel, -1.)
# print('length hpxinfo: ', len(hpxinfo))
pix = []
for i in range(len(pixcnts)-1):
    inds = pixorder[pixcnts[i]:pixcnts[i+1]]
    # print(type(inds[0]))
    pix = pixnums[inds][0]
    # print(pix)
    array_g[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_g[inds]))) - 9.)
    array_r[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_r[inds]))) - 9.)
    array_z[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_z[inds]))) - 9.)
    # array_g[pix] = np.median(gobs[inds])
    # print('array_g: ', array_g[pix])
    # print(gobs[inds])
    # print('median:', np.median(gobs[inds]))
    # array_r[pix] = np.median(robs[inds])
    # array_z[pix] = np.median(zobs[inds])

# print(array_g[np.where(array_g > -999)])

# five_sig_flux_g = np.concatenate([five_sig_g_flux_LRG, five_sig_g_flux_BKG])
# five_sig_mag_g = np.concatenate([five_sig_g_mag_LRG, five_sig_g_mag_BKG])
# five_sig_flux_r = np.concatenate([five_sig_r_flux_LRG, five_sig_r_flux_BKG])
# five_sig_mag_r = np.concatenate([five_sig_r_mag_LRG, five_sig_r_mag_BKG])
# five_sig_flux_z = np.concatenate([five_sig_z_flux_LRG, five_sig_z_flux_BKG])
# five_sig_mag_z = np.concatenate([five_sig_z_mag_LRG, five_sig_z_mag_BKG])

# print('length hpxinfo: ', len(hpxinfo))
# print('pix: ', pix)
# print('pixinverse: ', pixinverse)
# galdepth_med = hpxinfo[pixorder]
# print('length hpxinfo ne 0: ', len(hpxinfo[np.where(hpxinfo > 0)]))
# print('hpxinfo: ', hpxinfo)

masked_map = hp.ma(array_g, badval = -1)

hp.gnomview(array_g, xsize=210, ysize=160, rot=(-116.5, 8.25), flip='geo', nest=True, title='median gmag depth (nobs >= 2)')
plt.show()

hp.gnomview(masked_map[0], xsize=210, ysize=160, rot=(-116.5, 8.25), flip='geo', nest=True, title='median gmag depth (nobs >= 2)')
plt.show()
#
# hp.gnomview(array_r, xsize=200, ysize=150, rot=(-116.5, 8.25), flip='geo', nest=True, title='median rmag depth (nobs >= 2)')
# plt.show()

# hp.gnomview(array_z, xsize=200, ysize=150, rot=(-116.5, 8.25), flip='geo', nest=True, title='median zmag depth (nobs >= 2)')
# plt.show()

print('end program')
