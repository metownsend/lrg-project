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
hdulist4 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p010-250p015-dr7.fits') # this is one sweep file of the DECaLS data

# hdulist = fits.open('/Users/mindy/Research/Data/lrgProjectData/survey-dr5-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
# hdulist2 = fits.open('/Users/mindy/Research/Data/lrgProjectData/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
# hdulist3 = fits.open('/Users/mindy/Research/Data/lrgProjectData/sweep-240p005-250p010.fits') # this is one sweep file of the DECaLS data


SpecObj_data = hdulist[1].data
SDSS_data = hdulist2[1].data
DECaLS_data1 = hdulist3[1].data
DECaLS_data2 = hdulist4[1].data

# ---------------------------------------------------------------------------------------------------------------------
# id_ALL1, ra_LRG1, dec_LRG1, ra_BKG1, dec_BKG1, rmag_BKG1, gmag_BKG1, zmag_BKG1, color_BKG1, rmag_LRG1, gmag_LRG1, zmag_LRG1, color_LRG1, z_LRG1, gdepth_LRG1, rdepth_LRG1, zdepth_LRG1, gdepth_BKG1, rdepth_BKG1, zdepth_BKG1, gobs_LRG1, robs_LRG1, zobs_LRG1, gobs_BKG1, robs_BKG1, zobs_BKG1, gflux_LRG1, rflux_LRG1, zflux_LRG1, gflux_BKG1, rflux_BKG1, zflux_BKG1 = readData(SpecObj_data, SDSS_data, DECaLS_data1)
# id_ALL2, ra_LRG2, dec_LRG2, ra_BKG2, dec_BKG2, rmag_BKG2, gmag_BKG2, zmag_BKG2, color_BKG2, rmag_LRG2, gmag_LRG2, zmag_LRG2, color_LRG2, z_LRG2, gdepth_LRG2, rdepth_LRG2, zdepth_LRG2, gdepth_BKG2, rdepth_BKG2, zdepth_BKG2, gobs_LRG2, robs_LRG2, zobs_LRG2, gobs_BKG2, robs_BKG2, zobs_BKG2, gflux_LRG2, rflux_LRG2, zflux_LRG2, gflux_BKG2, rflux_BKG2, zflux_BKG2 = readData(SpecObj_data, SDSS_data, DECaLS_data2)

# id_ALL1, ra_LRG1, dec_LRG1, ra_BKG1, dec_BKG1, z_LRG1, gdepth_LRG1, rdepth_LRG1, zdepth_LRG1, gdepth_BKG1, rdepth_BKG1, zdepth_BKG1, gobs_LRG1, robs_LRG1, zobs_LRG1, gobs_BKG1, robs_BKG1, zobs_BKG1, flux_ivar_g_LRG1, flux_ivar_r_LRG1, flux_ivar_z_LRG1, flux_ivar_g_BKG1, flux_ivar_r_BKG1, flux_ivar_z_BKG1, gflux_LRG1, rflux_LRG1, zflux_LRG1, gflux_BKG1, rflux_BKG1, zflux_BKG1 = readData(SpecObj_data, SDSS_data, DECaLS_data1)
# id_ALL2, ra_LRG2, dec_LRG2, ra_BKG2, dec_BKG2, z_LRG2, gdepth_LRG2, rdepth_LRG2, zdepth_LRG2, gdepth_BKG2, rdepth_BKG2, zdepth_BKG2, gobs_LRG2, robs_LRG2, zobs_LRG2, gobs_BKG2, robs_BKG2, zobs_BKG2, flux_ivar_g_LRG2, flux_ivar_r_LRG2, flux_ivar_z_LRG2, flux_ivar_g_BKG2, flux_ivar_r_BKG2, flux_ivar_z_BKG2, gflux_LRG2, rflux_LRG2, zflux_LRG2, gflux_BKG2, rflux_BKG2, zflux_BKG2 = readData(SpecObj_data, SDSS_data, DECaLS_data2)

id_ALL1, ra_LRG1, dec_LRG1, ra_BKG1, dec_BKG1, z_LRG1, gdepth_LRG1, rdepth_LRG1, zdepth_LRG1, gdepth_BKG1, rdepth_BKG1, zdepth_BKG1, gobs_LRG1, robs_LRG1, zobs_LRG1, gobs_BKG1, robs_BKG1, zobs_BKG1, gflux_LRG1, rflux_LRG1, zflux_LRG1, gflux_BKG1, rflux_BKG1, zflux_BKG1, w1flux_LRG1, w2flux_LRG1, w3flux_LRG1, w4flux_LRG1, w1flux_BKG1, w2flux_BKG1, w3flux_BKG1, w4flux_BKG1, plate_LRG1, tile_LRG1, specobjid_LRG1, objid_LRG1, brickid_LRG1 = readData(SpecObj_data, SDSS_data, DECaLS_data1)
id_ALL2, ra_LRG2, dec_LRG2, ra_BKG2, dec_BKG2, z_LRG2, gdepth_LRG2, rdepth_LRG2, zdepth_LRG2, gdepth_BKG2, rdepth_BKG2, zdepth_BKG2, gobs_LRG2, robs_LRG2, zobs_LRG2, gobs_BKG2, robs_BKG2, zobs_BKG2, gflux_LRG2, rflux_LRG2, zflux_LRG2, gflux_BKG2, rflux_BKG2, zflux_BKG2, w1flux_LRG2, w2flux_LRG2, w3flux_LRG2, w4flux_LRG2, w1flux_BKG2, w2flux_BKG2, w3flux_BKG2, w4flux_BKG2, plate_LRG2, tile_LRG2, specobjid_LRG2, objid_LRG2, brickid_LRG2 = readData(SpecObj_data, SDSS_data, DECaLS_data2)


ra = np.concatenate([ra_LRG1, ra_LRG2, ra_BKG1, ra_BKG2])
dec = np.concatenate([dec_LRG1, dec_LRG2, dec_BKG1, dec_BKG2])
z_LRG = np.concatenate([z_LRG1, z_LRG2])

gmag_LRG1 = 22.5 - 2.5 * np.log10(gflux_LRG1)
gmag_LRG2 = 22.5 - 2.5 * np.log10(gflux_LRG2)
rmag_LRG1 = 22.5 - 2.5 * np.log10(rflux_LRG1)
rmag_LRG2 = 22.5 - 2.5 * np.log10(rflux_LRG2)
zmag_LRG1 = 22.5 - 2.5 * np.log10(zflux_LRG1)
zmag_LRG2 = 22.5 - 2.5 * np.log10(zflux_LRG2)


# calculate BKG magnitudes

gmag_BKG1 = 22.5 - 2.5 * np.log10(gflux_BKG1)
gmag_BKG2 = 22.5 - 2.5 * np.log10(gflux_BKG2)
rmag_BKG1 = 22.5 - 2.5 * np.log10(rflux_BKG1)
rmag_BKG2 = 22.5 - 2.5 * np.log10(rflux_BKG2)
zmag_BKG1 = 22.5 - 2.5 * np.log10(zflux_BKG1)
zmag_BKG2 = 22.5 - 2.5 * np.log10(zflux_BKG2)

gmag = np.concatenate([gmag_LRG1, gmag_LRG2, gmag_BKG1, gmag_BKG2])
rmag = np.concatenate([rmag_LRG1, rmag_LRG2, rmag_BKG1, rmag_BKG2])
zmag = np.concatenate([zmag_LRG1, zmag_LRG2, zmag_BKG1, zmag_BKG2])
# color_BKG = np.concatenate([color_BKG1, color_BKG2])
# galdepth_g = np.concatenate([gdepth_LRG1, gdepth_LRG2, gdepth_BKG1, gdepth_BKG2])
# galdepth_r = np.concatenate([rdepth_LRG1, rdepth_LRG2, rdepth_BKG1, rdepth_BKG2])
# galdepth_z = np.concatenate([zdepth_LRG1, zdepth_LRG2, zdepth_BKG1, zdepth_BKG2])
# gobs = np.concatenate([gobs_LRG1, gobs_LRG2, gobs_BKG1, gobs_BKG2])
# robs = np.concatenate([robs_LRG1, robs_LRG2, robs_BKG1, robs_BKG2])
# zobs = np.concatenate([zobs_LRG1, zobs_LRG2, zobs_BKG1, zobs_BKG2])
# gflux_ivar = np.concatenate([flux_ivar_g_LRG1, flux_ivar_g_LRG2, flux_ivar_g_BKG1, flux_ivar_g_BKG2])
# rflux_ivar = np.concatenate([flux_ivar_r_LRG1, flux_ivar_r_LRG2, flux_ivar_r_BKG1, flux_ivar_r_BKG2])
# zflux_ivar = np.concatenate([flux_ivar_z_LRG1, flux_ivar_z_LRG2, flux_ivar_z_BKG1, flux_ivar_z_BKG2])
# gflux = np.concatenate([gflux_LRG1, gflux_LRG2, gflux_BKG1, gflux_BKG2])
# rflux = np.concatenate([rflux_LRG1, rflux_LRG2, rflux_BKG1, rflux_BKG2])
# zflux = np.concatenate([zflux_LRG1, zflux_LRG2, zflux_BKG1, zflux_BKG2])


print("end readdata")

# -------------------------------------------------------------------------------------------------------------------

# zdetect = np.where((zflux_ivar > 6.) & ((gflux_ivar < 6.) | (rflux_ivar < 6.)))
# print(len(zflux))
# print(len(zflux[zdetect]))
# print(ra[np.where(zflux[zdetect])])
# print(dec[np.where(zflux[zdetect])])

# # Attempt at finding non-detections
# garray = []
# for i in range(len(gflux)):
#     if ((gflux[i] > -3.*(gflux_ivar[i])**(-0.5)) & (gflux[i] < 3.*(gflux_ivar[i]**(-0.5)))):
#         garray.append(-999)
#     else:
#         garray.append(gflux[i])
#
# garray = np.array(garray)
# print(len(garray))
# print('non-detections in g: ', len(garray[np.where(garray == -999)]))
# print('detections in g: ', len(garray[np.where(garray != -999)]))
#
# print('---------')
#
# rarray = []
# for i in range(len(rflux)):
#     if ((rflux[i] > -3.*(rflux_ivar[i])**(-0.5)) & (rflux[i] < 3.*(rflux_ivar[i]**(-0.5)))):
#         rarray.append(-999)
#     else:
#         rarray.append(rflux[i])
#
# rarray = np.array(rarray)
# print(len(rarray))
# print('non-detections in r: ', len(rarray[np.where(rarray == -999)]))
# print('detections in r: ', len(rarray[np.where(rarray != -999)]))
#
# print('---------')
#
# zarray = []
# for i in range(len(zflux)):
#     if ((zflux[i] > -3.*(zflux_ivar[i])**(-0.5)) & (zflux[i] < 3.*(zflux_ivar[i]**(-0.5)))):
#         zarray.append(-999)
#     else:
#         zarray.append(zflux[i])
#
# zarray = np.array(zarray)
# print(len(zarray))
# print('non-detections in z: ', len(zarray[np.where(zarray == -999)]))
# print('detections in z: ', len(zarray[np.where(zarray != -999)]))
#
# print('---------')
#
# r_nondetect = rarray[np.where((zarray != -999) & (rarray == -999))]
# g_nondetect = garray[np.where((zarray != -999) & (garray == -999))]
#
# print('non-detections in z and r: ', len(r_nondetect))
# print('non-detections in z and g: ', len(g_nondetect))
#
# print('---------')
#
# print(len(garray[np.where((garray < 0.) & (garray != -999))]))
# print(len(rarray[np.where((rarray < 0.) & (rarray != -999))]))
# print(len(zarray[np.where((zarray < 0.) & (zarray != -999))]))
#
# print('---------')

# -------------------------------------------------------------------------------------------------------------------

# DTT_Gyr, age_Gyr, zage_Gyr, DCMR_Mpc, DCMR_Gyr, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr, V_Gpc = cosmoCalcfunc(z_LRG)

# print("end cosmoCalc")

# ---------------------------------------------------------------------------------------------------------------------

# row = 10
# column = 10
# # creates histogram for survey sources; excludes LRGs
# H, xedges, yedges = np.histogram2d(rmag_BKG, color_BKG, normed=False)
# # print("xedges: ", xedges)
# # print("yedges: ", yedges)
#
# # Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# # divided by the area
# sd = H/(25.) # * (3600.**2.)) # converts square degrees to square arcseconds
#
# distance = 0.4
#
# distance_kpc, near, gal_tree = nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, color_BKG, xedges, yedges)

# print('end nearNeighbor')

# ---------------------------------------------------------------------------------------------------------------------

# Make HEALPix map using just RA/Dec; end up with a density plot

nside = 1024
npixel = hp.nside2npix(nside)

# ra = np.concatenate([ra_LRG, ra_BKG])
# dec = np.concatenate([dec_LRG, dec_BKG])
# gmag = np.concatenate([gmag_LRG, gmag_BKG])
# rmag = np.concatenate([rmag_LRG, rmag_BKG])
# zmag = np.concatenate([zmag_LRG, zmag_BKG])
#
# ra_cut = ra[np.where(zmag <= 21.5)]
# dec_cut = dec[np.where(zmag <= 21.5)]

# plt.title("Nobs Distribution")
# plt.xlabel(r'$nobs$')
# plt.ylabel(r'$counts$')
# plt.hist(gobs, color='green', alpha=0.5, label='gmag', bins=50)
# plt.hist(robs, color='red', alpha=0.5, label='rmag', bins=50)
# plt.hist(zobs, color='blue', alpha=0.5, label='zmag', bins=50)
# plt.legend(loc="upper right", fontsize = 15)
# plt.show()

# Convert ra/dec into theta/phi
theta = []
phi = []

for i in range(len(ra_cut)):
    theta.append(np.radians(90. - dec_cut[i]))
    phi.append(np.radians(ra_cut[i]))

print('length phi: ', len(phi))
# Convert angles theta and phi to pixel numbers
pixnums = hp.ang2pix(nside, theta, phi, nest=True)
print('pixnums: ', pixnums)
print(pixnums[0])
print('length pixnums: ', len(pixnums))


# Create a HEALPix map from pix
mapp = np.bincount(pixnums, minlength=npixel)
# print('map where ne 0: ', mapp[np.where(mapp > 0)])
# print('length map == 0: ', len(mapp[np.where(mapp == 0)]))
# print('length map: ', len(mapp))

# masked_map = np.zeros(len(mapp))
# masked_map[(mapp == 0)] = 1
#
# m = hp.ma(mapp)
# m.mask = masked_map

# Plot mapp
hp.gnomview(mapp, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Density Map (nobs >= 2)')
plt.show()

# -------------------------------------------------------------------------------------------------------------------

# This section maps various values of the galaxies into a HEALPix map so the result is basically a density map of
# galaxy properties. In this code the new maps are array_g, array_r, and array_z.

pixorder = np.argsort(pixnums)
pixels, pixinverse, pixcnts = np.unique(pixnums, return_inverse=True, return_counts=True)
pixcnts = np.insert(pixcnts, 0, 0)
pixcnts = np.cumsum(pixcnts)

array_g = np.full(npixel, -1.)
array_r = np.full(npixel, -1.)
array_z = np.full(npixel, -1.)
pix = []
for i in range(len(pixcnts)-1):
    inds = pixorder[pixcnts[i]:pixcnts[i+1]]
    # print(type(inds[0]))
    pix = pixnums[inds][0]
    # print(pix)
    array_z[pix] = np.median(zmag[inds])
    # array_g[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_g[inds]))) - 9.)
    # array_r[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_r[inds]))) - 9.)
    # array_z[pix] = -2.5*(np.log10(5. / np.sqrt(np.median(galdepth_z[inds]))) - 9.)
    # array_g[pix] = np.median(gobs[inds])
    # array_r[pix] = np.median(robs[inds])
    # array_z[pix] = np.median(zobs[inds])
    # array_g[pix] = np.median(gmag[inds])
    # array_r[pix] = np.median(rmag[inds])
    # array_z[pix] = np.median(zmag[inds])


# masked_map_g = np.zeros(len(array_g))
# masked_map_g[(array_g == -1.)] = 1
#
# mg = hp.ma(array_g)
# mg.mask = masked_map_g
#
# masked_map_r = np.zeros(len(array_r))
# masked_map_r[(array_r == -1.)] = 1
#
# mr = hp.ma(array_r)
# mr.mask = masked_map_r

masked_map_z = np.zeros(len(array_z))
masked_map_z[(array_z == -1.)] = 1

mz = hp.ma(array_z)
mz.mask = masked_map_z
#
# hp.gnomview(mg, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Median gmag (nobs >= 2)')
# plt.show()
#
# hp.gnomview(mr, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Median rmag (nobs >= 2)')
# plt.show()
#
# hp.gnomview(mz, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Median zmag (nobs >= 2)')
# plt.show()

# -------------------------------------------------------------------------------------------------------------------

# Attempt to isolate only the top 2 percent of galaxies in whatever galaxy characteristic I'm looking at

# sorted_array_g = np.sort(array_g[np.where(array_g != -1.)])
# reverse_sorted_g = sorted_array_g[::-1]
# cutlen_g = len(reverse_sorted_g) * 0.98
# ng = np.rint(cutlen_g)
#
# sorted_array_r = np.sort(array_r[np.where(array_r != -1.)])
# reverse_sorted_r = sorted_array_r[::-1]
# cutlen_r = len(reverse_sorted_r) * 0.98
# nr = np.rint(cutlen_r)
#
sorted_array_z = np.sort(array_z[np.where(array_z != -1.)])
reverse_sorted_z = sorted_array_z[::-1]
cutlen_z = len(reverse_sorted_z) * 0.98
nz = np.rint(cutlen_z)
print(reverse_sorted)
print(nz)

ra98 = ra[np.where(zmag >= reverse_sorted_z[np.int64(nz)])]
dec98 = dec[np.where(zmag >= reverse_sorted_z[np.int64(nz)])]

theta98 = []
phi98 = []

for i in range(len(ra98)):
    theta98.append(np.radians(90. - dec98[i]))
    phi98.append(np.radians(ra98[i]))

# Convert angles theta and phi to pixel numbers
pixnums98 = hp.ang2pix(nside, theta98, phi98, nest=True)

# Create a HEALPix map from pix
density_map = np.bincount(pixnums98, minlength=npixel)

# masked_density = np.zeros(len(density_map))
# masked_density[(density_map == -1.)] = 1
# md = hp.ma(density_map)
# md.mask = masked_density

# Plot mapp
hp.gnomview(density_map, xsize=225, ysize=225, rot=(-116.5, 8.25), flip='geo', nest=True, title='Density Map (zmag selected; nobs >=2)')
plt.show()

# -------------------------------------------------------------------------------------------------------------------

# Creates a mask that masks out pixels with no sources. -1 flag set when array_g, array_r, and array_z are initialized

# masked_map_g = np.zeros(len(array_g))
# masked_map_g[(array_g == -1.)] = 1
#
# mg = hp.ma(array_g)
# mg.mask = masked_map_g
#
# masked_map_r = np.zeros(len(array_r))
# masked_map_r[(array_r == -1.)] = 1
#
# mr = hp.ma(array_r)
# mr.mask = masked_map_r
#
# masked_map_z = np.zeros(len(array_z))
# masked_map_z[(array_z == -1.)] = 1
#
# mz = hp.ma(array_z)
# mz.mask = masked_map_z

# -------------------------------------------------------------------------------------------------------------------

# hp.gnomview(mg, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Nobs in g', cbar=None)
# fig = plt.gcf()
# ax = plt.gca()
# image = ax.get_images()[0]
# cmap = fig.colorbar(image, ax=ax, orientation='horizontal', extend='max')
# # image.set_clim(vmax=5)
# plt.show()
#
# hp.gnomview(mr, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Nobs in r', cbar=None)
# fig = plt.gcf()
# ax = plt.gca()
# image = ax.get_images()[0]
# cmap = fig.colorbar(image, ax=ax, orientation='horizontal', extend='max')
# # image.set_clim(vmax=5)
# plt.show()
#
# hp.gnomview(mz, xsize=225, ysize=225, rot=(-116.5, 9.), flip='geo', nest=True, title='Nobs in z', cbar=None)
# fig = plt.gcf()
# ax = plt.gca()
# image = ax.get_images()[0]
# cmap = fig.colorbar(image, ax=ax, orientation='horizontal', extend='max')
# # image.set_clim(vmax=5)
# plt.show()


# plt.scatter(nobs_g, array_g, s=0.5, c='green')
# plt.xlabel(r'$gobs$')
# plt.ylabel(r'depth')
# plt.xlim(0., 32.)
# plt.ylim(23.5, 26.)
# plt.title('gobs vs. limiting depth (gmag; nobs >= 2)')
# plt.show()
#
# plt.scatter(nobs_r, array_r, s=0.5, c='red')
# plt.xlabel(r'$robs$')
# plt.ylabel(r'depth')
# plt.xlim(0., 26.)
# plt.ylim(23., 25.5)
# plt.title('robs vs. limiting depth (rmag; nobs >= 2)')
# plt.show()
#
# plt.scatter(nobs_z, array_z, s=0.5, c='blue')
# plt.xlabel(r'$zobs$')
# plt.ylabel(r'depth')
# plt.xlim(0.,18.)
# plt.ylim(22., 24.5)
# plt.title('zobs vs. limiting depth (zmag; nobs >= 2)')
# plt.show()

# plt.scatter(nobs_g, mapp, s=0.5, c='green')
# plt.xlabel(r'$gobs$')
# plt.ylabel(r'$number$ $of$ $sources$')
# plt.xlim(0.)
# plt.ylim(0.)
# plt.title('gobs vs. number of sources (gmag; nobs >= 2)')
# plt.show()
#
# plt.scatter(nobs_r, mapp, s=0.5, c='red')
# plt.xlabel(r'$robs$')
# plt.ylabel(r'$number$ $of$ $sources$')
# plt.xlim(0.)
# plt.ylim(0.)
# plt.title('robs vs. number of sources (rmag; nobs >= 2)')
# plt.show()
#
# plt.scatter(nobs_z, mapp, s=0.5, c='blue')
# plt.xlabel(r'$zobs$')
# plt.ylabel(r'$number$ $of$ $sources$')
# plt.xlim(0.)
# plt.ylim(0.)
# plt.title('zobs vs. number of sources (zmag; nobs >= 2)')
# plt.show()

# plt.title("Limiting Magnitude Distribution (gmag)")
# plt.hist(array_g[np.where(array_g != -1.)], bins=50, color='green', alpha=0.5)
# plt.xlabel(r'$limiting$ $gmag$')
# plt.ylabel(r'$counts$')
# plt.xlim(22., 26.)
# plt.gca().invert_xaxis()
# plt.axvline(x=reverse_sorted_g[np.int64(ng)] , linewidth=1, color='black')
# plt.text(24.1, 350, '{} mag'.format(np.around(reverse_sorted_g[np.int64(ng)], decimals=2)), fontsize=8)
# plt.show()
#
# plt.title("Limiting Magnitude Distribution (rmag)")
# plt.hist(array_r[np.where(array_r != -1.)], bins=50, color='red', alpha=0.5)
# plt.xlabel(r'$limiting$ $rmag$')
# plt.ylabel(r'$counts$')
# plt.gca().invert_xaxis()
# plt.xlim(21., 23.)
# plt.axvline(x=reverse_sorted_r[np.int64(nr)] , linewidth=1, color='black')
# plt.text(23.3, 550, '{} mag'.format(np.around(reverse_sorted_r[np.int64(nr)], decimals=2)), fontsize=8)
# plt.show()
#
plt.title("Limiting Magnitude Distribution (zmag)")
plt.hist(array_z[np.where(array_z != -1.)], bins=50, color='blue', alpha=0.5)
plt.xlabel(r'$limiting$ $zmag$')
plt.ylabel(r'$counts$')
plt.xlim(19., 24.)
plt.gca().invert_xaxis()
plt.axvline(x=reverse_sorted_z[np.int64(nz)], linewidth=1, color='black')
plt.text(22.4, 450, '{} mag'.format(np.around(reverse_sorted_z[np.int64(nz)], decimals=2)), fontsize=8)
plt.show()

# plt.scatter(zmag, gmag, s=0.5, c='purple')
# plt.xlabel(r'$zmag$')
# plt.ylabel(r'$gmag$')
# # plt.xlim(0.)
# # plt.ylim(0.)
# plt.title('zmag vs gmag')
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.axvline(x=reverse_sorted_z[np.int64(nz)], linewidth=1, color='black')
# plt.axhline(y=reverse_sorted_g[np.int64(ng)], linewidth=1, color='black')
# plt.show()
#
# plt.scatter(zmag, rmag, s=0.5, c='mediumvioletred')
# plt.xlabel(r'$zmag$')
# plt.ylabel(r'$rmag$')
# # plt.xlim(0.)
# # plt.ylim(0.)
# plt.title('zmag vs rmag')
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.axvline(x=reverse_sorted_z[np.int64(nz)], linewidth=1, color='black')
# plt.axhline(y=reverse_sorted_r[np.int64(nr)], linewidth=1, color='black')
# plt.show()

# depth_g_bin_1 = array_g[np.where((-1. < nobs_g) & (3 >= nobs_g))]
# depth_g_bin_2 = array_g[np.where(nobs_g > 3)]
# depth_r_bin_1 = array_r[np.where((-1. < nobs_r) & (3 >= nobs_r))]
# depth_r_bin_2 = array_r[np.where(nobs_r > 3)]
# depth_z_bin_1 = array_z[np.where((-1. < nobs_z) & (3 >= nobs_z))]
# depth_z_bin_2 = array_z[np.where(nobs_z > 3)]

# plt.title("Magnitude Distribution (median gmag per pixel)")
# plt.hist(array_g[np.where(array_g != -1.)], bins=50, color='lightgreen', alpha=0.5, label='gobs >= 3')
# plt.xlabel(r'$gmag$')
# plt.ylabel(r'$counts$')
# plt.legend(loc='upper right')
# plt.xlim(17., 27.5)
# # plt.gca().invert_xaxis()
# plt.show()
#
# plt.title("Magnitude Distribution (median rmag per pixel)")
# plt.hist(array_r[np.where(array_r != -1.)], bins=50, color='indianred', alpha=0.5, label='robs >= 3')
# plt.xlabel(r'$rmag$')
# plt.ylabel(r'$counts$')
# plt.legend(loc='upper right')
# plt.xlim(16., 25.5)
# # plt.gca().invert_xaxis()
# plt.show()
#
# plt.title("Magnitude Distribution (median zmag per pixel)")
# plt.hist(array_z[np.where(array_z != -1.)], bins=50, color='cornflowerblue', alpha=0.5, label='zobs >= 3')
# plt.xlabel(r'$$zmag$')
# plt.ylabel(r'$counts$')
# plt.legend(loc='upper right')
# plt.xlim(15.5, 24.)
# # plt.gca().invert_xaxis()
# plt.show()

print('end program')
