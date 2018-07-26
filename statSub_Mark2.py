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

# ----------------------------------------------------------------------------------------

# Reading in data and assigning it to variables even though Greg seems to think it's a waste of time.

hdulist = fits.open('/Users/mtownsend/anaconda/Data/survey-dr5-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
                                                                 # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
hdulist2 = fits.open('/Users/mtownsend/anaconda/Data/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
hdulist3 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p005-250p010.fits') # this is one brick of the DECaLS data
SpecObj_data = hdulist[1].data
SDSS_data = hdulist2[1].data
DECaLS_data = hdulist3[1].data

# Put data in arrays

# Read in data from SDSS file

# Redshift of galaxies according to sdss
z = []
z = SDSS_data.field('Z') 

# Class of object
gal_class = []
gal_class = SDSS_data.field('CLASS')

# What survey the data is from
survey = []
survey = SDSS_data.field('SURVEY')

# SPECPRIMARY; set to 1 for primary observation of object, 0 otherwise
spec = []
spec = SDSS_data.field('SPECPRIMARY')

# Bitmask of spectroscopic warning values; need set to 0
zwarn_noqso = []
zwarn_noqso = SDSS_data.field('ZWARNING_NOQSO')

# Spectroscopic classification for certain redshift?
class_noqso = []
class_noqso = SDSS_data.field('CLASS_NOQSO')

# Array for LRG targets
targets = []
targets = SDSS_data.field('BOSS_TARGET1')

# Section of code to find LRG targets
	
# Function to find LOWZ targets
divideBy2Vec = np.vectorize(divideBy2)

a = divideBy2Vec(targets) # gives binary in string form

b = []
c = []

for i in range(len(a)):
	b.append(list((a[i])))
	b[i].reverse()
	
# print(b)

lrg = []

# Finds flags for BOSS LOWZ and CMASS sample
for i in range(len(b)):
	try:
		if (b[i][0] == '1') or (b[i][1] == '1'):
			lrg.append(int(1))
		else:
			lrg.append(int(0))
	except IndexError:
		pass
# 		print('pass statement')
		lrg.append(int(0))
		
lrg = np.array(lrg)

# ------------------------------------------------------------------------------------------------------------

# Read in data from SDSS row matched DECaLS file

# Object ID from survey file; value -1 for non-matches
objid_MATCHED = []
objid_MATCHED = SpecObj_data.field('OBJID')

# Add bridkid
brickid_MATCHED = []
brickid_MATCHED = SpecObj_data.field('BRICKID')

# Add brickname
brickname_MATCHED = []
brickname_MATCHED = SpecObj_data.field('BRICKNAME')

# Only galaxies included
gal_type_MATCHED = []
gal_type_MATCHED = SpecObj_data.field('TYPE') 

# RA
ra_MATCHED = []
ra_MATCHED = SpecObj_data.field('RA')

# Dec
dec_MATCHED = []
dec_MATCHED = SpecObj_data.field('DEC')

# flux_g
gflux_MATCHED = []
gflux_MATCHED = SpecObj_data.field('FLUX_G')

# flux_r
rflux_MATCHED = []
rflux_MATCHED = SpecObj_data.field('FLUX_R')

# flux_z
zflux_MATCHED = []
zflux_MATCHED = SpecObj_data.field('FLUX_Z')

# nobs == number of images that contribute to the central pixel
# nobs_g
gobs_MATCHED = []
gobs_MATCHED = SpecObj_data.field('NOBS_G')

# nobs_r
robs_MATCHED = []
robs_MATCHED = SpecObj_data.field('NOBS_R')

# nobs_z
zobs_MATCHED = []
zobs_MATCHED = SpecObj_data.field('NOBS_Z')

# Create a unique identifier by combining BRICKID and OBJID

id_MATCHED = []

for i in range(len(objid_MATCHED)):
    if (objid_MATCHED[i] == -1):
        id_MATCHED.append(-1)
    else:
        temp1 = str(brickid_MATCHED[i]) + str(objid_MATCHED[i])
        id_MATCHED.append(temp1)

id_MATCHED = np.array(id_MATCHED)
# ------------------------------------------------------------------------------------------------------------

# Read in data from DECaLS bricks

# Object ID from survey file; value -1 for non-matches
objid_ALL = []
objid_ALL = DECaLS_data.field('OBJID')
# print(len(objid_ALL))

# Add bridkid
brickid_ALL = []
brickid_ALL = DECaLS_data.field('BRICKID')
# print(len(brickid_ALL))

# Add brickname
brickname_ALL = []
brickname_ALL = DECaLS_data.field('BRICKNAME')

# Only galaxies included
gal_type_ALL = []
gal_type_ALL = DECaLS_data.field('TYPE') 

# RA
ra_ALL = []
ra_ALL = DECaLS_data.field('RA')

# Dec
dec_ALL = []
dec_ALL = DECaLS_data.field('DEC')

# flux_g
gflux_ALL = []
gflux_ALL = DECaLS_data.field('FLUX_G')

# flux_r
rflux_ALL = []
rflux_ALL = DECaLS_data.field('FLUX_R')

# flux_z
zflux_ALL = []
zflux_ALL = DECaLS_data.field('FLUX_Z')

# nobs == number of images that contribute to the central pixel
# nobs_g
gobs_ALL = []
gobs_ALL = DECaLS_data.field('NOBS_G')

# nobs_r
robs_ALL = []
robs_ALL = DECaLS_data.field('NOBS_R')

# nobs_z
zobs_ALL = []
zobs_ALL = DECaLS_data.field('NOBS_Z')

id_ALL = []

for i in range(len(objid_ALL)):
    temp2 = str(brickid_ALL[i]) + str(objid_ALL[i])
    id_ALL.append(temp2)

id_ALL = np.array(id_ALL)

# ------------------------------------------------------------------------------------------------------------

# Make cuts to separate LRGs and background galaxies

# Selects only LRGs (with other cuts)
LRG_cut = ((gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (zobs_MATCHED >= 3) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) &(zflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))

id_LRG = []
id_LRG = np.array(id_LRG)
id_LRG = id_MATCHED[np.where(LRG_cut)]

idcut = []

# This creates a list that is the length of id_ALL that matches LRGs from the DECaLS/SDSS file to the DECaLS file
# Use id_cut_noLRG == 0 to get galaxy sources that are NOT identified LRGs 
# For use in narrowing down DECaLS-only file (ie 'ALL')
for i in range(len(id_ALL)):
    if any(id_LRG == id_ALL[i]):
        idcut.append(1)
    else:
        idcut.append(0)

idcut = np.array(idcut)

z_lrg = []
ra_lrg = []
dec_lrg = []
for i in range(len(id_ALL)):
    if (idcut[i] == 1):
        z_lrg.append(z[np.where(id_MATCHED == id_ALL[i])])
        ra_lrg.append(ra_MATCHED[np.where(id_MATCHED == id_ALL[i])])
        dec_lrg.append(dec_MATCHED[np.where(id_MATCHED == id_ALL[i])])

# print('length of z_lrg:', len(z_lrg))
z_lrg = np.array(z_lrg)
z_LRG = np.concatenate(z_lrg)
ra_lrg = np.array(ra_lrg)
ra_LRG = np.concatenate(ra_lrg)
dec_lrg = np.array(dec_lrg)
dec_LRG = np.concatenate(dec_lrg)

# Cut out LRGs
no_LRG_cut = ((idcut == 0) & (gobs_ALL >= 3.) & (robs_ALL >= 3.) & (zobs_ALL >= 3.) & (gflux_ALL > 0.) & (rflux_ALL > 0.) & (zflux_ALL > 0.) & ((gal_type_ALL == 'SIMP') | (gal_type_ALL == "DEV") | (gal_type_ALL == "EXP") | (gal_type_ALL == "REX")) & (ra_ALL >= 241) & (ra_ALL <= 246) & (dec_ALL >= 6.5) & (dec_ALL <= 11.5))

# Flux cuts

# Flux in g for only LRGs
gflux_LRG = gflux_ALL[np.where(idcut == 1)]

# Flux in r for only LRGs
rflux_LRG = rflux_ALL[np.where(idcut == 1)]

# Flux in g for only LRGs
zflux_LRG = zflux_ALL[np.where(idcut == 1)]

# Flux in g for all galaxies in DECaLS
gflux_BKG = gflux_ALL[np.where(no_LRG_cut)]

# Flux in r for all galaxies in DECaLS
rflux_BKG = rflux_ALL[np.where(no_LRG_cut)]

# Flux in z for all galaxies in DECaLS
zflux_BKG = zflux_ALL[np.where(no_LRG_cut)]


# Obs cuts

# Number of images in g for only LRGs
gobs_LRG = gobs_ALL[np.where(idcut == 1)]

# Number of images in r for only LRGs
robs_LRG = robs_ALL[np.where(idcut == 1)]

# Number of images in g for only LRGs
zobs_LRG = zobs_ALL[np.where(idcut == 1)]

# Number of images in g for all galaxies in DECaLS
gobs_BKG = gobs_ALL[np.where(no_LRG_cut)]

# Number of images in r for all galaxies in DECaLS
robs_BKG = robs_ALL[np.where(no_LRG_cut)]

# Number of images in z for all galaxies in DECaLS
zobs_BKG = zobs_ALL[np.where(no_LRG_cut)]

gmag_LRG = 22.5 - 2.5 * np.log10(gflux_LRG)
rmag_LRG = 22.5 - 2.5 * np.log10(rflux_LRG)
zmag_LRG = 22.5 - 2.5 * np.log10(zflux_LRG)

color_LRG = gmag_LRG - rmag_LRG

gmag_BKG = 22.5 - 2.5 * np.log10(gflux_BKG)
rmag_BKG = 22.5 - 2.5 * np.log10(rflux_BKG)
zmag_BKG = 22.5 - 2.5 * np.log10(zflux_BKG)

color_BKG = gmag_BKG - rmag_BKG

print("end data parsing")

# ------------------------------------------------------------------------------------------------------------

# cosmoCalc function to find scale (kpc_DA)
# This is a modified Python Code for this cosmological calculator (http://www.astro.ucla.edu/~wright/CC.python),
# Which is in turn modified from http: http://www.astro.ucla.edu/~wright/CosmoCalc.html.

DTT_Gyr, age_Gyr, zage_Gyr, DCMR_Mpc, DCMR_Gyr, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr, V_Gpc = cosmoCalcfunc(z_LRG)

print("end CosmoCalc")

# ------------------------------------------------------------------------------------------------------------

# Create a 2D histogram that creates evenly-spaced bins and counts the points in each bin. H is the matrix 
# with the number of points per bin. Uses this number to calculate the surface density, by dividing the number
# of sources per bin by the area of the EDR. (This is done for every bin.)

# Histogram in color-magnitude space

# Create a 2D histogram that creates evenly-spaced bins and counts the points in each bin
# H is the matrix with the number of points per bin
# xedges, yedges are the bounds of the bins
row = 10
column = 10
# creates histogram for survey sources; excludes LRGs
H, xedges, yedges = np.histogram2d(rmag_BKG, color_BKG, normed=False)

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area 
sd = H/(17.5 * (3600.**2.)) # converts 25 square degrees to square arcseconds

print("end surface density calculation")

# ------------------------------------------------------------------------------------------------------------

# Counting NEAR NEIGHBORS (nn) using KDTree
# Result is an array of the number of near neighbors each LRG has

ra_BKG = ra_ALL[np.where(no_LRG_cut)]
dec_BKG = dec_ALL[np.where(no_LRG_cut)]

# Distance from which we are looking for satellites around the LRGs
distance = 0.5 # in Mpc
distance_kpc = distance * 10.**3. # in kpc

dist = []
for i in range(len(kpc_DA)):
    dist.append((distance_kpc / kpc_DA[i]) * 1./3600.) 

# print(type(dist))    
# Plot RA/Dec plot with circles around LRGs
# Plots LRG sources and EDR sources
# plt.scatter(ra_BKG, dec_BKG, s = 3, marker = '+', color='red')
# plt.scatter(ra_LRG, dec_LRG, s = 3, marker = 'o', color='blue')
# Draws circle of some radius around the LRG sources
# Circles too small to really see in the plot, but I have tested it with bigger radii to make sure it works if I
# ever need it.
# circle = []
# for i in range(len(ra_LRG)):
#     circle = plt.Circle((ra_LRG[i],dec_LRG[i]), dist[i], color='green', fill=False)
#     plt.gcf().gca().add_artist(circle)

# plt.xlabel(r'$RA$')
# plt.ylabel(r'$Dec$')
# plt.show()

# Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # Fake LRG sources
zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # Fake EDR sources
# print('len zip_list_LRG', len(zip_list_LRG))

# Creates a tree of EDR sources
gal_tree = KDTree(zip_list_BKG)

# returns a list of EDR sources that are within some radius r of an LRG
nn = gal_tree.query_radius(zip_list_LRG,r=dist,count_only=True)

# find indices of near neighbors
# creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
# arrays could be empty
ind = gal_tree.query_radius(zip_list_LRG,r=dist)

# Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
# NEAR is the list of 2D arrays of satellite galaxies as a funciton of color and magnitude
near = []

for i in range(len(ind)):
    # Creates a zero array if there are no near neighbors
    if len(ind[i]) == 0:
        hist2d = np.zeros((len(xedges)-1,len(yedges)-1))
        near.append(hist2d)
    # Creates a 2D histogram for satellite galaxies
    else:
        hist2d, x_notuse, y_notuse = np.histogram2d(rmag_BKG[ind[i]], color_BKG[ind[i]], bins=(xedges, yedges), normed=False)
        near.append(hist2d)

print("end near neighbor calculation")

# ------------------------------------------------------------------------------------------------------------

# Calculates NUMBER OF EXPECTED INTERLOPER GALAXIES (Nbkg) by first calculating the solid angle omega enclosed in 
# radius distance_kpc from the LRG. Then find the number of interloper galaxies by multiplying omega by the surface 
# density to find the number of interloper galaxies as a function of color and magnitude.

# Calculate solid angle omega for every radius ()
# omega = []
#
# for i in range(len(kpc_DA)):
#     omega.append((np.pi * distance_kpc**2.)/(kpc_DA[i])**2.) # in square arcsec
#
# # Multiply omega by the surface density
# Nbkg = []
#
# for i in range(len(omega)):
#     Nbkg.append(sd * omega[i])
#
# # Plots LRG sources and EDR sources
# plt.title("RA vs Dec")
# plt.scatter(ra_BKG, dec_BKG, s = 1, marker = '+', color='red')
# plt.scatter(ra_LRG, dec_LRG, s = 1, marker = '*', color='blue')
# plt.xlabel(r'$RA$')
# plt.ylabel(r'$Dec$')

# Plots circles around LRG sources 
# circle = []
# for i in range(len(ra_LRG)):
#     circle = plt.Circle((ra_LRG[i],dec_LRG[i]), dist[i], color='green', fill=False)
#     plt.gcf().gca().add_artist(circle)
#     
# prints ID numbers next to LRG
# This will be slightly different when using real data because those sources have real ID numbers
# a = np.arange(0,len(ra_LRG))
# for i, txt in enumerate(a):
#     text = plt.annotate(txt, (ra_LRG[i],dec_LRG[i]))
#     text.set_fontsize(7)
# 
# plt.show()


# Calculates NUMBER OF EXPECTED INTERLOPER GALAXIES (Nbkg) by first calculating the solid angle omega enclosed in
# radius distance_kpc from the LRG. Then find the number of interloper galaxies by multiplying omega by the surface
# density to find the number of interloper galaxies as a function of color and magnitude.

# Calculate solid angle omega for every radius ()
omega = []

for i in range(len(kpc_DA)):
    omega.append((np.pi * distance_kpc ** 2.) / (kpc_DA[i]) ** 2.)  # in square arcsec

# Counting the LOCAL BACKGROUND using KDTree
# Result is an array of the number of near neighbors each LRG has

ra_BKG = ra_ALL[np.where(no_LRG_cut)]
dec_BKG = dec_ALL[np.where(no_LRG_cut)]

# Distance from which we are looking for satellites around the LRGs
local_distance = 5.  # in Mpc
local_distance_kpc = local_distance * 10. ** 3.  # in kpc

local_dist = []
for i in range(len(kpc_DA)):
    local_dist.append((local_distance_kpc / kpc_DA[i]) * 1. / 3600.)  # needs to be in degree for kd tree because
    # coordinates are in degree

# Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
zip_list_LRG = list(zip(ra_LRG, dec_LRG))  # LRG sources
zip_list_BKG = list(zip(ra_BKG, dec_BKG))  # survey sources

# Creates a tree of EDR sources
gal_tree = KDTree(zip_list_BKG)

# returns a list of EDR sources that are within some radius r of an LRG
local_nn = gal_tree.query_radius(zip_list_LRG, r=local_dist, count_only=True)

# find indices of near neighbors
# creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
# arrays could be empty
local_ind = gal_tree.query_radius(zip_list_LRG, r=local_dist)
# print(ind)
# print(type(ind[5]))
# ind5 = ind[0]
# print(ind5)
# print(type(ind5[0]))

# Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
# LOCAL_BKG is the list of 2D arrays of survey galaxies as a funciton of color and magnitude
local_bkg = []

for i in range(len(local_ind)):
    # Creates a zero array if there are no near neighbors
    if len(local_ind[i]) == 0:
        hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
        local_bkg.append(hist2d)
    # Creates a 2D histogram for satellite galaxies
    else:
        hist2d, x_notuse, y_notuse = np.histogram2d(rmag_BKG[local_ind[i]], color_BKG[local_ind[i]],
                                                    bins=(xedges, yedges), normed=False)
        local_bkg.append(hist2d)

r = []
for i in range(len(kpc_DA)):
    r.append(distance_kpc / kpc_DA[i])

sigma = []
for i in range(len(r)):
    sigma.append((local_bkg[i] / (np.pi * r[i] ** 2.)))

print(np.shape(sigma))
print(np.shape(omega))

Nbkg = []
for i in range(len(omega)):
    Nbkg.append((sigma[i] * omega[i]) * ((np.pi * r[i] ** 2.) / (17.5 * 3600. ** 2.)))


print("end background galaxy calculation")

# ------------------------------------------------------------------------------------------------------------

# Calculate NUMBER OF SATELLITE GALAXIES (Nsat) by subtracting interloper galaxies from near neighbor galaxies as a 
# function of color and magnitude.

Nsat = np.array(near) - np.array(Nbkg)
print(len(Nsat))

print("end satellite galaxy calculation")

print("end of program")

# ------------------------------------------------------------------------------------------------------------
    

# Plots

# magHist(gmag_BKG, rmag_BKG, zmag_BKG)
# zHist(z_LRG)
# totalPlots(Nsat, Nbkg, near)
# cmd(rmag_BKG, color_BKG, rmag_LRG, color_LRG)
# z_cut_Nsat(z_LRG, Nsat)
# rmag_cut_Nsat(rmag_LRG, Nsat)
# gmag_cut_Nsat(gmag_LRG, Nsat)
# zmag_cut_Nsat(zmag_LRG, Nsat)
# rmag_cut_near(rmag_LRG, near)
# gmag_cut_near(gmag_LRG, near)
# zmag_cut_near(zmag_LRG, near)
# healpix(ra_BKG, dec_BKG, ra_LRG, dec_LRG, gmag_BKG, rmag_BKG, zmag_BKG)
# z_cut_near(z_LRG, near)
# bootplot(1000, 0.68, np.median, sumsat)
