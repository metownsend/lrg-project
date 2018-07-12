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

def divideBy2(decNumber):

	# from pythonds.basic.stack import Stack
# 	import numpy as np

	np.vectorize(decNumber)  
	remstack = Stack()
	
	if decNumber == 0: return "0"
	
	while decNumber > 0:
		rem = decNumber % 2
		remstack.push(rem)
		decNumber = decNumber // 2
		
	binString = ""
	while not remstack.isEmpty():
		binString = binString + str(remstack.pop())
			
	return binString
	
	
	
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
# print('length of sdss lrg array: ', len(lrg))
# print('length of lrg only array:', len(lrg[np.where(lrg == 1)]))

# ------------------------------------------------------------------------------------------------------------

# Read in data from SDSS row matched DECaLS file

# Object ID from survey file; value -1 for non-matches
objid_MATCHED = []
objid_MATCHED = SpecObj_data.field('OBJID')
# print(len(objid_LRG))
# print(type(objid_LRG[1]))

# Add bridkid
brickid_MATCHED = []
brickid_MATCHED = SpecObj_data.field('BRICKID')
# print(len(brickid_LRG))

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

# Create a unique identifier by combinding BRICKID and OBJID

id_MATCHED = []

for i in range(len(objid_MATCHED)):
    if (objid_MATCHED[i] == -1):
        id_MATCHED.append(-1)
    else:
        temp1 = str(brickid_MATCHED[i]) + str(objid_MATCHED[i])
        id_MATCHED.append(temp1)

# print('length of row matched targets in SDSS and DECaLS: ', len(id_MATCHED)) 
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
    
# print('length of DECaLS targets in brick: ', len(id_ALL))

id_ALL = np.array(id_ALL)

# print('length of id_ALL: ', len(id_ALL))

# ------------------------------------------------------------------------------------------------------------

# Make cuts to separate LRGs and background galaxies

# Selects only LRGs (with other cuts)
LRG_cut = ((gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (zobs_MATCHED >= 3) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) &(zflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))

id_LRG = []
id_LRG = np.array(id_LRG)
id_LRG = id_MATCHED[np.where(LRG_cut)]
# print('length of id_MATCHED with LRG_cut (id_LRG):', len(id_LRG))

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
# print('length of idcut:', len(idcut))
# print('length of idcut = 1 (is an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 1)]))
# print('length of idcut = 0 (is not an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 0)]))

# idcut1 = idcut[np.where(idcut == 1)] 

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

# LRG_cut = ((id_cut_LRG == 1) & (gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))
# & (brickid_LRG == brickid_ALL)
# print(len(LOWZ_cut))

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

# print('LRGs only')
# print('length gobs:', len(gobs_LRG))
# print('length robs;', len(robs_LRG))
# print('length gflux:', len(gflux_LRG))
# print('length rflux:', len(rflux_LRG))

gmag_LRG = 22.5 - 2.5 * np.log10(gflux_LRG)
rmag_LRG = 22.5 - 2.5 * np.log10(rflux_LRG)
zmag_LRG = 22.5 - 2.5 * np.log10(zflux_LRG)

color_LRG = gmag_LRG - rmag_LRG

# print("length of gmag array = ", len(gmag_LRG))
# print('shape of gmag array:', gmag_LRG.shape)
# print("length of rmag array = ",len(rmag_LRG))
# print("length of zmag array = ",len(zmag_LRG))
# print("length of color array = ", len(color_LRG))
# print("Max gmag = ", np.amax(gmag_LRG))
# print("Min gmag = ", np.amin(gmag_LRG))
# print("Max rmag = ", np.amax(rmag_LRG))
# print("Min rmag = ", np.amin(rmag_LRG))
# print("Max zmag = ", np.amax(zmag_LRG))
# print("Min zmag = ", np.amin(zmag_LRG))
# print("Min color = ", np.amin(color_LRG))
# print("Max color = ", np.amax(color_LRG))
# print("")
# print('Background only')
# print('length gobs:', len(gobs_BKG))
# print('length robs:', len(robs_BKG))
# print('length gflux:', len(gflux_BKG))
# print('length rflux:', len(rflux_BKG))

gmag_BKG = 22.5 - 2.5 * np.log10(gflux_BKG)
rmag_BKG = 22.5 - 2.5 * np.log10(rflux_BKG)
zmag_BKG = 22.5 - 2.5 * np.log10(zflux_BKG)

color_BKG = gmag_BKG - rmag_BKG

# print("length of gmag array = ", len(gmag_BKG))
# print('shape of gmag array:', gmag_LRG.shape)
# print("length of rmag array = ",len(rmag_BKG))
# print("length of zmag array = ",len(zmag_BKG))
# print("length of color array = ", len(color_BKG))
# print("Max gmag = ", np.amax(gmag_BKG))
# print("Min gmag = ", np.amin(gmag_BKG))
# print("Max rmag = ", np.amax(rmag_BKG))
# print("Min rmag = ", np.amin(rmag_BKG))
# print("Max zmag = ", np.amax(zmag_BKG))
# print("Min zmag = ", np.amin(zmag_BKG))
# print("Min color = ", np.amin(color_BKG))
# print("Max color = ", np.amax(color_BKG))

# plt.hist(gmag_BKG, bins=50, color='green', alpha=0.5)
# plt.hist(rmag_BKG, bins=50, color='red', alpha=0.5)
# plt.hist(zmag_BKG, bins=50, color='lightblue', alpha=0.5)
# plt.show()

# print("")
# # Only LRGs
# print("")
# print("Only LRGs")
# print("Max z = ", np.amax(z))
# print("Min z = ", np.amin(z))
# print('shape of z_lrg:', z_lrg.shape)
# plt.hist(z_LRG, bins=50)
# plt.show()

print("end data parsing")

# ------------------------------------------------------------------------------------------------------------

# cosmoCalc function to find scale (kpc_DA)
# This is a modified Python Code for this cosmological calculator (http://www.astro.ucla.edu/~wright/CC.python),
# Which is in turn modified from http: http://www.astro.ucla.edu/~wright/CosmoCalc.html. 

# I know this isn't ideal but for some reason the function won't import, even though it imports just fine in other
# files in the same directory

# def cosmoCalcfunc(z):
#     import numpy as np
#     from math import sqrt
#     from math import exp
#     from math import sin
#     from math import pi
# 
# # Calculate scale to get areas
#     H0 = 69.6
#     WM = 0.286
#     WV = 0.714
# # z = 0.209855
# 
# # initialize constants
# 
#     WR = 0.        # Omega(radiation)
#     WK = 0.        # Omega curvaturve = 1-Omega(total)
#     c = 299792.458 # velocity of light in km/sec
#     Tyr = 977.8    # coefficent for converting 1/H into Gyr
#     DTT = 0.5      # time from z to now in units of 1/H0
#     DTT_Gyr = []  # value of DTT in Gyr
#     age = 0.5      # age of Universe in units of 1/H0
#     age_Gyr = []  # value of age in Gyr
#     zage = 0.1     # age of Universe at redshift z in units of 1/H0
#     zage_Gyr = [] # value of zage in Gyr
#     DCMR = 0.0     # comoving radial distance in units of c/H0
#     DCMR_Mpc = [] 
#     DCMR_Gyr = []
#     DA = 0.0       # angular size distance
#     DA_Mpc = []
#     DA_Gyr = []
#     kpc_DA = []
#     DL = 0.0       # luminosity distance
#     DL_Mpc = []
#     DL_Gyr = []   # DL in units of billions of light years
#     V_Gpc = []
#     a = 1.0        # 1/(1+z), the scale factor of the Universe
#     az = 0.5       # 1/(1+z(object))
# 
#     h = H0/100.
#     WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
#     WK = 1-WM-WR-WV
# 
#     for j in range(len(z)):
#         az = 1.0/(1+1.0*z[j])
#         age = 0.
#         n=1000         # number of points in integrals
#         for i in range(n):
#             a = az*(i+0.5)/n
#             adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
#             age = age + 1./adot
# 
#         zage = az*age/n
#         zage_Gyr.append((Tyr/H0)*zage)
#         DTT = 0.0
#         DCMR = 0.0
# 
# 	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
#         for i in range(n):
#             a = az+(1-az)*(i+0.5)/n
#             adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
#             DTT = DTT + 1./adot
#             DCMR = DCMR + 1./(a*adot)
# 
#         DTT = (1.-az)*DTT/n
#         DCMR = (1.-az)*DCMR/n
#         age = DTT+zage
#         age_Gyr.append(age*(Tyr/H0))
#         DTT_Gyr.append((Tyr/H0)*DTT)
#         DCMR_Gyr.append((Tyr/H0)*DCMR)
#         DCMR_Mpc.append((c/H0)*DCMR)
# 
# 	# tangential comoving distance
# 
#         ratio = 1.00
#         x = sqrt(abs(WK))*DCMR
#         if x > 0.1:
#             if WK > 0:
#                 ratio =  0.5*(exp(x)-exp(-x))/x 
#             else:
#                 ratio = sin(x)/x
#         else:
#             y = x*x
#             if WK < 0: y = -y
#             ratio = 1. + y/6. + y*y/120.
#         DCMT = ratio*DCMR
#         DA = az*DCMT
#         DA_Mpc.append((c/H0)*DA)
#         kpc_DA.append(DA_Mpc[j]/206.264806)
#         DA_Gyr.append((Tyr/H0)*DA)
#         DL = DA/(az*az)
#         DL_Mpc.append((c/H0)*DL)
#         DL_Gyr.append((Tyr/H0)*DL)
# 
# 	# comoving volume computation
# 
#         ratio = 1.00
#         x = sqrt(abs(WK))*DCMR
#         if x > 0.1:
#             if WK > 0:
#                 ratio = (0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
#             else:
#                 ratio = (x/2. - sin(2.*x)/4.)/(x*x*x/3.)
#         else:
#             y = x*x
#             if WK < 0: y = -y
#             ratio = 1. + y/5. + (2./105.)*y*y
#         VCM = ratio*DCMR*DCMR*DCMR/3.
#         V_Gpc.append(4.*pi*((0.001*c/H0)**3)*VCM)
#         
#     return(DTT_Gyr, age_Gyr, zage_Gyr, DCMR_Mpc, DCMR_Gyr, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr, V_Gpc)

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
# print("H:")
# print(H)
# print('-------')
# print('shape H')
# print(np.shape(H))

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area 
sd = H/(17.5 * (3600.**2.)) # converts 25 square degrees to square arcseconds
# print("sd:")
# print(sd)
# print('-------')

# Scatter plot of points with bin lines drawn
# fig, ax = plt.subplots()
# ax.set_xticks(xedges, minor=False)
# ax.set_yticks(yedges, minor=True)
# ax.xaxis.grid(True, which='major')
# ax.yaxis.grid(True, which='minor')

# plt.scatter(rmag_BKG, color_BKG, s = 1, marker = '+', color='red')
# plt.scatter(rmag_LRG, color_LRG, s = 1, marker = '*', color='blue')
# plt.gca().invert_xaxis()
# plt.title("Color-Magnitude Diagram")
# plt.xlabel(r'$r-mag$')
# plt.ylabel(r'$(g-r)$ $color$')
# plt.show()

# color codes bins by surface density with color bar; should make sense when compared to scatter plot
# plt.imshow(sd, cmap=plt.cm.PuRd, extent=(xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]))
# plt.colorbar(orientation='vertical')
# plt.gca().invert_xaxis()
# plt.title("Surface Density Histogram")
# plt.show()

print("end surface density calculation")

# ------------------------------------------------------------------------------------------------------------

# Counting NEAR NEIGHBORS (nn) using KDTree
# Result is an array of the number of near neighbors each LRG has

ra_BKG = ra_ALL[np.where(no_LRG_cut)]
dec_BKG = dec_ALL[np.where(no_LRG_cut)]

# print("length ra_lrg:", len(ra_LRG))
# print("length dec_lrg:", len(dec_LRG))
# print("length ra_BKG:", len(ra_BKG))
# print("length dec_BKG:", len(dec_BKG))

# Distance from which we are looking for satellites around the LRGs
distance = 0.5 # in Mpc
distance_kpc = distance * 10.**3. # in kpc

dist = []
for i in range(len(kpc_DA)):
    dist.append((distance_kpc / kpc_DA[i]) * 1./3600.) 
    
# dist = np.concatenate(dist)
# print('length dist:', len(dist))

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
omega = []

for i in range(len(kpc_DA)):
    omega.append((np.pi * distance_kpc**2.)/(kpc_DA[i])**2.) # in square arcsec

# Multiply omega by the surface density
Nbkg = []

for i in range(len(omega)):
    Nbkg.append(sd * omega[i])
    
# Plots LRG sources and EDR sources
plt.title("RA vs Dec")
plt.scatter(ra_BKG, dec_BKG, s = 1, marker = '+', color='red')
plt.scatter(ra_LRG, dec_LRG, s = 1, marker = '*', color='blue')
plt.xlabel(r'$RA$')
plt.ylabel(r'$Dec$')

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
