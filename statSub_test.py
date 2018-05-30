from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pylab as plt 
import matplotlib.lines as mlines
from matplotlib.legend import Legend
# from pythonds.basic.stack import Stack
from math import *
import random
from sklearn.neighbors import KDTree
import random

# Define all relevant physical variables needed throughout the program, including:
# - number of LRG sources
# - number of survey (non-LRG) sources
# - coordinates for LRGs
# - coordinates for survey sources
# - projected radius from LRGs
# - LRG redshift
# - gmag and rmag for survey sources
# - gmag and rmag for LRGs
# - (g-r) color survey sources
# - (g-r) color for LRGs

# Number of LRG sources
lrg = 50

# Number of survey (non-LRG) sources
survey = 1000

# Fake coordinates for LRG sources
random.seed(10)
x0 = [random.uniform(0,5) for j in range(lrg)]

random.seed(10)
y0 = [random.uniform(0,5) for j in range(lrg)]

y0 = random.sample(y0, len(y0))
print("length of fake LRG sources (ra) = ", len(x0))
print("length of fake LRG sources (dec) = ", len(y0))
# print("x0 = ", x0)
# print("y0 = ", y0)
# print(type(y0))

# Fake coordinates for survey sources
x1 = [random.uniform(0,5) for j in range(survey)]

y1 = [random.uniform(0,5) for j in range(survey)]
y1 = random.sample(y1, len(y1))
    
print("length of fake survey sources (ra) = ", len(x1))
print("length of fake survey sources (dec) = ", len(y1))    

# In order for there to be at least one guaranteed satellite, combine the survey and lrg position arrays so that one
# survey source position is the same as an LRG position
x_plus = np.concatenate([x0, x1])
y_plus = np.concatenate([y0, y1])

print("length of x_plus (ra) (lrgs + survey) =", len(x_plus))
print("length of y_plus (dec) (lrg + survey) =", len(y_plus))

# Redshift for LRG 
random.seed(10)
z = [random.uniform(0.1,0.5) for j in range(lrg)]
    
print("length of array of redshifts for fake LRGs =", len(z))
print("max z = ", np.amax(z))
print("min z = ", np.amin(z))

# Magnitudes for survey sources
gmag_plus = [random.uniform(14,29) for j in range(len(x_plus))]
rmag_plus = [random.uniform(13,24) for j in range(len(x_plus))]

gmag_plus = np.array(gmag_plus)
rmag_plus = np.array(rmag_plus)
color_plus = gmag_plus - rmag_plus
    
print("length of gmag for survey soruces =", len(gmag_plus))
print("length of rmag for survey soruces =", len(rmag_plus))
print("max gmag_plus = ", np.amax(gmag_plus))
print("min gmag_plus = ", np.amin(gmag_plus))
print("max rmag_plus = ", np.amax(rmag_plus))
print("min rmag_plus = ", np.amin(rmag_plus))
print("length of color_plus = ", len(color_plus))
print("max color = ", np.amax(color_plus))
print("min color = ", np.amin(color_plus))

# Magnitudes for LRG sources
gmag_lrg = [random.uniform(17,23) for j in range(lrg)]
rmag_lrg = [random.uniform(16,21) for j in range(lrg)]
    
print("length of gmag for survey soruces =", len(gmag_lrg))
print("length of rmag for survey soruces =", len(rmag_lrg))
print("max gmag_lrg = ", np.amax(gmag_lrg))
print("min gmag_lrg = ", np.amin(gmag_lrg))
print("max rmag_lrg = ", np.amax(rmag_lrg))
print("min rmag_lrg = ", np.amin(rmag_lrg))

gmag_lrg = np.array(gmag_lrg)
rmag_lrg = np.array(rmag_lrg)
color_lrg = gmag_lrg - rmag_lrg
print("length of color_lrg = ", len(color_lrg))

distance = 1. # in Mpc
print("distance in Mpc =", distance)
distance_kpc = distance * 10.**3. # in kpc
print("distance in kpc = ", distance_kpc)

# cosmoCalc function to find comoving radial distance (DCMR_Mpc) and scale (kpc_DA)

def cosmoCalc(z):

# 	import numpy as np
# 	from math import sqrt
# 	from math import exp
# 	from math import sin
# 	from math import pi

# Calculate scale to get areas
	H0 = 69.6
	WM = 0.286
	WV = 0.714
# z = 0.209855

# initialize constants

	WR = 0.        # Omega(radiation)
	WK = 0.        # Omega curvaturve = 1-Omega(total)
	c = 299792.458 # velocity of light in km/sec
	Tyr = 977.8    # coefficent for converting 1/H into Gyr
	DTT = 0.5      # time from z to now in units of 1/H0
	DTT_Gyr = []  # value of DTT in Gyr
	age = 0.5      # age of Universe in units of 1/H0
	age_Gyr = []  # value of age in Gyr
	zage = 0.1     # age of Universe at redshift z in units of 1/H0
	zage_Gyr = [] # value of zage in Gyr
	DCMR = 0.0     # comoving radial distance in units of c/H0
	DCMR_Mpc = [] 
	DCMR_Gyr = []
	DA = 0.0       # angular size distance
	DA_Mpc = []
	DA_Gyr = []
	kpc_DA = []
	DL = 0.0       # luminosity distance
	DL_Mpc = []
	DL_Gyr = []   # DL in units of billions of light years
	V_Gpc = []
	a = 1.0        # 1/(1+z), the scale factor of the Universe
	az = 0.5       # 1/(1+z(object))

	h = H0/100.
	WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
	WK = 1.-WM-WR-WV

	for j in range(len(z)):
		az = 1.0/(1+1.0*z[j])
		age = 0.
		n=1000         # number of points in integrals
		for i in range(n):
			a = az*(i+0.5)/n
			adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
			age = age + 1./adot

		zage = az*age/n
		zage_Gyr.append((Tyr/H0)*zage)
		DTT = 0.0
		DCMR = 0.0

	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
		for i in range(n):
			a = az+(1.-az)*(i+0.5)/n
			adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
			DTT = DTT + 1./adot
			DCMR = DCMR + 1./(a*adot)

		DTT = (1.-az)*DTT/n
		DCMR = (1.-az)*DCMR/n
		age = DTT+zage
		age_Gyr.append(age*(Tyr/H0))
		DTT_Gyr.append((Tyr/H0)*DTT)
		DCMR_Gyr.append((Tyr/H0)*DCMR)
		DCMR_Mpc.append((c/H0)*DCMR)

	# tangential comoving distance

		ratio = 1.00
		x = sqrt(abs(WK))*DCMR
		if x > 0.1:
			if WK > 0:
				ratio =  0.5*(exp(x)-exp(-x))/x 
			else:
				ratio = sin(x)/x
		else:
			y = x*x
		if WK < 0: y = -y
		ratio = 1. + y/6. + y*y/120.
		DCMT = ratio*DCMR
		DA = az*DCMT
		DA_Mpc.append((c/H0)*DA)
		kpc_DA.append(DA_Mpc[j]/206.264806)
		DA_Gyr.append((Tyr/H0)*DA)
		DL = DA/(az*az)
		DL_Mpc.append((c/H0)*DL)
		DL_Gyr.append((Tyr/H0)*DL)

	# comoving volume computation

		ratio = 1.00
		x = sqrt(abs(WK))*DCMR
		if x > 0.1:
			if WK > 0:
				ratio = (0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
			else:
				ratio = (x/2. - sin(2.*x)/4.)/(x*x*x/3.)
		else:
			y = x*x
			if WK < 0: y = -y
			ratio = 1. + y/5. + (2./105.)*y*y
		VCM = ratio*DCMR*DCMR*DCMR/3.
		V_Gpc.append(4.*pi*((0.001*c/H0)**3)*VCM)

	return(age_Gyr, zage_Gyr, DTT_Gyr, DL_Mpc, DL_Gyr, V_Gpc, DA_Mpc, DA_Gyr, DCMR_Mpc, DCMR_Gyr, kpc_DA, DL_Mpc, DL_Gyr)


age_Gyr, zage_Gyr, DTT_Gyr, DL_Mpc, DL_Gyr, V_Gpc, DA_Mpc, DA_Gyr, DCMR_Mpc, DCMR_Gyr, kpc_DA, DL_Mpc, DL_Gyr = cosmoCalc(z)

# Calculate surface density as a function of color and magniude by making a 2D histogram and dividing by the area
# of the survey space

# Create a 2D histogram that creates evenly-spaced bins and counts the points in each bin
# H is the matrix with the number of points per bin
# xedges, yedges are the bounds of the bins
# H, xedges, yedges = np.histogram2d(x0, y0, bins=[5,5], normed=False)

# row = 5
# column = 5

xedges = np.array([13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.])
yedges = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10., 12., 14., 16.]) 

# xedges =  [ 13.,  15.,  17.,   19.,  21., 24.]
# yedges =  [ -10.,  -5.,   0.,   5.,  11., 16.]

H, xedges, yedges = np.histogram2d(rmag_plus, color_plus, bins=(xedges,yedges), normed=False)
print("H:")
print(H)
print('-------')

# Uses the numbers counted in the histogram to calculate a surface density: For each cell, the number of sources
# divided by the area 
sd = H/(25. * (3600.**2.)) # converts 25 square degrees to square arcseconds
print("sd:")
print(sd)
print('-------')

# Scatter plot of points with bin lines drawn
fig, ax = plt.subplots()
ax.set_xticks(xedges, minor=False)
ax.set_yticks(yedges, minor=True)
ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')

plt.scatter(rmag_plus, color_plus, s = 4, color='red')
plt.scatter(rmag_lrg, color_lrg, s = 4, color='blue')
plt.gca().invert_xaxis()
plt.title("Color-Magnitude Diagram")
plt.show()

# color codes bins by surface density with color bar; should make sense when compared to scatter plot
plt.imshow(H, cmap=plt.cm.PuRd, extent=(xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]))
plt.colorbar(orientation='vertical')
plt.gca().invert_xaxis()
plt.title("Surface Density Histogram")
plt.show()

print("xedges = ", xedges)
print("yedges = ", yedges)

# Counting near neighbors using KDTree
# Result is an array of the number of near neighbors each LRG has

dist = [] # in degree
for i in range(len(kpc_DA)):
    dist.append((distance_kpc / kpc_DA[i]) * (1./3600.)) 

print("dist is", dist)

# Plots LRG sources and EDR sources
plt.scatter(x_plus, y_plus, s = 3, color='red')
plt.scatter(x0, y0, s = 3, color='blue')

# Draws circle of some radius around the LRG sources
# Circles too small to really see in the plot, but I have tested it with bigger radii to make sure it works if I
# ever need it.
circle = []
for i in range(len(x0)):
    circle = plt.Circle((x0[i],y0[i]), dist[i], color='green', fill=False)
    plt.gcf().gca().add_artist(circle)

plt.show()

# Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
# zip_list0 = list(zip(x0, y0))
zip_list0 = list(zip(x0, y0)) # Fake LRG sources
zip_list1 = list(zip(x_plus, y_plus)) # Fake EDR sources
# print(type(zip_list))
# print(zip_list)
# print(zip_list0[0])

# Creates a tree of EDR sources
gal_tree = KDTree(zip_list1)

# returns a list of EDR sources that are within some radius r of an LRG
nn = gal_tree.query_radius(zip_list0,r=dist,count_only=True)
print("nn =", nn)
# print("length of nn = ", len(nn))

# find indices of near neighbors
ind = gal_tree.query_radius(zip_list0,r=dist)
# print("length of nn index = ", len(ind))
print("nn index = ", ind)

total_ind = np.concatenate(ind)
print("total index array: ", total_ind)
print(len(total_ind))

# Array that gives actual number of near neighbors for every LRG
num = []

for i in range(len(ind)):
    num.append(len(ind[i]))

print("num is", num)
print("length of num is", len(num))

# Create 2D histograms in bins of color and magnitude for near neighbors found above
# Result is a 2D array of the number of near neighbors for every LRG in bins of color and magnitude.

near = []

# Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
for i in range(len(ind)):
    if len(ind[i]) == 0:
        hist2d = np.zeros((len(xedges),len(yedges)))
        near.append(hist2d)
#         print("list is empty")
    else:
#         print(ind[i])
#         print(i)
        hist2d, x_notuse, y_notuse = np.histogram2d(rmag_plus[ind[i]], color_plus[ind[i]], bins=[xedges, yedges], normed=False)
        near.append(hist2d)
#         print(hist2d)

print(np.shape(near))

# Calculates number of expected interloper galaxies by first calculating the solid angle omega enclosed in radius r from 
# the LRG. Then find the number of interloper galaxies by multiplying omega by the surface density to find the 
# number of interloper galaxies as a function of color and magnitude.

area = []
area = np.pi * distance**2. # in square Mpc
# print("comoving radial distance = ",DCMR_Mpc)

# Calculate solid angle omega for every rad
omega = []

for i in range(len(kpc_DA)):
    omega.append((np.pi * distance_kpc**2.)/(kpc_DA[i])**2.) # in square arcsec

# print("rad is", rad)
# print("omega is", omega)
# print(type(omega))
# print(r[1]) 


# Multiply omega by the surface density
# Like the code above, this needs to be reshaped to make bins of color and magnitude.
Nbkg = []

for i in range(len(omega)):
    bkg = sd * omega[i]
    Nbkg.append(bkg)
#     print(i)
#     print(Nbkg[i])
    
print(np.shape(Nbkg))
print("Nbkg is", Nbkg[0])
print(len(DCMR_Mpc))
    
# Plots LRG sources and EDR sources
plt.scatter(x_plus, y_plus, s = 1, color='red')
plt.scatter(x0, y0, s = 1, color='blue')

circle = []
for i in range(len(x0)):
    circle = plt.Circle((x0[i],y0[i]), dist[i], color='green', fill=False)
    plt.gcf().gca().add_artist(circle)
    
a = np.arange(0,50)
for i, txt in enumerate(a):
    text = plt.annotate(txt, (x0[i],y0[i]), textcoords='offset points')
    text.set_fontsize(8)

# plt.xlim(0.,1.)
# plt.ylim(1.,2.)
plt.show()

# Calculate number of satellite galaxies by subtracting interloper galaxies from near neighbor galaxies as a function 
# of color and magnitude.

Nsat = np.array(near) - np.array(Nbkg)

# print(len(Nsat))
# print(np.shape(Nsat))
print(Nsat[0])

print("End of program")