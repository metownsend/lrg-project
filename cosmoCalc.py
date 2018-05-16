# This is a modified Python Code for this cosmological calculator (http://www.astro.ucla.edu/~wright/CC.python),
# Which is in turn modified from http: http://www.astro.ucla.edu/~wright/CosmoCalc.html. 

# Redshift values
# redshift_cut = redshift[np.where(LOWZ_cut)]
# z = redshift_cut[np.where((gobs_LOWZ >= 3) & (robs_LOWZ >= 3) & (gflux_LOWZ > 0.) & (rflux_LOWZ > 0.))]


def cosmoCalcfunc(z):

	import numpy as np
	from math import sqrt
	from math import exp
	from math import sin
	from math import pi

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
		V_Gpc.append(4.*np.pi*((0.001*c/H0)**3)*VCM)

	return(age_Gyr, zage_Gyr, DTT_Gyr, DL_Mpc, DL_Gyr, V_Gpc, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc, DL_Gyr)
