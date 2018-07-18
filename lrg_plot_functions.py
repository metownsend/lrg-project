# Plotting functions

# ----------------------------------------------------------------------------------------

# Function to plot histograms for Nsat, Nbkg, and near
def totalPlots(Nsat, Nbkg, near):

	import matplotlib.pylab as plt 	
	import numpy as np
	
	sumsat = []
	sumbkg = []
	sumnear = []
	
	# Sum up number of satellite galaxies for every LRG
	for i in range(len(Nsat)):
		sumsat.append(np.sum(Nsat[i]))
	# Sum up number of background galaxies for every LRG
	for i in range(len(Nbkg)):
		sumbkg.append(np.sum(Nbkg[i]))
	# Sum up number of near neighbors for every LRG
	for i in range(len(near)):
		sumnear.append(np.sum(near[i]))

	meannear = np.mean(sumnear)
	print("mean near neighbor is", meannear)
	
	sdnear = np.std(sumnear)
	print("standard deviation of near neighbor is", sdnear)

	meanbkg = np.mean(sumbkg)
	print("mean interloper is", meanbkg)

	sdbkg = np.std(sumbkg)
	print("standard deviation of interloper is", sdbkg)

	meansat = np.mean(sumsat)
	print("mean number of satellites is", meansat)
	
	mediansat = np.median(sumsat)
	print("median number of satellites is", mediansat)

	sdsat = np.std(sumsat)
	print("standard deviation of satellites is", sdsat)

	plt.rcParams["figure.figsize"] = [10, 8]
	plt.title("Histogram of the Number of Satellite Galaxies", fontsize=15)
	plt.hist(sumsat, bins=100)
	plt.axvline(linewidth=1, color='r')
	plt.show()

	plt.rcParams["figure.figsize"] = [10, 8]
	plt.title("Histogram of the Number of Near Neighbors", fontsize=15)
	plt.hist(sumnear, bins=100)
	plt.show()

	plt.rcParams["figure.figsize"] = [10, 8]
	plt.title("Histogram of the Number of Expected Interlopers", fontsize=15)
	plt.hist(sumbkg, bins=100)
	plt.show()
  
    
# ----------------------------------------------------------------------------------------


# Function to plot a CMD
def cmd(rmag_BKG, color_BKG, rmag_LRG, color_LRG, xedges, yedges):

	import matplotlib.pyplot as plt
	import numpy as np 

	plot = fig, ax = plt.subplots(figsize=(10, 8))
	ax.set_xticks(xedges, minor=False)
	ax.set_yticks(yedges, minor=True)
	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='minor')

	plt.scatter(rmag_BKG, color_BKG, s = 1, marker = '+', color='red', label="Background")
	plt.scatter(rmag_LRG, color_LRG, s = 1, marker = '*', color='blue', label='LRGs')
	plt.gca().invert_xaxis()
	plt.title("Color-Magnitude Diagram", fontsize=15)
	plt.xlabel(r'$r-mag$')
	plt.ylabel(r'$(g-r)$ $color$')
	plt.legend(loc='upper right')
	plt.show()


# ----------------------------------------------------------------------------------------
    
    
# Function to plot histograms for Nsat, Nbkg, and near with redshift cuts
def z_cut_Nsat(z_LRG, Nsat):
	
	import matplotlib.pyplot as plt
	import numpy as np 
	
# z < 0.2
	Nsat1z = Nsat[np.where(z_LRG < 0.2)]
# print(len(Nsat1z))

	sumsat1z = []
	for i in range(len(Nsat1z)):
		sumsat1z.append(np.sum(Nsat1z[i]))
    
# 0.2 <= z < 0.3
	Nsat2z = Nsat[np.where((z_LRG >= 0.2) & (0.3 > z_LRG))]
# print(len(Nsat2z))

	sumsat2z = []
	for i in range(len(Nsat2z)):
		sumsat2z.append(np.sum(Nsat2z[i]))

# 0.3 <= z < 0.4
	Nsat3z = Nsat[np.where((z_LRG >= 0.3) & (0.4 > z_LRG))]
# print(len(Nsat3z))

	sumsat3z = []
	for i in range(len(Nsat3z)):
		sumsat3z.append(np.sum(Nsat3z[i]))
    
# 0.4 <= z < 0.5
	Nsat4z = Nsat[np.where((z_LRG >= 0.4) & (0.5 > z_LRG))]
# print(len(Nsat4z))
    
	sumsat4z = []
	for i in range(len(Nsat4z)):
		sumsat4z.append(np.sum(Nsat4z[i]))
    
# 0.5 <= z < 0.6
	Nsat5z = Nsat[np.where((z_LRG >= 0.5) & (0.6 > z_LRG))]
# print(len(Nsat5z))

	sumsat5z = []
	for i in range(len(Nsat5z)):
		sumsat5z.append(np.sum(Nsat5z[i]))
    
# 0.6 <= z < 0.7
	Nsat6z = Nsat[np.where((z_LRG >= 0.6) & (0.7 > z_LRG))]
# print(len(Nsat6z))

	sumsat6z = []
	for i in range(len(Nsat6z)):
		sumsat6z.append(np.sum(Nsat6z[i]))
    
# z >= 0.7
	Nsat7z = Nsat[np.where(z_LRG >= 0.7)]
# print(len(Nsat7z))

	sumsat7z = []
	for i in range(len(Nsat7z)):
		sumsat7z.append(np.sum(Nsat7z[i]))
    
# Calculate mean for every redshift slice
	mean_sumsat1z = np.mean(sumsat1z)
	print("mean number of satellites at z < 0.2:", mean_sumsat1z)
	mean_sumsat2z = np.mean(sumsat2z)
	print("mean number of satellites at 0.2 <= z < 0.3:", mean_sumsat2z)
	mean_sumsat3z = np.mean(sumsat3z)
	print("mean number of satellites at 0.3 <= z < 0.4:", mean_sumsat3z)
	mean_sumsat4z = np.mean(sumsat4z)
	print("mean number of satellites at 0.4 <= z < 0.5:", mean_sumsat4z)
	mean_sumsat5z = np.mean(sumsat5z)
	print("mean number of satellites at 0.5 <= z < 0.6:", mean_sumsat5z)
	mean_sumsat6z = np.mean(sumsat6z)
	print("mean number of satellites at 0.6 <= z < 0.7:", mean_sumsat6z)
	mean_sumsat7z = np.mean(sumsat7z)
	print("mean number of satellites at z < 0.7:", mean_sumsat7z)

	print('total number of Nsat arrays:', len(Nsat1z) + len(Nsat2z) + len(Nsat3z) + len(Nsat4z) + len(Nsat5z) + len(Nsat6z) + len(Nsat7z))

	plt.title("Histogram of the Number of Satellite Galaxies at Different LRG Redshift Slices")
	plt.hist(sumsat1z, bins=25, alpha=0.5, label='z < 0.2')
	plt.hist(sumsat2z, bins=25, alpha=0.5, label='0.2 <= z < 0.3')
	plt.hist(sumsat3z, bins=25, alpha=0.5, label='0.3 <= z < 0.4')
	plt.hist(sumsat4z, bins=25, alpha=0.5, label='0.4 <= z < 0.5')
	plt.hist(sumsat5z, bins=25, alpha=0.5, label='0.5 <= z < 0.6')
	plt.hist(sumsat6z, bins=25, alpha=0.5, label='0.6 <= z < 0.7')
	plt.hist(sumsat7z, bins=25, alpha=0.5, label='z >= 0.7')
	plt.xlabel(r'$satellites$')
	plt.ylabel(r'$counts$')
	plt.legend(loc='upper right')
	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Satellite Galaxies at Different LRG Redshift Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumsat1z, bins=25, color='lightblue', label='z < 0.2')
	axarr[0, 0].legend(loc='upper right')
	axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumsat2z, bins=25, color='orange',label='0.2 <= z < 0.3')
	axarr[0, 1].legend(loc='upper right')
	axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumsat3z, bins=25, color='green', label='0.3 <= z < 0.4')
	axarr[1, 0].legend(loc='upper right')
	axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumsat4z, bins=25, color='red', label='0.4 <= z < 0.5')
	axarr[1, 1].legend(loc='upper right')
	axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumsat5z, bins=25, color='purple', label='0.5 <= z < 0.6')
	axarr[2,0].legend(loc='upper right')
	axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumsat6z, bins=25, color='brown', label='0.6 <= z < 0.7')
	axarr[2,1].legend(loc='upper right')
	axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumsat7z, bins=25, color='pink', label='z >= 0.7')
	axarr[3,0].legend(loc='upper right')
	axarr[3,0].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
    # Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
    # f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='satellites', ylabel='counts')

	plt.show()


# ----------------------------------------------------------------------------------------


# Plots for near with redshift cuts
def z_cut_near(z_LRG, near):

	import matplotlib.pyplot as plt
	import numpy as np 

	near = np.array(near)

# bins of ~0.1

# z < 0.2
	near1z = near[np.where(z_LRG < 0.2)]
# print(len(Nsat1))

	sumnear1z = []
	for i in range(len(near1z)):
		sumnear1z.append(np.sum(near1z[i]))

# 0.2 <= z < 0.3
	near2z = near[np.where((z_LRG >= 0.2) & (0.3 > z_LRG))]
# print(len(Nsat2))

	sumnear2z = []
	for i in range(len(near2z)):
		sumnear2z.append(np.sum(near2z[i]))

# 0.3 <= z < 0.4
	near3z = near[np.where((z_LRG >= 0.3) & (0.4 > z_LRG))]
# print(len(Nsat3))

	sumnear3z = []
	for i in range(len(near3z)):
		sumnear3z.append(np.sum(near3z[i]))

# 0.4 <= z < 0.5
	near4z = near[np.where((z_LRG >= 0.4) & (0.5 > z_LRG))]
# print(len(Nsat4))

	sumnear4z = []
	for i in range(len(near4z)):
		sumnear4z.append(np.sum(near4z[i]))

# 0.5 <= z < 0.6
	near5z = near[np.where((z_LRG >= 0.5) & (0.6 > z_LRG))]
# print(len(Nsat5))

	sumnear5z = []
	for i in range(len(near5z)):
		sumnear5z.append(np.sum(near5z[i]))

# 0.6 <= z < 0.7
	near6z = near[np.where((z_LRG >= 0.6) & (0.7 > z_LRG))]
# print(len(Nsat6))

	sumnear6z = []
	for i in range(len(near6z)):
		sumnear6z.append(np.sum(near6z[i]))

# z >= 0.7
	near7z = near[np.where(z_LRG >= 0.7)]
# print(len(Nsat7))

	sumnear7z = []
	for i in range(len(near7z)):
		sumnear7z.append(np.sum(near7z[i]))
    
# Calculate mean for every redshift slice
	mean_sumnear1z = np.mean(sumnear1z)
	print("mean number of near neighbors at z < 0.2:", mean_sumnear1z)
	mean_sumnear2z = np.mean(sumnear2z)
	print("mean number of near neighbors at 0.2 <= z < 0.3:", mean_sumnear2z)
	mean_sumnear3z = np.mean(sumnear3z)
	print("mean number of near neighbors at 0.3 <= z < 0.4:", mean_sumnear3z)
	mean_sumnear4z = np.mean(sumnear4z)
	print("mean number of near neighbors at 0.4 <= z < 0.5:", mean_sumnear4z)
	mean_sumnear5z = np.mean(sumnear5z)
	print("mean number of near neighbors at 0.5 <= z < 0.6:", mean_sumnear5z)
	mean_sumnear6z = np.mean(sumnear6z)
	print("mean number of near neighbors at 0.6 <= z < 0.7:", mean_sumnear6z)
	mean_sumnear7z = np.mean(sumnear7z)
	print("mean number of near neighbors at z < 0.7:", mean_sumnear7z)
    
	print('total number of sumnear arrays:', len(near1z) + len(near2z) + len(near3z) + len(near4z) + len(near5z) + len(near6z) + len(near7z))

	# plt.title("Histogram of the Number of Near Neighbors at Different LRG Redshift Slices")
# 	plt.hist(sumnear1z, bins=25, alpha=0.5, label='z < 0.2')
# 	plt.hist(sumnear2z, bins=25, alpha=0.5, label='0.2 <= z < 0.3')
# 	plt.hist(sumnear3z, bins=25, alpha=0.5, label='0.3 <= z < 0.4')
# 	plt.hist(sumnear4z, bins=25, alpha=0.5, label='0.4 <= z < 0.5')
# 	plt.hist(sumnear5z, bins=25, alpha=0.5, label='0.5 <= z < 0.6')
# 	plt.hist(sumnear6z, bins=25, alpha=0.5, label='0.6 <= z < 0.7')
# 	plt.hist(sumnear7z, bins=25, alpha=0.5, label='z >= 0.7')
# 	plt.xlabel(r'$near$ $neighbors$')
# 	plt.ylabel(r'$counts$')
# 	plt.legend(loc='upper right')
# 	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Near Neighbor Galaxies at Different LRG Redshift Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumnear1z, bins=25, color='lightblue', label='z < 0.2')
	axarr[0, 0].legend(loc='upper right')
# axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumnear2z, bins=25, color='orange',label='0.2 <= z < 0.3')
	axarr[0, 1].legend(loc='upper right')
# axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumnear3z, bins=25, color='green', label='0.3 <= z < 0.4')
	axarr[1, 0].legend(loc='upper right')
# axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumnear4z, bins=25, color='red', label='0.4 <= z < 0.5')
	axarr[1, 1].legend(loc='upper right')
# axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumnear5z, bins=25, color='purple', label='0.5 <= z < 0.6')
	axarr[2,0].legend(loc='upper right')
# axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumnear6z, bins=25, color='brown', label='0.6 <= z < 0.7')
	axarr[2,1].legend(loc='upper right')
# axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumnear7z, bins=25, color='pink', label='z >= 0.7')
	axarr[3,0].legend(loc='upper right')
# axarr[3,0].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='near neighbors', ylabel='counts')

	plt.show()


# ----------------------------------------------------------------------------------------


# Plot distribution of Nsat in different LRG rmag bins
def rmag_cut_Nsat(rmag_LRG, Nsat):
	
	import matplotlib.pyplot as plt
	import numpy as np 

	rmag_LRG = np.array(rmag_LRG)
    
# bins of ~1 mag

# 15 <= rmag < 16
	Nsat1r = Nsat[np.where((rmag_LRG >= 15.) & (16. > rmag_LRG))]
# print(len(Nsat1))

	sumsat1r = []
	for i in range(len(Nsat1r)):
		sumsat1r.append(np.sum(Nsat1r[i]))

# 16 <= rmag < 17
	Nsat2r = Nsat[np.where((rmag_LRG >= 16.) & (17. > rmag_LRG))]
# print(len(Nsat2))

	sumsat2r = []
	for i in range(len(Nsat2r)):
		sumsat2r.append(np.sum(Nsat2r[i]))

# 17 <= rmag < 18
	Nsat3r = Nsat[np.where((rmag_LRG >= 17.) & (18. > rmag_LRG))]
# print(len(Nsat3))

	sumsat3r = []
	for i in range(len(Nsat3r)):
		sumsat3r.append(np.sum(Nsat3r[i]))

# 18 <= rmag < 19
	Nsat4r = Nsat[np.where((rmag_LRG >= 18.) & (19. > rmag_LRG))]
# print(len(Nsat4))

	sumsat4r = []
	for i in range(len(Nsat4r)):
		sumsat4r.append(np.sum(Nsat4r[i]))

# 19 <= rmag < 20
	Nsat5r = Nsat[np.where((rmag_LRG >= 19.) & (20. > rmag_LRG))]
# print(len(Nsat5))

	sumsat5r = []
	for i in range(len(Nsat5r)):
		sumsat5r.append(np.sum(Nsat5r[i]))

# 20 <= rmag < 21
	Nsat6r = Nsat[np.where((rmag_LRG >= 20.) & (21. > rmag_LRG))]
# print(len(Nsat6))

	sumsat6r = []
	for i in range(len(Nsat6r)):
		sumsat6r.append(np.sum(Nsat6r[i]))

# rmag >= 21
	Nsat7r = Nsat[np.where(rmag_LRG >= 21.)]
# print(len(Nsat7))

	sumsat7r = []
	for i in range(len(Nsat7r)):
		sumsat7r.append(np.sum(Nsat7r[i]))
    
# Calculate mean for every rmag slice
	mean_sumsat1r = np.mean(sumsat1r)
	print("mean number of satellites at 15 <= rmag < 16:", mean_sumsat1r)
	mean_sumsat2r = np.mean(sumsat2r)
	print("mean number of satellites at 16 <= rmag < 17:", mean_sumsat2r)
	mean_sumsat3r = np.mean(sumsat3r)
	print("mean number of satellites at 17 <= rmag < 18:", mean_sumsat3r)
	mean_sumsat4r = np.mean(sumsat4r)
	print("mean number of satellites at 18 <= rmag < 19:", mean_sumsat4r)
	mean_sumsat5r = np.mean(sumsat5r)
	print("mean number of satellites at 19 <= rmag < 20:", mean_sumsat5r)
	mean_sumsat6r = np.mean(sumsat6r)
	print("mean number of satellites at 20 <= rmag < 21:", mean_sumsat6r)
	mean_sumsat7r = np.mean(sumsat7r)
	print("mean number of satellites at rmag >= 21:", mean_sumsat7r)
    
	print('total number of Nsat arrays:', len(Nsat1r) + len(Nsat2r) + len(Nsat3r) + len(Nsat4r) + len(Nsat5r) + len(Nsat6r) + len(Nsat7r))

	plt.title("Histogram of the Number of Satellite Galaxies at Different LRG rmag")
	plt.hist(sumsat1r, bins=25, alpha=0.5, label='15 <= rmag < 16')
	plt.hist(sumsat2r, bins=25, alpha=0.5, label='16 <= rmag < 17')
	plt.hist(sumsat3r, bins=25, alpha=0.5, label='17 <= rmag < 18')
	plt.hist(sumsat4r, bins=25, alpha=0.5, label='18 <= rmag < 19')
	plt.hist(sumsat5r, bins=25, alpha=0.5, label='19 <= rmag < 20')
	plt.hist(sumsat6r, bins=25, alpha=0.5, label='20 <= rmag < 21')
	plt.hist(sumsat7r, bins=25, alpha=0.5, label='rmag >= 21')
	plt.xlabel(r'$satellite$')
	plt.ylabel(r'$counts$')
	plt.legend(loc='upper right')
	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Satellite Galaxies at Different R-Magnitude Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumsat1r, bins=25, color='lightblue', label='15 <= rmag < 16')
	axarr[0, 0].legend(loc='upper right')
	axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumsat2r, bins=25, color='orange', label='16 <= rmag < 17')
	axarr[0, 1].legend(loc='upper right')
	axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumsat3r, bins=25, color='green', label='17 <= rmag < 18')
	axarr[1, 0].legend(loc='upper right')
	axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumsat4r, bins=25, color='red', label='18 <= rmag < 19')
	axarr[1, 1].legend(loc='upper right')
	axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumsat5r, bins=25, color='purple', label='19 <= rmag < 20')
	axarr[2,0].legend(loc='upper right')
	axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumsat6r, bins=25, color='brown', label='20 <= rmag < 21')
	axarr[2,1].legend(loc='upper right')
	axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumsat7r, bins=25, color='pink', label='rmag >= 21')
	axarr[3,0].legend(loc='upper right')
	axarr[3,0].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='satellites', ylabel='counts')

	plt.show()
    
    
# ----------------------------------------------------------------------------------------


# Plot distribution of Nsat in different LRG gmag bins
def gmag_cut_Nsat(gmag_LRG, Nsat):
	
	import matplotlib.pyplot as plt
	import numpy as np 

	gmag_LRG = np.array(gmag_LRG)

# bins of ~1 mag

# 16 <= gmag < 17
	Nsat1g = Nsat[np.where((gmag_LRG >= 16.) & (17. > gmag_LRG))]
# print(len(Nsat1g))

	sumsat1g = []
	for i in range(len(Nsat1g)):
		sumsat1g.append(np.sum(Nsat1g[i]))

# 17 <= gmag < 18
	Nsat2g = Nsat[np.where((gmag_LRG >= 17.) & (18. > gmag_LRG))]
# print(len(Nsat2g))

	sumsat2g = []
	for i in range(len(Nsat2g)):
		sumsat2g.append(np.sum(Nsat2g[i]))

# 18 <= gmag < 19
	Nsat3g = Nsat[np.where((gmag_LRG >= 18.) & (19. > gmag_LRG))]
# print(len(Nsat3g))

	sumsat3g = []
	for i in range(len(Nsat3g)):
		sumsat3g.append(np.sum(Nsat3g[i]))

# 19 <= gmag < 20
	Nsat4g = Nsat[np.where((gmag_LRG >= 19.) & (20. > gmag_LRG))]
# print(len(Nsat4g))

	sumsat4g = []
	for i in range(len(Nsat4g)):
		sumsat4g.append(np.sum(Nsat4g[i]))

# 20 <= gmag < 21
	Nsat5g = Nsat[np.where((gmag_LRG >= 20.) & (21. > gmag_LRG))]
# print(len(Nsat5g))

	sumsat5g = []
	for i in range(len(Nsat5g)):
		sumsat5g.append(np.sum(Nsat5g[i]))

# 21 <= gmag < 22
	Nsat6g = Nsat[np.where((gmag_LRG >= 21.) & (22. > gmag_LRG))]
# print(len(Nsat6g))

	sumsat6g = []
	for i in range(len(Nsat6g)):
		sumsat6g.append(np.sum(Nsat6g[i]))
    
# 22 <= gmag < 23
	Nsat7g = Nsat[np.where((gmag_LRG >= 22.) & (23. > gmag_LRG))]
# print(len(Nsat7g))

	sumsat7g = []
	for i in range(len(Nsat7g)):
		sumsat7g.append(np.sum(Nsat7g[i]))

# gmag >= 23
	Nsat8g = Nsat[np.where(gmag_LRG >= 23.)]
# print(len(Nsat8))

	sumsat8g = []
	for i in range(len(Nsat8g)):
		sumsat8g.append(np.sum(Nsat8g[i]))  

# Calculate mean for every gmag slice
	mean_sumsat1g = np.mean(sumsat1g)
	print("mean number of satellites at 16 <= gmag < 17:", mean_sumsat1g)
	mean_sumsat2g = np.mean(sumsat2g)
	print("mean number of satellites at 17 <= gmag < 18:", mean_sumsat2g)
	mean_sumsat3g = np.mean(sumsat3g)
	print("mean number of satellites at 18 <= gmag < 19:", mean_sumsat3g)
	mean_sumsat4g = np.mean(sumsat4g)
	print("mean number of satellites at 19 <= gmag < 20:", mean_sumsat4g)
	mean_sumsat5g = np.mean(sumsat5g)
	print("mean number of satellites at 20 <= gmag < 21:", mean_sumsat5g)
	mean_sumsat6g = np.mean(sumsat6g)
	print("mean number of satellites at 21 <= gmag < 22:", mean_sumsat6g)
	mean_sumsat7g = np.mean(sumsat7g)
	print("mean number of satellites at 22 <= gmag < 23:", mean_sumsat7g)
	mean_sumsat8g = np.mean(sumsat8g)
	print("mean number of satellites at gmag >= 23:", mean_sumsat8g)

	print('total number of Nsat arrays:', len(Nsat1g) + len(Nsat2g) + len(Nsat3g) + len(Nsat4g) + len(Nsat5g) + len(Nsat6g) + len(Nsat7g) + len(Nsat8g))

	plt.title("Histogram of the Number of Satellite Galaxies at Different LRG gmag")               # alsdkfjalsdfjka;sldfkjlsdfjlsdjkflsdkjflsdkjf;lsdkjflsdkjfl
	plt.hist(sumsat1g, bins=25, alpha=0.5, label='16 <= gmag < 17')
	plt.hist(sumsat2g, bins=25, alpha=0.5, label='17 <= gmag < 18')
	plt.hist(sumsat3g, bins=25, alpha=0.5, label='18 <= gmag < 19')
	plt.hist(sumsat4g, bins=25, alpha=0.5, label='19 <= gmag < 20')
	plt.hist(sumsat5g, bins=25, alpha=0.5, label='20 <= gmag < 21')
	plt.hist(sumsat6g, bins=25, alpha=0.5, label='21 <= gmag < 22')
	plt.hist(sumsat7g, bins=25, alpha=0.5, label='22 <= gmag < 23')
	plt.hist(sumsat8g, bins=25, alpha=0.5, label='gmag >= 23')
	plt.xlabel(r'$satellite$')
	plt.ylabel(r'$counts$')
	plt.legend(loc='upper right')
	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Satellite Galaxies at Different G-Magnitude Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumsat1g, bins=25, color='lightblue', label='16 <= gmag < 17')
	axarr[0, 0].legend(loc='upper right')
	axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumsat2g, bins=25, color='orange', label='17 <= gmag < 18')
	axarr[0, 1].legend(loc='upper right')
	axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumsat3g, bins=25, color='green', label='18 <= gmag < 19')
	axarr[1, 0].legend(loc='upper right')
	axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumsat4g, bins=25, color='red', label='19 <= gmag < 20')
	axarr[1, 1].legend(loc='upper right')
	axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumsat5g, bins=25, color='purple', label='20 <= gmag < 21')
	axarr[2,0].legend(loc='upper right')
	axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumsat6g, bins=25, color='brown', label='21 <= gmag < 22')
	axarr[2,1].legend(loc='upper right')
	axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumsat7g, bins=25, color='pink', label='22 <= gmag < 23')
	axarr[3,0].legend(loc='upper right')
	axarr[3,0].axvline(linewidth=1, color='black')

	axarr[3,1].hist(sumsat8g, bins=25, color='grey', label='gmag >= 23')
	axarr[3,1].legend(loc='upper right')
	axarr[3,1].axvline(linewidth=1, color='black')

# f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='satellites', ylabel='counts')

	plt.show()
    

# ----------------------------------------------------------------------------------------

    
# Plot distribution of Nsat in different LRG zmag bins
def zmag_cut_Nsat(zmag_LRG, Nsat):
	
	import matplotlib.pyplot as plt
	import numpy as np 

	zmag_LRG = np.array(zmag_LRG)

# bins of ~1 mag

# 14 <= zmag < 15
	Nsat1z = Nsat[np.where((zmag_LRG >= 14.) & (15. > zmag_LRG))]
# print(len(Nsat1g))

	sumsat1z = []
	for i in range(len(Nsat1z)):
		sumsat1z.append(np.sum(Nsat1z[i]))

# 15 <= zmag < 16
	Nsat2z = Nsat[np.where((zmag_LRG >= 15.) & (16. > zmag_LRG))]
# print(len(Nsat2g))

	sumsat2z = []
	for i in range(len(Nsat2z)):
		sumsat2z.append(np.sum(Nsat2z[i]))

# 16 <= zmag < 17
	Nsat3z = Nsat[np.where((zmag_LRG >= 16.) & (17. > zmag_LRG))]
# print(len(Nsat3g))

	sumsat3z = []
	for i in range(len(Nsat3z)):
		sumsat3z.append(np.sum(Nsat3z[i]))

# 17 <= zmag < 18
	Nsat4z = Nsat[np.where((zmag_LRG >= 17.) & (18. > zmag_LRG))]
# print(len(Nsat4z))

	sumsat4z = []
	for i in range(len(Nsat4z)):
		sumsat4z.append(np.sum(Nsat4z[i]))

# 18 <= zmag < 19
	Nsat5z = Nsat[np.where((zmag_LRG >= 18.) & (19. > zmag_LRG))]
# print(len(Nsat5g))

	sumsat5z = []
	for i in range(len(Nsat5z)):
		sumsat5z.append(np.sum(Nsat5z[i]))

# 19 <= zmag < 20
	Nsat6z = Nsat[np.where((zmag_LRG >= 19.) & (20. > zmag_LRG))]
# print(len(Nsat6g))

	sumsat6z = []
	for i in range(len(Nsat6z)):
		sumsat6z.append(np.sum(Nsat6z[i]))

# zmag < 20
	Nsat7z = Nsat[np.where(zmag_LRG >= 20.)]
# print(len(Nsat8))

	sumsat7z = []
	for i in range(len(Nsat7z)):
		sumsat7z.append(np.sum(Nsat7z[i]))

# Calculate mean for every zmag slice
	mean_sumsat1z = np.mean(sumsat1z)
	print("mean number of satellites at 14 <= zmag < 15:", mean_sumsat1z)
	mean_sumsat2z = np.mean(sumsat2z)
	print("mean number of satellites at 15 <= zmag < 16:", mean_sumsat2z)
	mean_sumsat3z = np.mean(sumsat3z)
	print("mean number of satellites at 16 <= zmag < 17:", mean_sumsat3z)
	mean_sumsat4z = np.mean(sumsat4z)
	print("mean number of satellites at 17 <= zmag < 18:", mean_sumsat4z)
	mean_sumsat5z = np.mean(sumsat5z)
	print("mean number of satellites at 18 <= zmag < 19:", mean_sumsat5z)
	mean_sumsat6z = np.mean(sumsat6z)
	print("mean number of satellites at 19 <= zmag < 20:", mean_sumsat6z)
	mean_sumsat7z = np.mean(sumsat7z)
	print("mean number of satellites at zmag >= 20:", mean_sumsat7z)

	print('total number of Nsat arrays:', len(Nsat1z) + len(Nsat2z) + len(Nsat3z) + len(Nsat4z) + len(Nsat5z) + len(Nsat6z) + len(Nsat7z))

	plt.title("Histogram of the Number of Satellite Galaxies at Different LRG gmag")               # alsdkfjalsdfjka;sldfkjlsdfjlsdjkflsdkjflsdkjf;lsdkjflsdkjfl
	plt.hist(sumsat1z, bins=25, alpha=0.5, label='14 <= zmag < 15')
	plt.hist(sumsat2z, bins=25, alpha=0.5, label='15 <= zmag < 16')
	plt.hist(sumsat3z, bins=25, alpha=0.5, label='16 <= zmag < 17')
	plt.hist(sumsat4z, bins=25, alpha=0.5, label='17 <= zmag < 18')
	plt.hist(sumsat5z, bins=25, alpha=0.5, label='18 <= zmag < 19')
	plt.hist(sumsat6z, bins=25, alpha=0.5, label='19 <= zmag < 20')
	plt.hist(sumsat7z, bins=25, alpha=0.5, label='zmag >= 20')
	plt.xlabel(r'$satellite$')
	plt.ylabel(r'$counts$')
	plt.legend(loc='upper right')
	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Satellite Galaxies at Different Z-Magnitude Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumsat1z, bins=25, color='lightblue', label='14 <= zmag < 15')
	axarr[0, 0].legend(loc='upper right')
	axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumsat2z, bins=25, color='orange', label='15 <= zmag < 16')
	axarr[0, 1].legend(loc='upper right')
	axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumsat3z, bins=25, color='green', label='16 <= zmag < 17')
	axarr[1, 0].legend(loc='upper right')
	axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumsat4z, bins=25, color='red', label='17 <= zmag < 18')
	axarr[1, 1].legend(loc='upper right')
	axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumsat5z, bins=25, color='purple', label='18 <= zmag < 19')
	axarr[2,0].legend(loc='upper right')
	axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumsat6z, bins=25, color='brown', label='19 <= zmag < 20')
	axarr[2,1].legend(loc='upper right')
	axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumsat7z, bins=25, color='pink', label='zmag >= 20')
	axarr[3,0].legend(loc='upper right')
	axarr[3,0].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='satellites', ylabel='counts')

	plt.show()



# Plot distribution of nn in different LRG rmag bins
def rmag_cut_near(rmag_LRG, near):
	
	import matplotlib.pyplot as plt
	import numpy as np 

	rmag_LRG = np.array(rmag_LRG)
	near = np.array(near)

# bins of ~1 mag

# 15 <= rmag < 16
	near1r = near[np.where((rmag_LRG >= 15.) & (16. > rmag_LRG))]
# print(len(Nsat1))

	sumnear1r = []
	for i in range(len(near1r)):
		sumnear1r.append(np.sum(near1r[i]))

# 16 <= rmag < 17
	near2r = near[np.where((rmag_LRG >= 16.) & (17. > rmag_LRG))]
# print(len(Nsat2))

	sumnear2r = []
	for i in range(len(near2r)):
		sumnear2r.append(np.sum(near2r[i]))

# 17 <= rmag < 18
	near3r = near[np.where((rmag_LRG >= 17.) & (18. > rmag_LRG))]
# print(len(Nsat3))

	sumnear3r = []
	for i in range(len(near3r)):
		sumnear3r.append(np.sum(near3r[i]))

# 18 <= rmag < 19
	near4r = near[np.where((rmag_LRG >= 18.) & (19. > rmag_LRG))]
# print(len(Nsat4))

	sumnear4r = []
	for i in range(len(near4r)):
		sumnear4r.append(np.sum(near4r[i]))

# 19 <= rmag < 20
	near5r = near[np.where((rmag_LRG >= 19.) & (20. > rmag_LRG))]
# print(len(Nsat5))

	sumnear5r = []
	for i in range(len(near5r)):
		sumnear5r.append(np.sum(near5r[i]))

# 20 <= rmag < 21
	near6r = near[np.where((rmag_LRG >= 20.) & (21. > rmag_LRG))]
# print(len(Nsat6))

	sumnear6r = []
	for i in range(len(near6r)):
		sumnear6r.append(np.sum(near6r[i]))

# rmag >= 21
	near7r = near[np.where(rmag_LRG >= 21.)]
# print(len(Nsat7))

	sumnear7r = []
	for i in range(len(near7r)):
		sumnear7r.append(np.sum(near7r[i]))
    
# Calculate mean for every rmag slice
	mean_sumnear1r = np.mean(sumnear1r)
	print("mean number of near neighbors at 15 <= rmag < 16:", mean_sumnear1r)
	mean_sumnear2r = np.mean(sumnear2r)
	print("mean number of near neighbors at 16 <= rmag < 17:", mean_sumnear2r)
	mean_sumnear3r = np.mean(sumnear3r)
	print("mean number of near neighbors at 17 <= rmag < 18:", mean_sumnear3r)
	mean_sumnear4r = np.mean(sumnear4r)
	print("mean number of near neighbors at 18 <= rmag < 19:", mean_sumnear4r)
	mean_sumnear5r = np.mean(sumnear5r)
	print("mean number of near neighbors at 19 <= rmag < 20:", mean_sumnear5r)
	mean_sumnear6r = np.mean(sumnear6r)
	print("mean number of near neighbors at 20 <= rmag < 21:", mean_sumnear6r)
	mean_sumnear7r = np.mean(sumnear7r)
	print("mean number of near neighbors at rmag >= 21:", mean_sumnear7r)
                                 
	print('total number of near arrays:', len(near1r) + len(near2r) + len(near3r) + len(near4r) + len(near5r) + len(near6r) + len(near7r))
    
	# plt.title("Histogram of the Number of Near Neighbors at Different LRG rmag")
# 	plt.hist(sumnear1r, bins=25, alpha=0.5, label='15 <= rmag < 16')
# 	plt.hist(sumnear2r, bins=25, alpha=0.5, label='16 <= rmag < 17')
# 	plt.hist(sumnear3r, bins=25, alpha=0.5, label='17 <= rmag < 18')
# 	plt.hist(sumnear4r, bins=25, alpha=0.5, label='18 <= rmag < 19')
# 	plt.hist(sumnear5r, bins=25, alpha=0.5, label='19 <= rmag < 20')
# 	plt.hist(sumnear6r, bins=25, alpha=0.5, label='20 <= rmag < 21')
# 	plt.hist(sumnear7r, bins=25, alpha=0.5, label='rmag >= 21')
# 	plt.xlabel(r'$Near$ $Neighbors$')
# 	plt.ylabel(r'$counts$')
# 	plt.legend(loc='upper right')
# 	plt.show()
    
	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Near Neighbor Galaxies at Different R-Magnitude Slices", fontsize=15, y=0.9)    # alsdjf'lasdjf'lasdfj'alsdfja;lsdfj;alsdfjdsf'
    
	axarr[0, 0].hist(sumnear1r, bins=25, color='lightblue', label='15 <= rmag < 16')
	axarr[0, 0].legend(loc='upper right')
# axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumnear2r, bins=25, color='orange', label='16 <= rmag < 17')
	axarr[0, 1].legend(loc='upper right')
# axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumnear3r, bins=25, color='green', label='17 <= rmag < 18')
	axarr[1, 0].legend(loc='upper right')
# axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumnear4r, bins=25, color='red', label='18 <= rmag < 19')
	axarr[1, 1].legend(loc='upper right')
# axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumnear5r, bins=25, color='purple', label='19 <= rmag < 20')
	axarr[2,0].legend(loc='upper right')
# axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumnear6r, bins=25, color='brown', label='20 <= rmag < 21')
	axarr[2,1].legend(loc='upper right')
# axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumnear7r, bins=25, color='pink', label='rmag >= 21')
	axarr[3,0].legend(loc='upper right')
# axarr[3,0].axvline(linewidth=1, color='black')

# axarr[3,1].hist(sumsat7z, bins=25, color='pink', label='z >= 0.7')
# axarr[3,1].legend(loc='upper right')
# axarr[3,1].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='near neighbors', ylabel='counts')

	plt.show()


# ----------------------------------------------------------------------------------------


# Plot distribution of nn in different LRG gmag bins
def gmag_cut_near(gmag_LRG, near):
	
	import matplotlib.pyplot as plt
	import numpy as np 

	gmag_LRG = np.array(gmag_LRG)
	near = np.array(near)

# bins of ~1 mag

# 16 <= gmag < 17
	near1g = near[np.where((gmag_LRG >= 16.) & (17. > gmag_LRG))]
# print(len(Nsat1))

	sumnear1g = []
	for i in range(len(near1g)):
		sumnear1g.append(np.sum(near1g[i]))

# 17 <= gmag < 18
	near2g = near[np.where((gmag_LRG >= 17.) & (18. > gmag_LRG))]
# print(len(near2g))

	sumnear2g = []
	for i in range(len(near2g)):
		sumnear2g.append(np.sum(near2g[i]))
# print(len(sumnear2g))
    
# 18 <= gmag < 19
	near3g = near[np.where((gmag_LRG >= 18.) & (19. > gmag_LRG))]
# print(len(Nsat3))

	sumnear3g = []
	for i in range(len(near3g)):
		sumnear3g.append(np.sum(near3g[i]))

# 19 <= gmag < 20
	near4g = near[np.where((gmag_LRG >= 19.) & (20. > gmag_LRG))]
# print(len(Nsat4))

	sumnear4g = []
	for i in range(len(near4g)):
		sumnear4g.append(np.sum(near4g[i]))

# 20 <= gmag < 21
	near5g = near[np.where((gmag_LRG >= 20.) & (21. > gmag_LRG))]
# print(len(Nsat5))

	sumnear5g = []
	for i in range(len(near5g)):
		sumnear5g.append(np.sum(near5g[i]))

# 21 <= gmag < 22
	near6g = near[np.where((gmag_LRG >= 21.) & (22. > gmag_LRG))]
# print(len(Nsat6))

	sumnear6g = []
	for i in range(len(near6g)):
		sumnear6g.append(np.sum(near6g[i]))
    
# 22 <= gmag < 23
	near7g = near[np.where((gmag_LRG >= 22.) & (23. > gmag_LRG))]
# print(len(Nsat7))

	sumnear7g = []
	for i in range(len(near7g)):
		sumnear7g.append(np.sum(near7g[i]))

# gmag >= 23
	near8g = near[np.where(gmag_LRG >= 23.)]
# print(len(Nsat8))

	sumnear8g = []
	for i in range(len(near8g)):
		sumnear8g.append(np.sum(near8g[i]))
    
# Calculate mean for every gmag slice
	mean_sumnear1g = np.mean(sumnear1g)
	print("mean number of near neighbors at 16 <= gmag < 17:", mean_sumnear1g)
	mean_sumnear2g = np.mean(sumnear2g)
	print("mean number of near neighbors at 17 <= gmag < 18:", mean_sumnear2g)
	mean_sumnear3g = np.mean(sumnear3g)
	print("mean number of near neighbors at 18 <= gmag < 19:", mean_sumnear3g)
	mean_sumnear4g = np.mean(sumnear4g)
	print("mean number of near neighbors at 19 <= gmag < 20:", mean_sumnear4g)
	mean_sumnear5g = np.mean(sumnear5g)
	print("mean number of near neighbors at 20 <= gmag < 21:", mean_sumnear5g)
	mean_sumnear6g = np.mean(sumnear6g)
	print("mean number of near neighbors at 21 <= gmag < 22:", mean_sumnear6g)
	mean_sumnear7g = np.mean(sumnear7g)
	print("mean number of near neighbors at 22 <= gmag < 23:", mean_sumnear7g)
	mean_sumnear8g = np.mean(sumnear8g)
	print("mean number of near neighbors at gmag >= 23:", mean_sumnear8g)
    
	print('total number of near arrays:', len(near1g) + len(near2g) + len(near3g) + len(near4g) + len(near5g) + len(near6g) + len(near7g) + len(near8g))

	# plt.title("Histogram of the Number of Near Neighbors at Different LRG gmag")
# 	plt.hist(sumnear1g, bins=25, alpha=0.5, label='16 <= gmag < 17')
# 	plt.hist(sumnear2g, bins=25, alpha=0.5, label='17 <= gmag < 18')
# 	plt.hist(sumnear3g, bins=25, alpha=0.5, label='18 <= gmag < 19')
# 	plt.hist(sumnear4g, bins=25, alpha=0.5, label='19 <= gmag < 20')
# 	plt.hist(sumnear5g, bins=25, alpha=0.5, label='20 <= gmag < 21')
# 	plt.hist(sumnear6g, bins=25, alpha=0.5, label='21 <= gmag < 22')
# 	plt.hist(sumnear7g, bins=25, alpha=0.5, label='22 <= gmag < 23')
# 	plt.hist(sumnear8g, bins=25, alpha=0.5, label='gmag >= 23')
# 	plt.xlabel(r'$Near$ $Neighbors$')
# 	plt.ylabel(r'$counts$')
# 	plt.legend(loc='upper right')
# 	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Near Neighbor Galaxies at Different G-Magnitude Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumnear1g, bins=25, color='lightblue', label='17 <= gmag < 18')
	axarr[0, 0].legend(loc='upper left')
# axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumnear2g, bins=25, color='orange', label='18 <= gmag < 19')
	axarr[0, 1].legend(loc='upper right')
# axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumnear3g, bins=25, color='green', label='19 <= gmag < 20')
	axarr[1, 0].legend(loc='upper right')
# axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumnear4g, bins=25, color='red', label='20 <= gmag < 21')
	axarr[1, 1].legend(loc='upper right')
# axarr[1, 1].axvline(linewidth=1, color='black')
        
	axarr[2,0].hist(sumnear5g, bins=25, color='purple', label='21 <= gmag < 22')
	axarr[2,0].legend(loc='upper right')
# axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumnear6g, bins=25, color='brown', label='22 <= gmag < 23')
	axarr[2,1].legend(loc='upper right')
# axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumnear7g, bins=25, color='pink', label='23 <= gmag < 24')
	axarr[3,0].legend(loc='upper right')
# axarr[3,0].axvline(linewidth=1, color='black')

	axarr[3,1].hist(sumnear8g, bins=25, color='grey', label='gmag >= 24')
	axarr[3,1].legend(loc='upper right')
# axarr[3,1].axvline(linewidth=1, color='black')

# f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='near neighbors', ylabel='counts')
	
	plt.show()
    
    
# ----------------------------------------------------------------------------------------


# Plot distribution of nn in different LRG zmag bins
def zmag_cut_near(zmag_LRG, near):
	
	import matplotlib.pylab as plt 
	import numpy as np 

	zmag_LRG = np.array(zmag_LRG)
	near = np.array(near)

    # bins of ~1 mag

# 14 <= zmag < 15
	near1z = near[np.where((zmag_LRG >= 14.) & (15. > zmag_LRG))]
# print(len(Nsat1))

	sumnear1z = []
	for i in range(len(near1z)):
		sumnear1z.append(np.sum(near1z[i]))

# 15 <= zmag < 16
	near2z = near[np.where((zmag_LRG >= 15.) & (16. > zmag_LRG))]
# print(len(near2g))

	sumnear2z = []
	for i in range(len(near2z)):
		sumnear2z.append(np.sum(near2z[i]))
# print(len(sumnear2g))
    
# 16 <= zmag < 17
	near3z = near[np.where((zmag_LRG >= 16.) & (17. > zmag_LRG))]
# print(len(Nsat3))

	sumnear3z = []
	for i in range(len(near3z)):
		sumnear3z.append(np.sum(near3z[i]))

# 17 <= zmag < 18
	near4z = near[np.where((zmag_LRG >= 17.) & (18. > zmag_LRG))]
# print(len(Nsat4))

	sumnear4z = []
	for i in range(len(near4z)):
		sumnear4z.append(np.sum(near4z[i]))

# 18 <= zmag < 19
	near5z = near[np.where((zmag_LRG >= 18.) & (19. > zmag_LRG))]
# print(len(Nsat5))

	sumnear5z = []
	for i in range(len(near5z)):
		sumnear5z.append(np.sum(near5z[i]))
    
# 19 <= zmag < 20
	near6z = near[np.where((zmag_LRG >= 19.) & (20. > zmag_LRG))]
# print(len(Nsat5))

	sumnear6z = []
	for i in range(len(near6z)):
		sumnear6z.append(np.sum(near6z[i]))

# zmag >= 20
	near7z = near[np.where(zmag_LRG >= 20.)]
# print(len(Nsat8))

	sumnear7z = []
	for i in range(len(near7z)):
		sumnear7z.append(np.sum(near7z[i]))
    
# Calculate mean for every zmag slice
	mean_sumnear1z = np.mean(sumnear1z)
	print("mean number of near neighbors at 14 <= zmag < 15:", mean_sumnear1z)
	mean_sumnear2z = np.mean(sumnear2z)
	print("mean number of near neighbors at 15 <= zmag < 16:", mean_sumnear2z)
	mean_sumnear3z = np.mean(sumnear3z)
	print("mean number of near neighbors at 16 <= zmag < 17:", mean_sumnear3z)
	mean_sumnear4z = np.mean(sumnear4z)
	print("mean number of near neighbors at 17 <= zmag < 18:", mean_sumnear4z)
	mean_sumnear5z = np.mean(sumnear5z)
	print("mean number of near neighbors at 18 <= zmag < 19:", mean_sumnear5z)
	mean_sumnear6z = np.mean(sumnear6z)
	print("mean number of near neighbors at 19 <= zmag < 20:", mean_sumnear6z)
	mean_sumnear7z = np.mean(sumnear7z)
	print("mean number of near neighbors at zmag >= 20:", mean_sumnear7z)
    
	print('total number of near arrays:', len(near1z) + len(near2z) + len(near3z) + len(near4z) + len(near5z) + len(near6z) + len(near7z)) 

	# plt.title("Histogram of the Number of Near Neighbors at Different LRG gmag")
# 	plt.hist(sumnear1z, bins=25, alpha=0.5, label='14 <= zmag < 15')
# 	plt.hist(sumnear2z, bins=25, alpha=0.5, label='15 <= zmag < 16')
# 	plt.hist(sumnear3z, bins=25, alpha=0.5, label='16 <= zmag < 17')
# 	plt.hist(sumnear4z, bins=25, alpha=0.5, label='17 <= zmag < 18')
# 	plt.hist(sumnear5z, bins=25, alpha=0.5, label='18 <= zmag < 19')
# 	plt.hist(sumnear6z, bins=25, alpha=0.5, label='19 <= zmag < 20')
# 	plt.hist(sumnear7z, bins=25, alpha=0.5, label='gmag >= 20')
# 	plt.xlabel(r'$Near$ $Neighbors$')
# 	plt.ylabel(r'$counts$')
# 	plt.legend(loc='upper right')
# 	plt.show()

	f, axarr = plt.subplots(4, 2, figsize=(15,15))
	f.suptitle("Histogram of the Number of Near Neighbor Galaxies at Different Z-Magnitude Slices", fontsize=15, y=0.9)

	axarr[0, 0].hist(sumnear1z, bins=25, color='lightblue', label='14 <= zmag < 15')
	axarr[0, 0].legend(loc='upper left')
# axarr[0, 0].axvline(linewidth=1, color='black')

	axarr[0, 1].hist(sumnear2z, bins=25, color='orange', label='15 <= zmag < 16')
	axarr[0, 1].legend(loc='upper right')
# axarr[0, 1].axvline(linewidth=1, color='black')

	axarr[1, 0].hist(sumnear3z, bins=25, color='green', label='16 <= zmag < 17')
	axarr[1, 0].legend(loc='upper right')
# axarr[1, 0].axvline(linewidth=1, color='black')

	axarr[1, 1].hist(sumnear4z, bins=25, color='red', label='17 <= zmag < 18')
	axarr[1, 1].legend(loc='upper right')
# axarr[1, 1].axvline(linewidth=1, color='black')

	axarr[2,0].hist(sumnear5z, bins=25, color='purple', label='18 <= zmag < 18')
	axarr[2,0].legend(loc='upper right')
# axarr[2,0].axvline(linewidth=1, color='black')

	axarr[2,1].hist(sumnear6z, bins=25, color='brown', label='19 <= zmag < 20')
	axarr[2,1].legend(loc='upper right')
# axarr[2,1].axvline(linewidth=1, color='black')

	axarr[3,0].hist(sumnear7z, bins=25, color='pink', label='zmag >= 20')
	axarr[3,0].legend(loc='upper right')
# axarr[3,0].axvline(linewidth=1, color='black')

	f.delaxes(axarr[3,1])
# Fine-tune figure; make subplots farther from each other.
	f.subplots_adjust(hspace=0.5)
# f.subplots_adjust(wspace=0.5)

	for ax in axarr.flat:
		ax.set(xlabel='near neighbors', ylabel='counts')

	plt.show()
    

# ----------------------------------------------------------------------------------------
    
    
# Use HEALPix/Healpy to plot of sources over the sky

# add flip='geo' to make it look like the RA-Dec plot
def healpix(ra_BKG, dec_BKG, ra_LRG, dec_LRG, gmag_BKG, rmag_BKG, zmag_BKG):

	import matplotlib.pyplot as plt
	import numpy as np 
	import healpy as hp
	
#     %matplotlib inline

# ra_LRG_mag_cut = ra_LRG[np.where((gmag_LRG > 24.) & (rmag_LRG > 24.) & (zmag_LRG > 24.))]
# dec_LRG_mag_cut = dec_LRG[np.where((gmag_LRG > 24.) & (rmag_LRG > 24.) & (zmag_LRG > 24.))]

	ra_BKG_mag_cut = ra_BKG[np.where((gmag_BKG < 21.) & (rmag_BKG < 21.) & (zmag_BKG < 21.))]
	dec_BKG_mag_cut = dec_BKG[np.where((gmag_BKG < 21.) & (rmag_BKG < 21.) & (zmag_BKG < 21.))]

	theta, phi = np.radians(90-dec_LRG), np.radians(ra_LRG)
	nside = 512
	npixel= hp.nside2npix(nside)
	m = hp.ang2pix(nside, theta, phi)
	map_ = np.bincount(m, minlength=npixel)
	hp.gnomview(map_,rot=(-116.5,9.),xsize=225,flip='geo', title="Only LRGs in EDR Area")

	ra = np.concatenate([ra_LRG, ra_BKG_mag_cut])
	dec = np.concatenate([dec_LRG, dec_BKG_mag_cut])

	theta, phi = np.radians(90-dec), np.radians(ra)
	nside = 512
	npixel= hp.nside2npix(nside)
	m = hp.ang2pix(nside, theta, phi)
	map_ = np.bincount(m, minlength=npixel)
	hp.gnomview(map_,rot=(-116.5,9.),xsize=225, flip='geo', title="All Sources in EDR Area")
	plt.show()      


# ----------------------------------------------------------------------------------------


def magHist(gmag_BKG, rmag_BKG, zmag_BKG):

	import matplotlib.pyplot as plt
	import numpy as np 

# 	figsize=(8, 6)
	plt.rcParams["figure.figsize"] = [10,8]
	plt.title("Manitude Distribution", fontsize = 15)
	plt.hist(gmag_BKG, bins=50, color='green', alpha=0.5, label='gmag')
	plt.hist(rmag_BKG, bins=50, color='red', alpha=0.5, label="rmag")
	plt.hist(zmag_BKG, bins=50, color='lightblue', alpha=0.5, label='zmag')
	plt.legend(loc="upper right", fontsize = 15)
	plt.xlabel(r'$magnitude$', fontsize = 15)
	plt.xticks(fontsize=10)
	plt.ylabel(r'$counts$', fontsize = 15)
	plt.yticks(fontsize=10)
	plt.show()


# ----------------------------------------------------------------------------------------
	
	
def zHist(z_LRG):

	import matplotlib.pyplot as plt
	import numpy as np

	plt.rcParams["figure.figsize"] = [10, 8]
	plt.title("Redshift Distribution", fontsize = 15)
	plt.xlabel(r'$redshift', fontsize=15)
	plt.ylabel(r'$counts$', fontsize=15)
	plt.hist(z_LRG, bins=50)
	plt.show()
    
    
# ----------------------------------------------------------------------------------------


# Function that calculates the bootstrapped median, gives the 68% confidence interval
# and plots the histogram with CI marked as vertical lines

def boot_med_plot(niter, confint, boot_func, array):

	from astropy.stats import bootstrap
	from astropy.utils import NumpyRNGContext
	import matplotlib.pyplot as plt
	import numpy as np 


# niter = 1000
	with NumpyRNGContext(1):
		bootmed = bootstrap(np.asarray(array), bootnum=niter, bootfunc=boot_func)
#         bootmean = bootstrap(np.asarray(sumsat), bootnum=niter, bootfunc=np.mean)

# Compute confidence interval of median

#     confint = 0.68
	sortmed = sorted(bootmed)
	lowind = int(round((1 - confint)/2*niter, 2))
	highind = int(round((1-((1 - confint)/2))*niter, 2))

	plt.rcParams["figure.figsize"] = [10, 8]
	plt.title("Histogram of Bootstrapped Median")
	plt.hist(bootmed, bins=50, color='indigo', alpha=0.5)
# plt.hist(bootmean, bins=25, color='violet', alpha=0.5)
	plt.axvline(x=sortmed[lowind])
	plt.axvline(x=sortmed[highind])
	plt.xlabel(r'$bootstrap median$', fontsize=15)
	plt.ylabel(r'$counts$', fontsize=15)
	plt.show()

	print("The median of Nsat:", np.median(array))
	print("The median of bootmed:", np.median(bootmed))
#     print("The mean of Nsat:", np.mean(sumsat))
	print("Low 68% confidence interval:", sortmed[lowind])
	print("High 68% confidence interval:", sortmed[highind])
	
	lowconf = sortmed[lowind]
	hiconf = sortmed[highind]
	med = np.median(bootmed)
	
	return(med, lowconf, hiconf)
    