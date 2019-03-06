# function to calculate near neighbors

# Counting NEAR NEIGHBORS (nn) using KDTree
# Result is an array of the number of near neighbors each LRG has

# def nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, color_BKG, xedges, yedges):
#
#     from astropy.io import fits
#     from astropy.table import Table
#     import numpy as np
#     import matplotlib.pylab as plt
#     import matplotlib.lines as mlines
#     from matplotlib.legend import Legend
#     from pythonds.basic.stack import Stack
#     from sklearn.neighbors import KDTree
#     import healpy as hp
#
# 	# distance == radius from LRG in which I look for near neighbors in Mpc
#
#     # Distance from which we are looking for satellites around the LRGs
#     distance_kpc = distance * 10.**3. # in kpc
#
#     dist = []
#     for i in range(len(kpc_DA)):
#         dist.append((distance_kpc / kpc_DA[i]) * 1./3600.)
#
#     # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
#     zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # LRG sources
#     zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # EDR sources
#
#     # Creates a tree of EDR sources
#     gal_tree = KDTree(zip_list_BKG)
#
#     # returns a list of EDR sources that are within some radius r of an LRG
#     nn = gal_tree.query_radius(zip_list_LRG,r=dist,count_only=True)
#
#     # find indices of near neighbors
#     # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
#     # arrays could be empty
#     ind = gal_tree.query_radius(zip_list_LRG,r=dist)
#
#     # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
#     # NEAR is the list of 2D arrays of satellite galaxies as a funciton of color and magnitude
#     near = []
#
#     for i in range(len(ind)):
#         # Creates a zero array if there are no near neighbors
#         if len(ind[i]) == 0:
#             hist2d = np.zeros((len(xedges)-1,len(yedges)-1))
#             near.append(hist2d)
#         # Creates a 2D histogram for satellite galaxies
#         else:
#             hist2d, x_notuse, y_notuse = np.histogram2d(rmag_BKG[ind[i]], color_BKG[ind[i]], bins=(xedges, yedges), normed=False)
#             near.append(hist2d)
#
#     return(distance_kpc, near, gal_tree)



def nearNeighbor(distance, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_LRG, rmag_BKG, color_LRG, color_BKG, xedges, yedges):

    # distance == radius from LRG in which I look for near neighbors in Mpc

    import numpy as np
    from sklearn.neighbors import KDTree

    distance_kpc = distance * 10. ** 3.  # in kpc

    dist = []
    for i in range(len(kpc_DA)):
        dist.append((distance_kpc / kpc_DA[i]) * 1. / 3600.) # in degree

     # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list0 = list(zip(ra_LRG, dec_LRG))  # LRG sources
    ra = np.concatenate([ra_LRG, ra_BKG])
    dec = np.concatenate([dec_LRG, dec_BKG])
    zip_list1 = list(zip(ra, dec))  # All sources


    # Creates a tree of EDR sources
    gal_tree = KDTree(zip_list1)

    # returns a list of EDR sources that are within some radius r of an LRG
    nn1 = gal_tree.query_radius(zip_list0, r=dist, count_only=True)
    # nn2 = gal_tree.query_radius(zip_list0, r=dist2, count_only=True)


    # find indices of near neighbors
    ind = gal_tree.query_radius(zip_list0, r=dist)
    

    ind2list = []
    ind2list = ind.tolist()

    index = []
    for i in range(len(ind2list)):
        index.append(ind2list[i].tolist())

    # Array that gives actual number of near neighbors for every LRG
    num = []

    for i in range(len(ind)):
        num.append(len(ind[i]))

    for i in range(len(index)):
        index[i] = [x for x in index[i] if x != i]

    near = []
    rmag = np.concatenate([rmag_LRG, rmag_BKG])
    color = np.concatenate([color_LRG, color_BKG])

    # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
    for i in range(len(index)):
        if len(index[i]) == 0:
            hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
            near.append(hist2d)
        else:

            hist2d, x_notuse, y_notuse = np.histogram2d(rmag[index[i]], color[index[i]], bins=(xedges, yedges),
                                                        normed=False)
            near.append(hist2d)

    return (distance_kpc, near, gal_tree, dist)
    # return (near, gal_tree)