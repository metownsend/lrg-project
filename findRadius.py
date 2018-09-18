

def findRadius(distance_inner, distance_outer)


    import numpy as np
    from sklearn.neighbors import KDTree

    # Distance from which we are looking for satellites around the LRGs
    # distance_r2 = 5.  # in Mpc
    distance_inner_kpc = distance_inner * 10. ** 3.  # in kpc

    # Distance from which we are looking for satellites around the LRGs
    distance_outer = 15.  # in Mpc
    distance_outer_kpc = distance_outer * 10. ** 3.  # in kpc

    dist_outer = []
    for i in range(len(kpc_DA)):
        dist_outer.append((distance_outer_kpc / kpc_DA[i]) * 1. / 3600.)

    # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list_LRG = list(zip(ra_LRG, dec_LRG))  # Fake LRG sources
    zip_list_BKG = list(zip(ra_BKG, dec_BKG))  # Fake EDR sources

    # Creates a tree of EDR sources
    gal_tree_outer = KDTree(zip_list_BKG)

    # find indices of sources
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_outer = gal_tree_outer.query_radius(zip_list_LRG, r=dist_outer)

    # returns a list of sources and their distances from the LRG within radius dist_outer
    nn_outer, d_outer = gal_tree_outer.query_radius(zip_list_LRG, r=dist_outer, return_distance=True)

    # print(d_r3)
    # print(len(d_r3))

    # d_outer is given in degree, since that is the input. This converts degree to arcsecond, kpc, and Mpc
    d_outer_arcsec = d_outer * 3600.
    # print(d_r3_arcsec)
    d_outer_kpc = d_outer * 3600. * kpc_DA
    # print(d_r3_kpc)
    d_outer_Mpc = d_outer_kpc * (1. / 1000.)
    # print(d_r3_Mpc)

    # print(ind_r3[2][np.where((d_r3_Mpc[2] >= 3.) & (d_r3_Mpc[2] < 3.5))])

    bkg = []
    for i in range(len(ind_outer)):
        # Creates a zero array if there are no near neighbors
        if len(ind_outer[i][np.where((d_outer_Mpc[i] >= 5.) & (d_outer_Mpc[i] < 5.5))]) == 0:
            temp = np.zeros((len(xedges) - 1, len(yedges) - 1))
            bkg.append(temp)
        # Creates a 2D histogram for satellite galaxies
        else:
            temp, x_notuse, y_notuse = np.histogram2d(
                rmag_BKG[ind_outer[i][np.where((d_outer_Mpc[i] >= 5.) & (d_outer_Mpc[i] < 5.5))]],
                color_BKG[ind_outer[i][np.where((d_outer_Mpc[i] >= 5.) & (d_outer_Mpc[i] < 5.5))]], bins=(xedges, yedges),
                normed=False)
            bkg1.append(temp1)


    # Area of an annulus A = pi * (outer radius**2 - inner radius**2)

    A1 = np.pi * (r2 ** 2. - r1 ** 2.)

    sigma = []
    for i in range(len(bkg)):
        sigma.append(bkg[i] / A)

    sum_sigma = np.sum(sigma)
    # print(sum_sigma1)

    error = np.sqrt(sum_sigma)/sum_sigma