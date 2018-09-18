

def findRadius(radius_inner, radius_outer, kpc_DA)


    import numpy as np
    from sklearn.neighbors import KDTree

    # Distance from which we are looking for satellites around the LRGs
    # distance_r2 = 5.  # in Mpc
    radius_inner_kpc = radius_inner * 10. ** 3.  # in kpc

    radius_inner_arcsec = []
    for i in range(len(kpc_DA)):
        radius_inner_arcsec.append(radius_inner_kpc / kpc_DA[i]) # only using for bkg array so only need dist_inner in arcsec

    # Distance from which we are looking for satellites around the LRGs
    radius_outer = 15.  # in Mpc
    radius_outer_kpc = radius_outer * 10. ** 3.  # in kpc

    radius_outer = []
    for i in range(len(kpc_DA)):
        radius_outer.append((radius_outer_kpc / kpc_DA[i]) * 1. / 3600.)

    # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list_LRG = list(zip(ra_LRG, dec_LRG))  # Fake LRG sources
    zip_list_BKG = list(zip(ra_BKG, dec_BKG))  # Fake EDR sources

    # Creates a tree of EDR sources
    gal_tree_outer = KDTree(zip_list_BKG)

    # find indices of sources
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_outer = gal_tree_outer.query_radius(zip_list_LRG, r=radius_outer)

    # returns a list of sources and their distances from the LRG within radius dist_outer
    nn_outer, dist_outer = gal_tree_outer.query_radius(zip_list_LRG, r=radius_outer, return_distance=True)

    # print(d_r3)
    # print(len(d_r3))

    # d_outer is given in degree, since that is the input. This converts degree to arcsecond, kpc, and Mpc
    dist_outer_arcsec = dist_outer * 3600.
    # print(d_r3_arcsec)
    dist_outer_kpc = dist_outer * 3600. * kpc_DA
    # print(d_r3_kpc)
    dist_outer_Mpc = dist_outer_kpc * (1. / 1000.)
    # print(d_r3_Mpc)

    # print(ind_r3[2][np.where((d_r3_Mpc[2] >= 3.) & (d_r3_Mpc[2] < 3.5))])

    bkg_kpc = []
    for i in range(len(ind_outer)):
        # Creates a zero array if there are no near neighbors
        if len(ind_outer[i][np.where((dist_outer_kpc[i] >= radius_inner_kpc) & (dist_outer_kpc[i] < radius_outer_kpc))]) == 0:
            temp_kpc = np.zeros((len(xedges) - 1, len(yedges) - 1))
            bkg_kpc.append(temp_kpc)
        # Creates a 2D histogram for satellite galaxies
        else:
            temp_kpc, x_notuse, y_notuse = np.histogram2d(
                rmag_BKG[ind_outer[i][np.where((dist_outer_kpc[i] >= radius_inner_kpc) & (dist_outer_kpc[i] < radius_outer_kpc))]],
                color_BKG[ind_outer[i][np.where((dist_outer_kpc[i] >= radius_inner_kpc) & (dist_outer_kpc[i] < radius_outer_kpc))]], bins=(xedges, yedges),
                normed=False)
            bkg_kpc.append(temp_kpc)

    bkg_arcsec = []
    for i in range(len(ind_outer)):
        # Creates a zero array if there are no near neighbors
        if len(ind_outer[i][np.where((dist_outer_arcsec[i] >= radius_inner_arcsec[i]) & (dist_outer_arcsec[i] < radius_outer_arcsec[i]))]) == 0:
            temp_kpc = np.zeros((len(xedges) - 1, len(yedges) - 1))
            bkg_kpc.append(temp_kpc)
        # Creates a 2D histogram for satellite galaxies
        else:
            temp_kpc, x_notuse, y_notuse = np.histogram2d(rmag_BKG[ind_outer[i][np.where((dist_outer_arcsec[i] >= radius_inner_arcsec[i]) & (dist_outer_arcsec[i] < radius_outer_arcsec[i]))]],
                color_BKG[ind_outer[i][np.where((dist_outer_arcsec[i] >= radius_inner_arcsec[i]) & (dist_outer_arcsec[i] < radius_outer_arcsec[i]))]], bins=(xedges, yedges),
                normed=False)
            bkg_kpc.append(temp_kpc)


    # Area of an annulus A = pi * (outer radius**2 - inner radius**2)

    # This area calculation only works for physical radius. Look at localBKG.py for how to get area in arcsec
    A_kpc = np.pi * (radius_outer** 2. - radius_inner** 2.)

    sigma_kpc = []
    for i in range(len(bkg_kpc)):
        sigma_kpc.append(bkg[i] / A_kpc)

    sum_sigma_kpc = np.sum(sigma_kpc)
    # print(sum_sigma1)

    error_kpc = np.sqrt(sum_sigma)/sum_sigma

    return(sum_sigma_kpc, error_kpc, dist_outer_arcsec, dist_outer_kpc, radius_inner_kpc, radius_outer_kpc,