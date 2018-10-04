
# A function to determine how far from the LRG we should go to get appropriate background

def bestBKG(a, b, dist_outer, ind_outer, radius_outer_kpc, kpc_DA, xedges, yedges, rmag_survey, color_survey):

    # a must be greater than b by at least 1

    import numpy as np

    # dist_outer is given in degree, since that is the input. This converts degree to arcsecond and kpc
    dist_outer_arcsec = []
    temp1 = []
    for i in range(len(kpc_DA)):
        for j in range(len(dist_outer[i])):
            x = np.float64(dist_outer[i][j] / kpc_DA[i])
            #         print(type(x))
            temp1.append(x)
        dist_outer_arcsec.append(temp1)
        temp1 = []

    dist_outer_kpc = []
    temp2 = []
    for i in range(len(kpc_DA)):
        for j in range(len(dist_outer[i])):
            y = np.float64(dist_outer[i][j] * 3600. * kpc_DA[i])
            temp2.append(y)
        dist_outer_kpc.append(temp2)
        temp2 = []

    # defines inner and outer radius for search annulus
    inner_radius = (radius_outer_kpc / a) * b
    outer_radius = (radius_outer_kpc / a) * (b + 1.)

    # creates a list of arrays of indices in dist_outer_kpc that include sources within search annulus
    dist_index = []
    for i in range(len(dist_outer_kpc)):
        index = \
        np.where((np.asarray(dist_outer_kpc[i]) > inner_radius) & (np.asarray(dist_outer_kpc[i]) < outer_radius))[0]
        dist_index.append(index)
        index = []

    # matches dist_index indices to actual indices of sources in their ra/dec arrays
    ind = []
    for i in range(len(ind_outer)):
        t = ind_outer[i][dist_index[i]]
        ind.append(t)

    # creates a CMD of only sources in search annulus
    bkg_kpc = []
    for i in range(len(ind)):
        # Creates a zero array if there are no near neighbors
        if len(ind[i]) == 0:
            temp_kpc1 = np.zeros((len(xedges) - 1, len(yedges) - 1))
            bkg_kpc.append(temp_kpc1)
        #         print("1")
        # Creates a 2D histogram for satellite galaxies
        else:
            temp_kpc2, x_notuse, y_notuse = np.histogram2d(rmag_survey[ind[i]], color_survey[ind[i]],
                                                           bins=(xedges, yedges), normed=False)
            bkg_kpc.append(temp_kpc2)
    #         print("2")

    # This area calculation only works for physical radius. Look at localBKG.py for how to get area in arcsec
    area_kpc = np.pi * ((outer_radius) ** 2. - (inner_radius) ** 2.)
    # print(area_kpc)

    # Calculate the surface density sigma for each LRG
    sigma_kpc = []
    for i in range(len(bkg_kpc)):
        sigma_kpc.append(bkg_kpc[i] / area_kpc)

    # sigma_arcsec = []
    # for i in range(len(bkg_arcsec)):
    #     sigma_arcsec.append(bkg_arcsec[i] / area_annulus[i])

    sum_sigma_kpc = np.sum(sigma_kpc)
    # print(sum_sigma_kpc)
    # sum_sigma_arcsec = np.sum(sigma_arcsec)

    error_kpc = np.sqrt(sum_sigma_kpc) / sum_sigma_kpc
    # print(error_kpc)
    # error_arcsec = np.sqrt(sum_sigma_arcsec) / sum_sigma_arcsec

    return (sum_sigma_kpc, outer_radius, inner_radius, bkg_kpc, error_kpc)