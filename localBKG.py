# Calculating local surface density and interlopers

def localBKG_and_interlopers(distance_kpc, distance_r2, distance_r3, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_LRG, rmag_BKG, color_LRG, color_BKG, xedges, yedges):
# def localBKG_and_interlopers(distance_r2, distance_r3, kpc_DA, DCMR_Mpc ,ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_LRG, rmag_BKG, color_LRG, color_BKG, xedges, yedges):

    # ra_LRG,dec_LRG are RA/Dec at center of search radius (in this case, ra_LRG and dec_LRG)
    # ra_BKG,dec_BKG are RA/Dec of every other source
    # distance_kpc is the distance from ra_LRG,dec_LRG we are looking for satellites.

    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    import matplotlib.pylab as plt
    import matplotlib.lines as mlines
    from matplotlib.legend import Legend
    from pythonds.basic.stack import Stack
    from sklearn.neighbors import KDTree
    import healpy as hp

    # Distance from which we are looking for satellites around the LRGs
    # distance_r2 = 2. # in Mpc
    # distance_r2_kpc = distance_r2 * 10.**3. # in kpc
    #
    # dist_r2 = []
    # for i in range(len(kpc_DA)):
    #     dist_r2.append((distance_r2_kpc / kpc_DA[i]) * 1./3600.)

    # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # Fake LRG sources
    # zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # Fake EDR sources
    ra = np.concatenate([ra_LRG, ra_BKG])
    dec = np.concatenate([dec_LRG, dec_BKG])
    zip_list = list(zip(ra, dec))  # All sources

    # Creates a tree of background sources
    gal_tree = KDTree(zip_list)

    # returns a list of background sources that are within some radius r of an LRG
    nn_r2 = gal_tree.query_radius(zip_list_LRG,r=distance_r2,count_only=True)

    # find indices of near neighbors
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_r2 = gal_tree.query_radius(zip_list_LRG,r=distance_r2)

    ind2list_r2 = []

    ind2list_r2 = ind_r2.tolist()

    index_r2 = []
    for i in range(len(ind2list_r2)):
        index_r2.append(ind2list_r2[i].tolist())


    # Distance from which we are looking for satellites around the LRGs
#     distance_r3 = 10. # in Mpc
#     distance_r3_kpc = distance_r3 * 10.**3. # in kpc

    # dist_r3 = []
    # for i in range(len(kpc_DA)):
    #     dist_r3.append((distance_r3_kpc / kpc_DA[i]) * 1./3600.)

    # # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    # zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # Fake LRG sources
    # # zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # Fake EDR sources

    # Creates a tree of EDR sources
    # gal_tree_r3 = KDTree(zip_list)

    # returns a list of EDR sources that are within some radius r of an LRG
    nn_r3 = gal_tree.query_radius(zip_list_LRG, r=distance_r3, count_only=True)

    # find indices of near neighbors
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_r3 = gal_tree.query_radius(zip_list_LRG, r=distance_r3)

    ind2list_r3 = []

    ind2list_r3 = ind_r3.tolist()

    index_r3 = []
    for i in range(len(ind2list_r3)):
        index_r3.append(ind2list_r3[i].tolist())


#     -------------------------------
#
#
# for i in range(len(index)):
#     index[i] = [x for x in index[i] if x != i]
#
# near = []
# rmag = np.concatenate([rmag_LRG, rmag_BKG])
# color = np.concatenate([color_LRG, color_BKG])
#
# # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
# for i in range(len(index)):
#     if len(index[i]) == 0:
#         hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
#         near.append(hist2d)
#     else:
#
#         hist2d, x_notuse, y_notuse = np.histogram2d(rmag[index[i]], color[index[i]], bins=(xedges, yedges),
#                                                     normed=False)
#         near.append(hist2d)
#
#     -------------------------------


    annulus_ind = []

    for i in range(len(index_r3)):
        index_r3[i] = [x for x in index_r3[i] if x not in index_r2[i]]
        # annulus_ind.append(l3)
        # print(index_r2[i])
        # print(index_r3[i])
        # print('-----')

    # print('len index_r3:', len(np.asarray(index_r3)))

    # print(len(new_ind[0]))

    # new_ind = np.asarray(annulus_ind)

    # number of galaxies in the annulus
    # numbkg = []
    # for i in range(len(new_ind)):
    #     numbkg.append(len(new_ind[i]))

    numbkg = []
    for i in range(len(index_r3)):
        numbkg.append(len(index_r3[i]))

    med_local = np.median(numbkg)
#     print("median number of local galaxies is", med_local)

    area_annulus = np.pi * (distance_r3**2. - distance_r2**2.)


    # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
    # LOCAL_BKG is the list of 2D arrays of survey galaxies as a function of color and magnitude
    local_bkg = []
    rmag = np.concatenate([rmag_LRG, rmag_BKG])
    color = np.concatenate([color_LRG, color_BKG])

    # Converts search radius to kpc
    # dist_r2 = []
    # for i in range(len(kpc_DA)):
    #     dist_r2.append((dist_r2 * 3600.) * kpc_DA[i]) # degree to arcsec to kpc
    #
    #
    # dist_r3 = []
    # for i in range(len(kpc_DA)):
    #     dist_r3.append((dist_r3 * 3600.) * kpc_DA[i])  # degree to arcsec to kpc
    #
    # area = []
    # for i in range(len(dist_r2)):
    #     area.append(np.pi * dist[i]**2.)


    # for i in range(len(new_ind)):
    #     # Creates a zero array if there are no near neighbors
    #     if len(new_ind[i]) == 0:
    #         hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
    #         local_bkg.append(hist2d)
    #     # Creates a 2D histogram for satellite galaxies
    #     else:
    #         hist2d, x_notuse, y_notuse = np.histogram2d(rmag[new_ind[i]], color[new_ind[i]], bins=(xedges, yedges), normed=False)
    #         local_bkg.append(hist2d)

    for i in range(len(index_r3)):
        # Creates a zero array if there are no near neighbors
        if len(index_r3[i]) == 0:
            hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
            local_bkg.append(hist2d)
        # Creates a 2D histogram for satellite galaxies
        else:
            hist2d, x_notuse, y_notuse = np.histogram2d(rmag[index_r3[i]], color[index_r3[i]], bins=(xedges, yedges), normed=False)
            local_bkg.append(hist2d)

    local_bkg = np.array(local_bkg)
        
    # local_bkg = np.asarray(local_bkg)
    # print('shape local_bkg: ', local_bkg.shape)
    # print("type local_bkg: ", type(local_bkg))
    # print('type local_bkg[0]: ', type(local_bkg[0]))

    # sigma is the surface density around individual LRGs. Each
    sigma = []
    for i in range(len(local_bkg)):
        t = local_bkg[i] / area_annulus
        # print(i)
        # print(area_annulus)
        sigma.append(t) # units: number / square degree
        # print(sigma[i])


    # sigma = np.asarray(sigma)
    # print(sigma.shape)

    print(len(sigma))

    omega = []
    for i in range(len(kpc_DA)):
        omega.append(((np.pi * distance_kpc ** 2.) / (kpc_DA[i]) ** 2.) * (1./3600.)**2.)  # in square degree

    # omega = np.asarray(omega)
    # print(omega.shape)
    # print(len(omega))
                 
    Nbkg = []
    for i in range(len(omega)):
        Nbkg.append((sigma[i] * omega[i]))

    Nbkg = np.asarray(Nbkg)


    return (numbkg, med_local, sigma, omega, Nbkg)

# return(numbkg, med_local, sigma, omega, Nbkg, r2, dist_r2, r3, dist_r3)  # returns number of galaxies in annulus for every LRG,
                                                # median number of galaxies in the annulus over all LRGs,
                                                # the surface density sigma for each LRG, 
                                                # the solid angle omega for each LRG, 
                                                # and calculated array of background galaxies for each LRG
