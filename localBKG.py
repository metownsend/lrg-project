# Calculating local surface density and interlopers

def localBKG_and_interlopers(distance_kpc, distance_r2, distance_r3, kpc_DA, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, color_BKG, xedges, yedges):
    
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
#     distance_r2 = 2. # in Mpc
    distance_r2_kpc = distance_r2 * 10.**3. # in kpc

    dist_r2 = []
    for i in range(len(kpc_DA)):
        dist_r2.append((distance_r2_kpc / kpc_DA[i]) * 1./3600.) 

    # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # Fake LRG sources
    zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # Fake EDR sources

    # Creates a tree of background sources
    gal_tree_r2 = KDTree(zip_list_BKG)

    # returns a list of background sources that are within some radius r of an LRG
    nn_r2 = gal_tree_r2.query_radius(zip_list_LRG,r=dist_r2,count_only=True)

    # find indices of near neighbors
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_r2 = gal_tree_r2.query_radius(zip_list_LRG,r=dist_r2)

    # Distance from which we are looking for satellites around the LRGs
#     distance_r3 = 10. # in Mpc
    distance_r3_kpc = distance_r3 * 10.**3. # in kpc

    dist_r3 = []
    for i in range(len(kpc_DA)):
        dist_r3.append((distance_r3_kpc / kpc_DA[i]) * 1./3600.) 

    # Creates a list of ordered pairs; zips ra and dec together so they can be fed into KDTree
    zip_list_LRG = list(zip(ra_LRG, dec_LRG)) # Fake LRG sources
    zip_list_BKG = list(zip(ra_BKG, dec_BKG)) # Fake EDR sources

    # Creates a tree of EDR sources
    gal_tree_r3 = KDTree(zip_list_BKG)

    # returns a list of EDR sources that are within some radius r of an LRG
    nn_r3 = gal_tree_r3.query_radius(zip_list_LRG,r=dist_r3,count_only=True)

    # find indices of near neighbors
    # creates a list of arrays that include the indices of satellite galaxies per LRG. In general, some or all of these
    # arrays could be empty
    ind_r3 = gal_tree_r3.query_radius(zip_list_LRG,r=dist_r3)

    annulus_ind = []

    for i in range(len(ind_r3)):
        l3 = [x for x in ind_r3[i] if x not in ind_r2[i]]
        annulus_ind.append(l3)

    # print(len(new_ind[0]))

    new_ind = np.asarray(annulus_ind)

    # number of galaxies in the annulus
    numbkg = []
    for i in range(len(new_ind)):
        numbkg.append(len(new_ind[i]))

    med_local = np.median(numbkg)
#     print("median number of local galaxies is", med_local)

    # inner radius
    r2 = []
    for i in range(len(kpc_DA)):
        r2.append((distance_r2_kpc / kpc_DA[i])) 
    
    # outer radius
    r3 = []
    for i in range(len(kpc_DA)):
        r3.append((distance_r3_kpc / kpc_DA[i])) 

    # area of inner circle
    area_r2 = []
    for i in range(len(dist_r2)):
        area_r2.append((np.pi * r2[i] ** 2.))
    
    # area of outer circle
    area_r3 = []
    for i in range(len(dist_r3)):
        area_r3.append((np.pi * r3[i] ** 2.))
    
    # area of annulus
    area_annulus = []
    for i in range(len(dist_r3)):
        area_annulus.append(area_r3[i] - area_r2[i])


    # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
    # LOCAL_BKG is the list of 2D arrays of survey galaxies as a function of color and magnitude
    local_bkg = []

    for i in range(len(new_ind)):
        # Creates a zero array if there are no near neighbors
        if len(new_ind[i]) == 0:
            hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
            local_bkg.append(hist2d)
        # Creates a 2D histogram for satellite galaxies
        else:
            hist2d, x_notuse, y_notuse = np.histogram2d(rmag_BKG[new_ind[i]], color_BKG[new_ind[i]], bins=(xedges, yedges), normed=False)
            local_bkg.append(hist2d)
        
    # sigma is the surface density around individual LRGs. Each
    sigma = []
    for i in range(len(area_annulus)):
        sigma.append(local_bkg[i] / area_annulus[i])
                 
    omega = []
    for i in range(len(kpc_DA)):
        omega.append((np.pi * distance_kpc ** 2.) / (kpc_DA[i]) ** 2.)  # in square arcsec
                 
    Nbkg = []
    for i in range(len(omega)):
        Nbkg.append((sigma[i] * omega[i]))
                 
    Nbkg = np.asarray(Nbkg)
    
    return(numbkg, med_local, sigma, omega, Nbkg, dist_r2, dist_r3, zip_list_LRG, zip_list_BKG)  # returns number of galaxies in annulus for every LRG, 
                                                # median number of galaxies in the annulus over all LRGs,
                                                # the surface density sigma for each LRG, 
                                                # the solid angle omega for each LRG, 
                                                # and calculated arrray of background galaxies for each LRG
