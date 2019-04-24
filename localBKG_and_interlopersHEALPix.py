# A function to calculate the local background around LRGs and calculate the expected number of interloper galaxies

def localBKG_and_interlopersHEALPix(nside, inner_dist, outer_dist, ra_LRG, dec_LRG, pixnums, mag, color, xedges, yedges, distance_kpc, kpc_DA):

    # inner_dist == inner radius of annulus used to define the background
    # outer_dist == outer radius of annulus used to define the background
    # ra_LRG, dec_LRG == ra and dec for only LRG sources
    # pixnums == list length of number of sources that holds what pixel the source exists in; same order at original
        # catalog (same as ra/dec/mag/etc.)
    # mag == list of magnitudes for all sources
    # color == relevant color for all sources
    # xedges, yedges = edges of bin for CMD
    # distance_kpc == search radius in kpc
    # kpc_DA == scale factor used to convert a physical distance to an angular distance dependent on redshift
        # calculated by cosmoCalc

    # HAS DEPENDENCIES ON PANDAS, ASTROPY_HEALPIX, NUMPY, HEALPY, and ASTROPY

    from astropy import units as u
    from astropy_healpix import HEALPix
    import pandas as pd
    import numpy as np
    import healpy as hpy

    hp = HEALPix(nside=nside, order='nested')

    # Run cone_search twice - once for inner radius and once for outer radius of annulus to use for calculating
    # local background

    inner_pix = []
    outer_pix = []
    for i in range(len(ra_LRG)):
        inner_pix.append(hp.cone_search_lonlat(ra_LRG[i] * u.deg, dec_LRG[i] * u.deg, radius=inner_dist * u.deg))
        outer_pix.append(hp.cone_search_lonlat(ra_LRG[i] * u.deg, dec_LRG[i] * u.deg, radius=outer_dist * u.deg))

    # find only pixels that are in the annulus

    annulus_pix = []
    for j in range(len(inner_pix)):
        annulus_pix.append(np.setdiff1d(outer_pix[j], inner_pix[j]))

    # matches pixel indices to sources; gives list of indices for all sources in annulus

    pixnum_index = pd.Index(pixnums)
    a = []
    indices = []

    for j in range(len(annulus_pix)):
        good_keys = np.unique(pixnum_index.intersection(annulus_pix[j]))
        for i in range(len(good_keys)):
            #         print(j)
            temp = (pixnum_index.get_loc(good_keys[i]))
            #         print(temp)
            a.append(np.where(temp == True))
        #         print("end")
        array = np.concatenate(a, axis=None)
        sort_array = np.sort(array)
        indices.append(sort_array)
        a = []

    # calculate area of annulus

    pix_area = hpy.nside2pixarea(nside, degrees=True)  # will return pixel area in square degree

    annulus_area = []
    for i in range(len(annulus_pix)):
        annulus_area.append(len(annulus_pix[i]) * pix_area)

    # make CMD of local background

    localBKG = []

    # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
    for i in range(len(indices)):
        if len(indices[i]) == 0:
            hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
            localBKG.append(hist2d)
        else:
            hist2d, x_notuse, y_notuse = np.histogram2d(mag[indices[i]], color[indices[i]], bins=(xedges, yedges),
                                                        normed=False)
            localBKG.append(hist2d)

    # calculate surface density (sigma) by dividing CMD by area of annulus

    sigma = []
    for i in range(len(annulus_area)):
        sigma.append(localBKG[i] / annulus_area[i])

    # calculate solid angle (omega) enclosed by search radius

    omega = []
    for i in range(len(kpc_DA)):
        omega.append(((np.pi * distance_kpc ** 2.) / (kpc_DA[i]) ** 2.) * (1. / 3600.) ** 2.)  # in square degree

    # calculate the number of expected interlopers

    Nbkg = []
    for i in range(len(omega)):
        Nbkg.append((sigma[i] * omega[i]))

    return (Nbkg, indices,omega, localBKG)