# A function to calculate near neighbors using HEALPix pixels

def nearNeighborsHEALPix(distance_kpc, kpc_DA, ra_LRG, dec_LRG, nside, pixnums, mag, color, xedges, yedges):

    # distance_kpc == search radius in kpc
    # kpc_DA == scale factor used to convert a physical distance to an angular distance dependent on redshift
        # calculated by cosmoCalc
    # ra_LRG, dec_LRG == ra and dec for only LRG sources
    # nside == resolution of the HEALPix map; defines the number of divisions along the side of a base-resolution pixel
    # pixnums == list length of number of sources that holds what pixel the source exists in; same order at original
        # catalog (same as ra/dec/mag/etc.)
    # mag == list of magnitudes for all sources
    # color == relevant color for all sources
    # xedges, yedges = edges of bin for CMD

    # HAS DEPENDENCIES ON PANDAS, ASTROPY_HEALPIX, and ASTROPY

    from astropy import units as u
    from astropy_healpix import HEALPix
    import pandas as pd

    # converts physical distance to degree
    dist = []
    for i in range(len(kpc_DA)):
        dist.append((distance_kpc / kpc_DA[i]) * 1. / 3600.)  # in degree

    hp = HEALPix(nside=nside, order='nested')

    search_pix = []
    for i in range(len(ra_LRG)):
        search_pix.append(hp.cone_search_lonlat(ra_LRG[i] * u.deg, dec_LRG[i] * u.deg, radius=dist[i] * u.deg))

    pixnum_index = pd.Index(pixnums)
    a = []
    indices = []

    for j in range(len(search_pix)):
        good_keys = np.unique(pixnum_index.intersection(search_pix[j]))
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

    near = []

    # Creates one list of number of near neighbors for every LRG (number of lists = number of LRGs)
    for i in range(len(indices)):
        if len(indices[i]) == 0:
            hist2d = np.zeros((len(xedges) - 1, len(yedges) - 1))
            near.append(hist2d)
        else:
            hist2d, x_notuse, y_notuse = np.histogram2d(mag[indices[i]], color[indices[i]], bins=(xedges, yedges),
                                                        normed=False)
            near.append(hist2d)

    return (indices, near) # returns list of indices of near neighbors for every LRG and a near neighbor CMD for every LRG
