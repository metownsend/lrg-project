# A function to read in data from the Legacy Surveys and SDSS

def readData(SpecObj_data, SDSS_data, DECaLS_data):

    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    import matplotlib.pylab as plt
    import matplotlib.lines as mlines
    from matplotlib.legend import Legend
    from pythonds.basic.stack import Stack
    from sklearn.neighbors import KDTree
    import healpy as hp


    # Put data in arrays

    # Read in data from SDSS file

    # Redshift of galaxies according to sdss
    z = []
    z = SDSS_data.field('Z')

    # Class of object
    gal_class = []
    gal_class = SDSS_data.field('CLASS')

    # What survey the data is from
    survey = []
    survey = SDSS_data.field('SURVEY')

    # SPECPRIMARY; set to 1 for primary observation of object, 0 otherwise
    spec = []
    spec = SDSS_data.field('SPECPRIMARY')

    # Bitmask of spectroscopic warning values; need set to 0
    zwarn_noqso = []
    zwarn_noqso = SDSS_data.field('ZWARNING_NOQSO')

    # Spectroscopic classification for certain redshift?
    class_noqso = []
    class_noqso = SDSS_data.field('CLASS_NOQSO')

    # Array for LRG targets
    targets = []
    targets = SDSS_data.field('BOSS_TARGET1')

    # Section of code to find LRG targets

    def divideBy2(decNumber):

    # from pythonds.basic.stack import Stack
    # import numpy as np

        np.vectorize(decNumber)
        remstack = Stack()

        if decNumber == 0: return "0"

        while decNumber > 0:
            rem = decNumber % 2
            remstack.push(rem)
            decNumber = decNumber // 2

        binString = ""
        while not remstack.isEmpty():
            binString = binString + str(remstack.pop())

        return binString


    # Function to find LOWZ targets
    divideBy2Vec = np.vectorize(divideBy2)

    a = divideBy2Vec(targets) # gives binary in string form

    b = []
    c = []

    for i in range(len(a)):
        b.append(list((a[i])))
        b[i].reverse()

    # print(b)

    lrg = []

    # Finds flags for BOSS LOWZ and CMASS sample
    for i in range(len(b)):
        try:
            if (b[i][0] == '1') or (b[i][1] == '1'):
                lrg.append(int(1))
            else:
                lrg.append(int(0))
        except IndexError:
            pass
            lrg.append(int(0))

    lrg = np.array(lrg)
    print('length of sdss lrg array: ', len(lrg))
    print('length of lrg only array:', len(lrg[np.where(lrg == 1)]))

    # ------------------------------------------------------------------------------------------------------------

    # Read in data from SDSS row matched DECaLS file

    # Object ID from survey file; value -1 for non-matches
    objid_MATCHED = []
    objid_MATCHED = SpecObj_data.field('OBJID')

    # Add bridkid
    brickid_MATCHED = []
    brickid_MATCHED = SpecObj_data.field('BRICKID')

    # Add brickname
    brickname_MATCHED = []
    brickname_MATCHED = SpecObj_data.field('BRICKNAME')

    # Only galaxies included
    gal_type_MATCHED = []
    gal_type_MATCHED = SpecObj_data.field('TYPE')

    # RA
    ra_MATCHED = []
    ra_MATCHED = SpecObj_data.field('RA')

    # Dec
    dec_MATCHED = []
    dec_MATCHED = SpecObj_data.field('DEC')

    # flux_g
    gflux_MATCHED = []
    gflux_MATCHED = SpecObj_data.field('FLUX_G')

    # flux_r
    rflux_MATCHED = []
    rflux_MATCHED = SpecObj_data.field('FLUX_R')

    # flux_z
    zflux_MATCHED = []
    zflux_MATCHED = SpecObj_data.field('FLUX_Z')

    # nobs == number of images that contribute to the central pixel
    # nobs_g
    gobs_MATCHED = []
    gobs_MATCHED = SpecObj_data.field('NOBS_G')

    # nobs_r
    robs_MATCHED = []
    robs_MATCHED = SpecObj_data.field('NOBS_R')

    # nobs_z
    zobs_MATCHED = []
    zobs_MATCHED = SpecObj_data.field('NOBS_Z')

    # Create a unique identifier by combinding BRICKID and OBJID

    id_MATCHED = []

    for i in range(len(objid_MATCHED)):
        if (objid_MATCHED[i] == -1):
            id_MATCHED.append(-1)
        else:
            temp1 = str(brickid_MATCHED[i]) + str(objid_MATCHED[i])
            id_MATCHED.append(temp1)

    print('length of row matched targets in SDSS and DECaLS: ', len(id_MATCHED))
    id_MATCHED = np.array(id_MATCHED)
    # ------------------------------------------------------------------------------------------------------------

    # Read in data from DECaLS bricks

    # Object ID from survey file; value -1 for non-matches
    objid_ALL = []
    objid_ALL = DECaLS_data.field('OBJID')
    # print(len(objid_ALL))

    # Add bridkid
    brickid_ALL = []
    brickid_ALL = DECaLS_data.field('BRICKID')

    # Add brickname
    brickname_ALL = []
    brickname_ALL = DECaLS_data.field('BRICKNAME')

    # Only galaxies included
    gal_type_ALL = []
    gal_type_ALL = DECaLS_data.field('TYPE')

    # RA
    ra_ALL = []
    ra_ALL = DECaLS_data.field('RA')

    # Dec
    dec_ALL = []
    dec_ALL = DECaLS_data.field('DEC')

    # flux_g
    gflux_ALL = []
    gflux_ALL = DECaLS_data.field('FLUX_G')

    # flux_r
    rflux_ALL = []
    rflux_ALL = DECaLS_data.field('FLUX_R')

    # flux_z
    zflux_ALL = []
    zflux_ALL = DECaLS_data.field('FLUX_Z')

    # nobs == number of images that contribute to the central pixel
    # nobs_g
    gobs_ALL = []
    gobs_ALL = DECaLS_data.field('NOBS_G')

    # nobs_r
    robs_ALL = []
    robs_ALL = DECaLS_data.field('NOBS_R')

    # nobs_z
    zobs_ALL = []
    zobs_ALL = DECaLS_data.field('NOBS_Z')

    id_ALL = []

    for i in range(len(objid_ALL)):
        temp2 = str(brickid_ALL[i]) + str(objid_ALL[i])
        id_ALL.append(temp2)

    print('length of DECaLS targets in brick: ', len(id_ALL))

    id_ALL = np.array(id_ALL)

    print('length of id_ALL: ', len(id_ALL))

    # ------------------------------------------------------------------------------------------------------------

    # Make cuts to separate LRGs and background galaxies

    # Selects only LRGs (with other cuts)
    LRG_cut = ((gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (zobs_MATCHED >= 3.) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) & (zflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))
    print(type(LRG_cut))
    # id_LRG = []
    # print(type(id_LRG))
    id_LRG = id_MATCHED[np.where(LRG_cut)]
    print('length of id_MATCHED with LRG_cut (id_LRG):', len(id_LRG))

    idcut = []

    # This creates a list that is the length of id_ALL that matches LRGs from the DECaLS/SDSS file to the DECaLS file
    # Use id_cut_noLRG == 0 to get galaxy sources that are NOT identified LRGs
    # For use in narrowing down DECaLS-only file (ie 'ALL')
    for i in range(len(id_ALL)):
        if any(id_LRG == id_ALL[i]):
            idcut.append(1)
        else:
            idcut.append(0)

    idcut = np.array(idcut)
    print('length of idcut:', len(idcut))
    print('length of idcut = 1 (is an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 1)]))
    print('length of idcut = 0 (is not an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 0)]))

    z_lrg = []
    ra_lrg = []
    dec_lrg = []
    for i in range(len(id_ALL)):
        if (idcut[i] == 1):
            z_lrg.append(z[np.where(id_MATCHED == id_ALL[i])])
            ra_lrg.append(ra_MATCHED[np.where(id_MATCHED == id_ALL[i])])
            dec_lrg.append(dec_MATCHED[np.where(id_MATCHED == id_ALL[i])])

    print('length of z_lrg:', len(z_lrg))
    z_lrg = np.array(z_lrg)
    z_LRG = np.concatenate(z_lrg)
    ra_lrg = np.array(ra_lrg)
    ra_LRG = np.concatenate(ra_lrg)
    dec_lrg = np.array(dec_lrg)
    dec_LRG = np.concatenate(dec_lrg)

    # LRG_cut = ((id_cut_LRG == 1) & (gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))
    # & (brickid_LRG == brickid_ALL)
    # print(len(LOWZ_cut))

    # Cut out LRGs
    no_LRG_cut = ((idcut == 0) & (gobs_ALL >= 3.) & (robs_ALL >= 3.) & (zobs_ALL >= 3.) & (gflux_ALL > 0.) & (rflux_ALL > 0.) & (zflux_ALL > 0.) & ((gal_type_ALL == 'SIMP') | (gal_type_ALL == "DEV") | (gal_type_ALL == "EXP") | (gal_type_ALL == "REX")) & (ra_ALL >= 241) & (ra_ALL <= 246) & (dec_ALL >= 6.5) & (dec_ALL <= 11.5))

    # Flux cuts

    # Flux in g for only LRGs
    gflux_LRG = gflux_ALL[np.where(idcut == 1)]

    # Flux in r for only LRGs
    rflux_LRG = rflux_ALL[np.where(idcut == 1)]

    # Flux in g for only LRGs
    zflux_LRG = zflux_ALL[np.where(idcut == 1)]

    # Flux in g for all galaxies in DECaLS
    gflux_BKG = gflux_ALL[np.where(no_LRG_cut)]

    # Flux in r for all galaxies in DECaLS
    rflux_BKG = rflux_ALL[np.where(no_LRG_cut)]

    # Flux in z for all galaxies in DECaLS
    zflux_BKG = zflux_ALL[np.where(no_LRG_cut)]


    # Obs cuts

    # Number of images in g for only LRGs
    gobs_LRG = gobs_ALL[np.where(idcut == 1)]

    # Number of images in r for only LRGs
    robs_LRG = robs_ALL[np.where(idcut == 1)]

    # Number of images in g for only LRGs
    zobs_LRG = zobs_ALL[np.where(idcut == 1)]

    # Number of images in g for all galaxies in DECaLS
    gobs_BKG = gobs_ALL[np.where(no_LRG_cut)]

    # Number of images in r for all galaxies in DECaLS
    robs_BKG = robs_ALL[np.where(no_LRG_cut)]

    # Number of images in z for all galaxies in DECaLS
    zobs_BKG = zobs_ALL[np.where(no_LRG_cut)]

    gmag_LRG = 22.5 - 2.5 * np.log10(gflux_LRG)
    rmag_LRG = 22.5 - 2.5 * np.log10(rflux_LRG)
    zmag_LRG = 22.5 - 2.5 * np.log10(zflux_LRG)

    color_LRG = gmag_LRG - rmag_LRG

    gmag_BKG = 22.5 - 2.5 * np.log10(gflux_BKG)
    rmag_BKG = 22.5 - 2.5 * np.log10(rflux_BKG)
    zmag_BKG = 22.5 - 2.5 * np.log10(zflux_BKG)

    color_BKG = gmag_BKG - rmag_BKG

    # plt.hist(gmag_BKG, bins=50, color='green', alpha=0.5)
    # plt.hist(rmag_BKG, bins=50, color='red', alpha=0.5)
    # plt.hist(zmag_BKG, bins=50, color='lightblue', alpha=0.5)
    # plt.show()
    #
    # plt.hist(z_LRG, bins=50)
    # plt.show()

    ra_BKG = ra_ALL[np.where(no_LRG_cut)]
    dec_BKG = dec_ALL[np.where(no_LRG_cut)]

    # print("end readData")

    return ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG



# from astropy.io import fits
# from astropy.table import Table
# import numpy as np
# import matplotlib.pylab as plt
# import matplotlib.lines as mlines
# from matplotlib.legend import Legend
# from pythonds.basic.stack import Stack
# from math import *
# from sklearn.neighbors import KDTree
# import healpy as hp
# from lrg_plot_functions import *
# from lrg_sum_functions import *
# from cosmo_Calc import *
# from divideByTwo import *
# from readData import *
# from nearNeighbors import *
# from localBKG import *
#
#
# hdulist = fits.open('/Users/mtownsend/anaconda/Data/survey-dr5-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
#                                                                  # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
# hdulist2 = fits.open('/Users/mtownsend/anaconda/Data/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
# hdulist3 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p005-250p010.fits') # this is one sweep file of the DECaLS data
# SpecObj_data = hdulist[1].data
# SDSS_data = hdulist2[1].data
# DECaLS_data = hdulist3[1].data
#
#
# ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG = readData(SpecObj_data, SDSS_data, DECaLS_data)
