def HEALPix_map(ra_LRG, dec_LRG, ra_BKG, dec_BKG, nside, radius): 
    
    import healpy as hp
    import numpy as np
    
    # Using HEALPy/HEALPix to find background sources
                              
    ra = np.concatenate([ra_LRG, ra_BKG])
    dec = np.concatenate([dec_LRG, dec_BKG])
    npixel= hp.nside2npix(nside)

    # Make HEALPix map
    # Convert ra/dec into theta/phi
    theta_LRG = []
    phi_LRG = []

    for i in range(len(ra_cut_LRG)):
        theta_LRG.append(np.radians(90.-dec_LRG[i]))
        phi_LRG.append(np.radians(ra_LRG[i]))
    
    theta = []
    phi = []

    for i in range(len(ra)):
        theta.append(np.radians(90.-dec[i]))
        phi.append(np.radians(ra[i]))

    
    # Convert angles theta and phi to pixel numbers
    pix = hp.ang2pix(nside, theta, phi)

    # Create a HEALPix map from pix
    mapp = np.bincount(pix, minlength=npixel)

    # Plot mapp
    hp.gnomview(mapp, xsize = 225, rot=(-116.5, 8.25), flip='geo')
    plt.show()

    # Extracting the number of sources 
    
    # Convert theta and phi to a vector
    vec = hp.ang2vec(theta_LRG, phi_LRG, lonlat=False)

    # Query_disc to find indices of pixels within a search radius
    indices = []

    for i in range(len(vec)):
        indices.append(hp.query_disc(nside, vec[i], np.radians(radius)))
    
    # Get arrays for sources per pixel in the search radius
    temp0 = [] # temporary array
    source_arrays = []

    for i in range(len(indices)):
        for j in range(len(indices[i])):
            temp0.append(mapp[indices[i][j]])
        source_arrays.append(temp0)
        temp0 = []
    
    
    # Get arrays of number of sources
    temp1 = [] # temporary array
    source_numbers = []

    for i in range(len(source_arrays)):
        for j in range(len(source_arrays[i])):
            temp1.append(source_arrays[i][j])
            temp2 = np.array(temp1)
        source_numbers.append(np.sum(temp2))
        temp1 = []
    

    return(source_numbers)