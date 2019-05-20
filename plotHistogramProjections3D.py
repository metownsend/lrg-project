def plotHistogramProjections3D(H, edges):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'figure.max_open_warning': 0})

    for i in range(len(H)):
        # make 2D projections of 3D histogram
        rz_v_zmag = H[i][:, :, :].sum(axis=2)
        rz_vs_zmag = np.flipud(rz_v_zmag)
        gr_v_zmag = H[i][:, :, :].sum(axis=0)
        gr_vs_zmag = np.flipud(gr_v_zmag.T)
        gr_v_rz = H[i][:, :, :].sum(axis=1)
        gr_vs_rz = np.flipud(gr_v_rz.T)

        # plot 2D histograms using matshow; 3 plots per image
        f, axarr = plt.subplots(3, 1, figsize=(15, 15))
        f.suptitle("Nsat 2D Projections")
        im1 = axarr[0].matshow(rz_vs_zmag, extent=[edges[1][0], edges[1][len(edges[1]) - 1], edges[0][0],
                                                   edges[0][len(edges[0]) - 1]])
        axarr[0].set_title("(r-z) vs zmag", pad=10)
        axarr[0].invert_xaxis()
        plt.colorbar(im1, ax=axarr[0])

        im2 = axarr[1].matshow(gr_vs_zmag, extent=[edges[1][0], edges[1][len(edges[1]) - 1], edges[2][0],
                                                   edges[2][len(edges[2]) - 1]])
        axarr[1].set_title("(g-r) vs zmag", pad=10)
        axarr[1].invert_xaxis()
        plt.colorbar(im2, ax=axarr[1])

        im3 = axarr[2].matshow(gr_vs_rz, extent=[edges[0][0], edges[0][len(edges[0]) - 1], edges[2][0],
                                                 edges[2][len(edges[2]) - 1]])
        axarr[2].set_title("(g-r) vs (r-z)",pad=10)
        axarr[2].invert_xaxis()
        plt.colorbar(im3, ax=axarr[2])

        # save image with incrementing file name
        plt.savefig('/Users/mtownsend/anaconda/GitHub/lrg-project/Plots/LRG_science_plots/HistProjections/NsatProjections/Nsat3Dproj{}.jpeg'.format(i))
        # plt.show()