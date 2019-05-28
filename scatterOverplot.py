def scatterOverplot(H, index, zmag, rzcolor, grcolor, edges):
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
        f, axarr = plt.subplots(1, 3, figsize=(15, 10))
        f.suptitle("Near Neighbors Overplotted Nsat Histogram")

        im1 = axarr[0].matshow(rz_vs_zmag, aspect='equal', extent=[edges[1][0], edges[1][len(edges[1]) - 1], edges[0][0], edges[0][len(edges[0]) - 1]], cmap='YlOrRd')
        axarr[0].scatter(zmag[index[0]], rzcolor[index[0]], s=2, c='blue')
        axarr[0].set_title("(r-z) vs zmag", pad=5)
        axarr[0].xaxis.set_ticks_position('bottom')
        axarr[0].invert_xaxis()
        axarr[0].set(xlabel='zmag', ylabel='(r-z)')
        axarr[0].set_ylim(bottom=-5, top=5)
        plt.colorbar(im1, ax=axarr[0], fraction=0.08, pad=0.05)

        im2 = axarr[1].matshow(gr_vs_zmag, aspect='equal', extent=[edges[1][0], edges[1][len(edges[1]) - 1], edges[2][0], edges[2][len(edges[2]) - 1]], cmap='YlOrRd')
        axarr[1].scatter(zmag[index[0]], grcolor[index[0]], s=2, c='blue')
        axarr[1].set_title("(g-r) vs zmag", pad=5)
        axarr[1].invert_xaxis()
        axarr[1].xaxis.set_ticks_position('bottom')
        axarr[1].set(xlabel='zmag', ylabel='(g-r)')
        axarr[1].set_ylim(bottom=-5, top=5)
        plt.colorbar(im2, ax=axarr[1], fraction=0.08, pad=0.05)

        im3 = axarr[2].matshow(gr_vs_rz, aspect=1.85, extent=[edges[0][0], edges[0][len(edges[0]) - 1], edges[2][0], edges[2][len(edges[2]) - 1]], cmap='YlOrRd')
        axarr[2].scatter(rzcolor[index[0]], grcolor[index[0]], s=2, c='blue')
        axarr[2].set_title("(g-r) vs (r-z)", pad=5)
        # axarr[2].invert_xaxis()
        axarr[2].xaxis.set_ticks_position('bottom')
        axarr[2].set(xlabel='(r-z)', ylabel='(g-r)')
        axarr[2].set_xlim(bottom=-5, top=5)
        axarr[2].set_ylim(bottom=-5, top=5)
        plt.colorbar(im3, ax=axarr[2], fraction=0.08, pad=0.05)

        f.subplots_adjust(wspace=0.8)

        # save image with incrementing file name
        plt.savefig('/Users/mtownsend/anaconda/GitHub/lrg-project/Plots/LRG_science_plots/HistProjections/Nsat_near_overplots/nearOverplot{}.jpeg'.format(i))
        # plt.show()