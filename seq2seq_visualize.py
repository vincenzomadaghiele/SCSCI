import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__ == '__main__': 

    # LOAD VAE TIMBRE SPACE
    corpus_name = 'elena-study'
    X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X)
    plot_scaler = StandardScaler()
    X_PCA_VAE = plot_scaler.fit_transform(X_PCA)

    # LOAD FLOW TIMBRE SPACE
    SYNTH_NAME = 'Granular'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X)
    plot_scaler = StandardScaler()
    X_PCA_FLOW = plot_scaler.fit_transform(X_PCA)

    # PLOT HEATMAP
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    heatmap, xedges, yedges = np.histogram2d(X_PCA_VAE[:,0], X_PCA_VAE[:,1], bins=150)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax[0].cla()
    ax[0].imshow(heatmap.T, extent=extent, origin='lower')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    heatmap, xedges, yedges = np.histogram2d(X_PCA_FLOW[:,0], X_PCA_FLOW[:,1], bins=150)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax[1].cla()
    ax[1].imshow(heatmap.T, extent=extent, origin='lower')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()




    # Source - https://stackoverflow.com/a/47350350
    # Posted by Jurgy, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-10-15, License - CC BY-SA 4.0

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage.filters import gaussian_filter

    def myplot(x, y, s, bins=1000):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    fig, axs = plt.subplots(2, 2)
    # Generate some test data
    x = X_PCA_FLOW[:,0]
    y = X_PCA_FLOW[:,1]
    sigmas = [0, 16, 32, 64]
    for ax, s in zip(axs.flatten(), sigmas):
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
    plt.show()
