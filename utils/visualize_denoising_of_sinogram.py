import numpy as np
import matplotlib.pyplot as plt

def visualize_denoising_of_sinogram(noisy_signal, inferred_noise, name_signal, gt_noise=None, name_noise=None, clim=[-3, 3]):
    noisy_signal = np.transpose(noisy_signal)
    inferred_noise = np.transpose(inferred_noise)
    name_signal = name_signal

    if gt_noise is not None:
        gt_noise = np.transpose(gt_noise)
        name_noise = name_noise
        gt_noise_provided = True
    else:
        gt_noise = np.zeros(inferred_noise.shape)
        name_noise = 'GT not available'
        gt_noise_provided = False

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(w=18, h=10)

    fig.sca(ax[0])
    im = ax[0].imshow(noisy_signal, clim=clim, cmap='bone')
    ax[0].set_title('Noisy [' + name_signal + ']')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, ax=ax[0], shrink=0.5)

    fig.sca(ax[1])
    im = ax[1].imshow(noisy_signal-inferred_noise, clim=clim, cmap='bone')
    ax[1].set_title('Denoised [' + name_signal + ']')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, ax=ax[1], shrink=0.5)

    fig.sca(ax[2])
    if gt_noise_provided:
        im = ax[2].imshow(gt_noise-inferred_noise, clim=clim, cmap='bone')
        ax[2].set_title('gt - inferred noise [' + name_noise + ']')
    else:
        im = ax[2].imshow(inferred_noise, clim=clim, cmap='bone')
        ax[2].set_title('Inferred noise [' + name_signal + ']')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, ax=ax[2], shrink=0.5)

    return fig
