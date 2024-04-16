import matplotlib.pyplot as plt
import numpy as np
import os
import utility.alma as ual
import utility.astro as uas
import utility.skymodels as usm

def plotter(inx, output_dir, beam_size):
    plot_dir = os.path.join(output_dir, 'plots')
    clean, clean_header = uas.load_fits(os.path.join(output_dir, "clean_cube_" + str(inx) +".fits"))
    dirty, dirty_header = uas.load_fits(os.path.join(output_dir, "dirty_cube_" + str(inx) +".fits"))
    clean = clean[0]
    dirty = dirty[0]
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cell_size = beam_size / 5 
    pixel_solid_angle = cell_size ** 2
    pix_to_beam = beam_solid_angle / pixel_solid_angle
    clean_spectrum = np.sum(clean[:, :, :], axis=(1, 2))
    dirty_spectrum = np.where(dirty < 0, 0, dirty)
    dirty_spectrum = np.nansum(dirty_spectrum[:, :, :], axis=(1, 2))
    clean_image = np.sum(clean[:, :, :], axis=0)[np.newaxis, :, :]
    dirty_image = np.nansum(dirty[:, :, :], axis=0)[np.newaxis, :, :]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(clean_image[0] * pix_to_beam, origin='lower')
    ax[1].imshow(dirty_image[0] * pix_to_beam, origin='lower')
    plt.colorbar(ax[0].imshow(clean_image[0] * pix_to_beam, origin='lower'), ax=ax[0], label='Jy/beam')
    plt.colorbar(ax[1].imshow(dirty_image[0] * pix_to_beam, origin='lower'), ax=ax[1], label='Jy/beam')
    ax[0].set_title('Sky Model Image')
    ax[1].set_title('ALMA Observed Image')
    plt.savefig(os.path.join(plot_dir, 'sim_{}.png'.format(inx)))
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(clean_spectrum * pix_to_beam)
    ax[1].plot(dirty_spectrum * pix_to_beam)
    ax[0].set_title('Clean Sky Model Spectrum')
    ax[1].set_title('ALMA Simulated Spectrum')
    ax[0].set_xlabel('Frequency Channel')
    plt.savefig(os.path.join(plot_dir, 'sim-spectra_{}.png'.format(inx)))
    plt.close()   
