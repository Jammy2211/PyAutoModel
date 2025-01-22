from astropy.io import fits
import numpy as np
import os
from os import path
import pytest
import shutil

import autogalaxy as ag


def create_fits(fits_path, array):
    file_dir = os.path.split(fits_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.exists(fits_path):
        os.remove(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(array))

    hdu_list.writeto(f"{fits_path}")


def clean_fits(fits_path):
    if path.exists(fits_path):
        shutil.rmtree(fits_path)


def test__from_fits__all_imaging_data_structures_are_flipped_for_ds9():
    fits_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")
    image_path = path.join(fits_path, "data.fits")
    noise_map_path = path.join(fits_path, "noise_map.fits")
    psf_path = path.join(fits_path, "psf.fits")

    create_fits(fits_path=image_path, array=[[1.0, 0.0], [0.0, 0.0]])
    create_fits(fits_path=noise_map_path, array=[[2.0, 1.0], [1.0, 1.0]])
    create_fits(fits_path=psf_path, array=[[1.0, 1.0], [0.0, 0.0]])

    dataset = ag.Imaging.from_fits(
        data_path=image_path,
        noise_map_path=noise_map_path,
        psf_path=psf_path,
        pixel_scales=0.1,
    )

    assert (dataset.data.native == np.array([[0.0, 0.0], [1.0, 0.0]])).all()
    assert (dataset.noise_map.native == np.array([[1.0, 1.0], [2.0, 1.0]])).all()
    assert (dataset.psf.native == np.array([[0.0, 0.0], [0.5, 0.5]])).all()

    dataset.output_to_fits(
        data_path=image_path,
        noise_map_path=noise_map_path,
        psf_path=psf_path,
        overwrite=True,
    )

    hdu_list = fits.open(image_path)
    image = np.array(hdu_list[0].data).astype("float64")
    assert (image == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

    hdu_list = fits.open(noise_map_path)
    noise_map = np.array(hdu_list[0].data).astype("float64")
    assert (noise_map == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

    hdu_list = fits.open(psf_path)
    psf = np.array(hdu_list[0].data).astype("float64")
    assert (psf == np.array([[0.5, 0.5], [0.0, 0.0]])).all()

    clean_fits(fits_path=fits_path)


def test__simulator__via_galaxies_from():
    psf = ag.Kernel2D.from_gaussian(shape_native=(7, 7), sigma=0.5, pixel_scales=0.05)

    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.Sersic(intensity=0.3))

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    imaging_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (20, 20)
    assert dataset.data.native[0, 0] != imaging_via_image.data.native[0, 0]
    assert dataset.data.native[10, 10] == imaging_via_image.data.native[10, 10]
    assert dataset.psf == pytest.approx(imaging_via_image.psf, 1.0e-4)
    assert (dataset.noise_map == imaging_via_image.noise_map).all()


def test__simulator__simulate_imaging_from_galaxy__source_galaxy__compare_to_imaging():
    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
        ),
    )

    galaxy_1 = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.Sersic(
            centre=(0.1, 0.1),
            ell_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = ag.Kernel2D.no_mask(values=[[1.0]], pixel_scales=0.2)

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    imaging_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (11, 11)
    assert dataset.data == pytest.approx(imaging_via_image.data, 1.0e-4)
    assert (dataset.psf == imaging_via_image.psf).all()
    assert (dataset.noise_map == imaging_via_image.noise_map).all()
