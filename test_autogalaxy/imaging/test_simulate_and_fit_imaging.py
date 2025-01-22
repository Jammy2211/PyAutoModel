import os
from os import path
import shutil

import numpy as np
import pytest

import autogalaxy as ag


def test__perfect_fit__chi_squared_0():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1)
    )
    galaxy_1 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.output_to_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    dataset = ag.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
        over_sample_size_lp=1,
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.8,
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_imaging_data_and_fit__known_likelihood():
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2)

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        disk=ag.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    galaxy_1 = ag.Galaxy(redshift=1.0, pixelization=pixelization)

    simulator = ag.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.0
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.figure_of_merit == pytest.approx(538.9777105858, 1.0e-2)

    # Check that using a Basis gives the same result.

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
            ag.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
        ]
    )

    galaxy_0 = ag.Galaxy(redshift=0.5, bulge=basis)

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.figure_of_merit == pytest.approx(538.9777105858, 1.0e-2)


def test__simulate_imaging_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.8,
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy])

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(sersic_index=1.0),
        disk=ag.lp_linear.Sersic(sersic_index=4.0),
    )

    fit_linear = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=[galaxy_linear],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.2]), 1.0e-2
    )

    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)

    assert fit.log_likelihood == fit_linear.figure_of_merit
    assert fit_linear.figure_of_merit == pytest.approx(-45.02798, 1.0e-4)

    galaxy_image = galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        convolver=masked_dataset.convolver,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[galaxy_linear] == pytest.approx(
        galaxy_image, 1.0e-4
    )
