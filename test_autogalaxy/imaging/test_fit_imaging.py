import numpy as np
import pytest

import autogalaxy as ag


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.m.MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.63377, 1.0e-4)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)


def test__noise_map__with_and_without_hyper_galaxy(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d_value=1.0))

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-20.7470, 1.0e-4)


def test__noise_map__with_hyper_galaxy_reaches_upper_limit(masked_imaging_7x7_no_blur):

    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=1.0e8, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-174.0565, 1.0e-4)


def test__image__with_and_without_hyper_background_sky(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d_value=1.0))

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7_no_blur, plane=plane, hyper_image_sky=hyper_image_sky
    )

    assert fit.image.slim == pytest.approx(np.full(fill_value=2.0, shape=(9,)), 1.0e-1)
    assert fit.log_likelihood == pytest.approx(-15.6337, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d_value=1.0))
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=3.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-18.1579, 1.0e-4)


def test__fit_figure_of_merit(masked_imaging_7x7, masked_imaging_covariance_7x7):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

    print(plane.cls_list_from(cls=ag.LightProfile))

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-75938.05, 1.0e-4)

    basis = ag.lp_basis.Basis(
        light_profile_list=[
            ag.lp.Sersic(intensity=1.0),
            ag.lp.Sersic(intensity=1.0),
        ]
    )

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(redshift=0.5, galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-75938.05, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.9005, 1.0e-4)

    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-6840.5851, 1.0e-4)

    g0_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0)
    )

    g1_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=4.0)
    )

    plane = ag.Plane(redshift=0.5, galaxies=[g0_linear_light, g1_linear_light])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.52327, 1.0e-4)

    basis = ag.lp_basis.Basis(
        light_profile_list=[
            ag.lp_linear.Sersic(sersic_index=1.0),
            ag.lp_linear.Sersic(sersic_index=4.0),
        ]
    )

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(redshift=0.5, galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.52327, 1.0e-4)

    basis = ag.lp_basis.Basis(
        light_profile_list=[
            ag.lp_linear.Sersic(sersic_index=1.0),
            ag.lp_linear.Sersic(sersic_index=4.0),
        ],
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(redshift=0.5, galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-29.21448984, 1.0e-4)

    g0_operated_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_operated.Sersic(intensity=1.0)
    )
    g1_operated_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_operated.Sersic(intensity=1.0)
    )

    plane = ag.Plane(redshift=0.5, galaxies=[g0_operated_light, g1_operated_light])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-342374.9618, 1.0e-4)

    g0_linear_operated_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear_operated.Sersic(sersic_index=1.0)
    )
    g1_linear_operated_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear_operated.Sersic(sersic_index=4.0)
    )

    plane = ag.Plane(
        redshift=0.5, galaxies=[g0_linear_operated_light, g1_linear_operated_light]
    )

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.7237273, 1.0e-4)

    plane = ag.Plane(redshift=0.5, galaxies=[g0_linear_light, galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.87827302, 1.0e-4)

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_covariance_7x7, plane=plane)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-130242.56, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):

    hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    hyper_galaxy = ag.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-12104.68, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-30.1448, 1.0e-4)

    galaxy_light = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-1132.9297, 1.0e-4)


def test__galaxy_model_image_dict(masked_imaging_7x7):

    # Normal Light Profiles Only

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0),
        light_profile_1=ag.lp.Sersic(intensity=2.0),
    )
    g3 = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1, g2, g3])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    g0_blurred_image_2d = g0.blurred_image_2d_from(
        grid=masked_imaging_7x7.grid,
        blurring_grid=masked_imaging_7x7.blurring_grid,
        convolver=masked_imaging_7x7.convolver,
    )

    g1_blurred_image_2d = g1.blurred_image_2d_from(
        grid=masked_imaging_7x7.grid,
        blurring_grid=masked_imaging_7x7.blurring_grid,
        convolver=masked_imaging_7x7.convolver,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(g0_blurred_image_2d, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_blurred_image_2d, 1.0e-4)
    assert fit.galaxy_model_image_dict[g2] == pytest.approx(
        g0_blurred_image_2d + g1_blurred_image_2d, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g3].slim == np.zeros(9)).all()

    assert fit.model_image == pytest.approx(
        fit.galaxy_model_image_dict[g0]
        + fit.galaxy_model_image_dict[g1]
        + fit.galaxy_model_image_dict[g2],
        1.0e-4,
    )

    # Linear Light Profiles only

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    plane = ag.Plane(redshift=0.5, galaxies=[g0_linear, g3])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        1.50112088e00, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g3] == np.zeros(9)).all()

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0_linear].native, 1.0e-4
    )

    # Pixelization + Regularizaiton only

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (fit.galaxy_model_image_dict[g0] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g1][4] == pytest.approx(1.2570779, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
        fit.inversion.mapped_reconstructed_image.native, 1.0e-4
    )

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g1].native, 1.0e-4
    )

    # Linear Light PRofiles + Pixelization + Regularizaiton

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(
        redshift=0.5, galaxies=[g0, g1_linear, g3, galaxy_pix_0, galaxy_pix_1]
    )

    masked_imaging_7x7.image[0] = 3.0

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_image_dict[g3] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g0][4] == pytest.approx(276.227301, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1_linear][4] == pytest.approx(
        -277.619503, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        1.085283555, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(
        1.085283673, 1.0e-4
    )

    mapped_reconstructed_image = (
        fit.galaxy_model_image_dict[g1_linear]
        + fit.galaxy_model_image_dict[galaxy_pix_0]
        + fit.galaxy_model_image_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_image == pytest.approx(
        fit.inversion.mapped_reconstructed_image, 1.0e-4
    )

    assert fit.model_image == pytest.approx(
        fit.galaxy_model_image_dict[g0] + fit.inversion.mapped_reconstructed_image,
        1.0e-4,
    )


def test__model_images_of_galaxies_list(masked_imaging_7x7):

    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    galaxy_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_light, galaxy_linear, galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.model_images_of_galaxies_list[0] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_light], 1.0e-4
    )
    assert fit.model_images_of_galaxies_list[1] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_linear], 1.0e-4
    )
    assert fit.model_images_of_galaxies_list[2] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix], 1.0e-4
    )


def test___unmasked_blurred_images(masked_imaging_7x7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    unmasked_blurred_image = plane.unmasked_blurred_image_2d_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (fit.unmasked_blurred_image == unmasked_blurred_image).all()

    unmasked_blurred_image_of_galaxies_list = plane.unmasked_blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (
        fit.unmasked_blurred_image_of_galaxies_list[0]
        == unmasked_blurred_image_of_galaxies_list[0]
    ).all()
    assert (
        fit.unmasked_blurred_image_of_galaxies_list[1]
        == unmasked_blurred_image_of_galaxies_list[1]
    ).all()


def test__subtracted_images_of_galaxies(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    g1 = ag.Galaxy(redshift=0.5)

    g2 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=3.0))

    plane = ag.Plane(redshift=0.5, galaxies=[g0, g1, g2])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.subtracted_images_of_galaxies_list[0].slim[0] == pytest.approx(
        0.520383, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_list[1].slim[0] == pytest.approx(
        0.360511, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_list[2].slim[0] == pytest.approx(
        0.840127, 1.0e-4
    )


def test__light_profile_linear__intensity_dict(masked_imaging_7x7):

    linear_light_0 = ag.lp_linear.Sersic(sersic_index=1.0)
    linear_light_1 = ag.lp_linear.Sersic(sersic_index=4.0)

    g0_linear_light = ag.Galaxy(redshift=0.5, bulge=linear_light_0)

    g1_linear_light = ag.Galaxy(redshift=0.5, bulge=linear_light_1)

    plane = ag.Plane(redshift=0.5, galaxies=[g0_linear_light, g1_linear_light])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.linear_light_profile_intensity_dict[linear_light_0] == pytest.approx(
        7.093227476666252, 1.0e-4
    )
    assert fit.linear_light_profile_intensity_dict[linear_light_1] == pytest.approx(
        -0.04694839915145, 1.0e-4
    )

    basis = ag.lp_basis.Basis(light_profile_list=[linear_light_0, linear_light_1])

    g_basis = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(redshift=0.5, galaxies=[g_basis])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.linear_light_profile_intensity_dict[linear_light_0] == pytest.approx(
        7.093227476666252, 1.0e-4
    )
    assert fit.linear_light_profile_intensity_dict[linear_light_1] == pytest.approx(
        -0.04694839915145, 1.0e-4
    )

    linear_light_2 = ag.lp_linear.Sersic(sersic_index=2.0)
    linear_light_3 = ag.lp_linear.Sersic(sersic_index=3.0)

    basis = ag.lp_basis.Basis(light_profile_list=[linear_light_2, linear_light_3])

    g_basis = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(redshift=0.5, galaxies=[g0_linear_light, g_basis])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.linear_light_profile_intensity_dict[linear_light_0] == pytest.approx(
        -21.77759470, 1.0e-4
    )
    assert fit.linear_light_profile_intensity_dict[linear_light_2] == pytest.approx(
        29.3935231947, 1.0e-4
    )
    assert fit.linear_light_profile_intensity_dict[linear_light_3] == pytest.approx(
        -4.77469646, 1.0e-4
    )


def test__plane_linear_light_profiles_to_light_profiles(masked_imaging_7x7):

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0))

    g1_linear = ag.Galaxy(redshift=1.0, bulge=ag.lp_linear.Sersic(sersic_index=4.0))

    plane = ag.Plane(galaxies=[g0, g0_linear, g1_linear])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)

    plane = fit.plane_linear_light_profiles_to_light_profiles

    assert plane.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)
    assert plane.galaxies[1].bulge.intensity == pytest.approx(7.0932274, 1.0e-4)
    assert plane.galaxies[2].bulge.intensity == pytest.approx(-1.04694839, 1.0e-4)


# def test__light_profile_no_convolve(masked_imaging_7x7):
#
#     g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Gaussian(intensity=1.0))
#     plane = ag.Plane(redshift=0.5, galaxies=[g0])
#     fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)
#
#     g0_no_convolve = ag.Galaxy(
#         redshift=0.5, bulge=ag.lp.Gaussian(intensity=1.0)
#     )
#     plane_no_convolve = ag.Plane(redshift=0.5, galaxies=[g0_no_convolve])
#     fit_no_convolve = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane_no_convolve)
#
#     assert fit_no_convolve.log_likelihood != fit.log_likelihood
