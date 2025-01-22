import numpy as np
import pytest

import autogalaxy as ag


def test__model_visibilities(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.model_data.slim[0] == pytest.approx(np.array([1.48496 + 0.0]), 1.0e-4)
    assert fit.log_likelihood == pytest.approx(-34.16859, 1.0e-4)


def test__fit_figure_of_merit(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0, g1])

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-2398107.3849, 1.0e-4)

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Sersic(intensity=1.0),
            ag.lp.Sersic(intensity=1.0),
        ]
    )

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-2398107.3849, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.01),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[ag.Galaxy(redshift=0.5), g0],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-66.90612, 1.0e-4)

    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[galaxy_light, galaxy_pix],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-283424.48941, 1.0e-4)

    g0_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0)
    )

    g1_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=4.0)
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, g1_linear_light]
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.44419, 1.0e-4)

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp_linear.Sersic(sersic_index=1.0),
            ag.lp_linear.Sersic(sersic_index=4.0),
        ]
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, g1_linear_light]
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.44419, 1.0e-4)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, galaxy_pix]
    )

    assert fit.log_evidence == pytest.approx(-35.16806296, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-35.16806296, 1.0e-4)


def test___fit_figure_of_merit__different_settings(
    interferometer_7, interferometer_7_lop
):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.01),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7_lop,
        galaxies=[ag.Galaxy(redshift=0.5), g0],
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=True
        ),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-71.5177, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-71.5177, 1.0e-4)


def test___galaxy_model_image_dict(interferometer_7):
    # Normal Light Profiles Only

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0),
        light_profile_1=ag.lp.Sersic(intensity=2.0),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_image = g0.image_2d_from(grid=interferometer_7.grids.lp)
    g1_image = g1.image_2d_from(grid=interferometer_7.grids.lp)

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(g0_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[g2] == pytest.approx(g0_image + g1_image, 1.0e-4)

    # Linear Light Profiles Only

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        0.99967378, 1.0e-4
    )

    # Pixelization + Regularizaiton only

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    mapper_grids = pixelization.mesh.mapper_grids_from(
        mask=interferometer_7.real_space_mask,
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        mapper_grids=mapper_grids,
        border_relocator=interferometer_7.grids.border_relocator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_image_dict[g0].native == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_image_dict[galaxy_pix_0] == pytest.approx(
        inversion.mapped_reconstructed_image.slim, 1.0e-4
    )

    # Linear Light PRofiles + Pixelization + Regularizaiton

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -1.65495642e03, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        -0.000164926, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(
        -0.000153471881, 1.0e-2
    )

    mapped_reconstructed_image = (
        fit.galaxy_model_image_dict[g0_linear]
        + fit.galaxy_model_image_dict[galaxy_pix_0]
        + fit.galaxy_model_image_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_image == pytest.approx(
        fit.inversion.mapped_reconstructed_image, 1.0e-4
    )


def test___galaxy_model_visibilities_dict(interferometer_7):
    # Normal Light Profiles Only

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0),
        light_profile_1=ag.lp.Sersic(intensity=2.0),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_visibilities = g0.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0] == pytest.approx(
        g0_visibilities, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1] == pytest.approx(
        g1_visibilities, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g2] == pytest.approx(
        g0_visibilities + g1_visibilities, 1.0e-4
    )

    # Linear Light Profiles Only

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        0.9999355379224923 - 1.6755584672528312e-20j, 1.0e-4
    )

    # Pixelization + Regularizaiton only

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    mapper_grids = pixelization.mesh.mapper_grids_from(
        mask=interferometer_7.real_space_mask,
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        mapper_grids=mapper_grids,
        border_relocator=interferometer_7.grids.border_relocator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_visibilities_dict[g0] == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0] == pytest.approx(
        inversion.mapped_reconstructed_data, 1.0e-4
    )

    # Linear Light PRofiles + Pixelization + Regularizaiton
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        -1655.3897388539438 + 2.7738811036788807e-17j, 1.0e-4
    )

    assert fit.galaxy_model_visibilities_dict[g1] == pytest.approx(
        g1_visibilities, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0][0] == pytest.approx(
        -0.00023567176731092987 + 0.2291185445396378j, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_1][0] == pytest.approx(
        -0.0002255902723656833 + 0.057279636670966916j, 1.0e-4
    )

    mapped_reconstructed_visibilities = (
        fit.galaxy_model_visibilities_dict[g0_linear]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_0]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_visibilities == pytest.approx(
        fit.inversion.mapped_reconstructed_data, 1.0e-4
    )


def test__model_visibilities_of_galaxies_list(interferometer_7):
    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    galaxy_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[galaxy_light, galaxy_linear, galaxy_pix]
    )

    assert fit.model_visibilities_of_galaxies_list[0] == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_light], 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[1] == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_linear], 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[2] == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_pix], 1.0e-4
    )
