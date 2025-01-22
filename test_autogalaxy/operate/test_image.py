from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__blurred_image_2d_from(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3, convolver_7x7
):
    lp = ag.lp.Sersic(intensity=1.0)

    image_2d = lp.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d = lp.image_2d_from(grid=blurring_grid_2d_7x7)

    blurred_image_2d_manual = convolver_7x7.convolve_image(
        image=image_2d, blurring_image=blurring_image_2d
    )

    lp_blurred_image_2d = lp.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_manual.native == pytest.approx(
        lp_blurred_image_2d.native, 1.0e-4
    )

    lp_blurred_image_2d = lp.blurred_image_2d_from(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    assert blurred_image_2d_manual.native == pytest.approx(
        lp_blurred_image_2d.native, 1.0e-4
    )

    light_not_operated = ag.lp.Sersic(intensity=1.0)
    light_operated = ag.lp_operated.Gaussian(intensity=1.0)

    image_2d_not_operated = light_not_operated.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d_not_operated = light_not_operated.image_2d_from(
        grid=blurring_grid_2d_7x7
    )
    image_2d_operated = light_operated.image_2d_from(grid=grid_2d_7x7)

    galaxy = ag.Galaxy(
        redshift=0.5, light=light_not_operated, light_operated=light_operated
    )

    blurred_image_2d = galaxy.blurred_image_2d_from(
        grid=grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    blurred_image_2d_manual_not_operated = convolver_7x7.convolve_image(
        image=image_2d_not_operated,
        blurring_image=blurring_image_2d_not_operated,
    )

    assert (
        blurred_image_2d
        == pytest.approx(blurred_image_2d_manual_not_operated + image_2d_operated),
        1.0e-4,
    )


def test__x1_galaxies__padded_image__compare_to_galaxy_images_using_padded_grid_stack(
    grid_2d_7x7,
):
    padded_grid = grid_2d_7x7.padded_grid_from(kernel_shape_native=(3, 3))

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=3.0))

    padded_g0_image = g0.image_2d_from(grid=padded_grid)

    padded_g1_image = g1.image_2d_from(grid=padded_grid)

    padded_g2_image = g2.image_2d_from(grid=padded_grid)

    galaxies = ag.Galaxies(galaxies=[g0, g1, g2])

    padded_image = galaxies.padded_image_2d_from(grid=grid_2d_7x7, psf_shape_2d=(3, 3))

    assert padded_image.shape_native == (9, 9)
    assert padded_image == pytest.approx(
        padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
    )


def test__unmasked_blurred_image_2d_from():
    psf = ag.Kernel2D.no_mask(
        values=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
        pixel_scales=1.0,
    )

    mask = ag.Mask2D(
        mask=[[True, True, True], [True, False, True], [True, True, True]],
        pixel_scales=1.0,
    )

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    lp = ag.lp.Sersic(intensity=0.1)

    unmasked_blurred_image_2d = lp.unmasked_blurred_image_2d_from(grid=grid, psf=psf)

    assert unmasked_blurred_image_2d.native == pytest.approx(
        np.array(
            [
                [1.31618566e-01, 2.48460648e02, 1.91885830e-01],
                [9.66709359e-02, 8.29737297e01, 1.65678804e02],
                [4.12176698e-02, 8.74484826e-02, 1.01484934e-01],
            ]
        ),
        1.0e-4,
    )

    light_not_operated = ag.lp.Gaussian(intensity=1.0)
    light_operated = ag.lp_operated.Gaussian(intensity=1.0)

    padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

    image_2d_not_operated = light_not_operated.image_2d_from(grid=padded_grid)

    blurred_image_2d_not_operated = padded_grid.mask.unmasked_blurred_array_from(
        padded_array=image_2d_not_operated, psf=psf, image_shape=grid.mask.shape
    )

    image_2d_operated = light_operated.image_2d_from(grid=grid)

    image_2d_operated = light_operated.image_2d_from(grid=padded_grid)

    image_2d_operated = padded_grid.mask.unmasked_blurred_array_from(
        padded_array=image_2d_operated,
        psf=ag.Kernel2D.no_blur(pixel_scales=1.0),
        image_shape=grid.mask.shape,
    )

    image_2d_manual = image_2d_operated + blurred_image_2d_not_operated

    galaxy = ag.Galaxy(
        redshift=0.5, light=light_not_operated, light_operated=light_operated
    )

    unmasked_blurred_image_2d = galaxy.unmasked_blurred_image_2d_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d == pytest.approx(image_2d_manual, 1.0e-4)


def test__visibilities_from_grid_and_transformer(grid_2d_7x7, transformer_7x7_7):
    lp = ag.lp.Sersic(intensity=1.0)
    lp_visibilities = lp.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    image_2d = lp.image_2d_from(grid=grid_2d_7x7)
    visibilities = transformer_7x7_7.visibilities_from(image=image_2d)

    assert visibilities == pytest.approx(lp_visibilities, 1.0e-4)


def test__blurred_image_2d_list_from(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3, convolver_7x7
):
    lp_0 = ag.lp.Gaussian(intensity=1.0)
    lp_1 = ag.lp.Gaussian(intensity=2.0)

    lp_0_blurred_image_2d = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    lp_1_blurred_image_2d = lp_1.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        lp_1_blurred_image_2d.native, 1.0e-4
    )

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        lp_1_blurred_image_2d.native, 1.0e-4
    )

    lp_operated = ag.lp_operated.Gaussian(intensity=3.0)

    image_2d_operated = lp_operated.image_2d_from(grid=grid_2d_7x7)

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_operated=lp_operated)

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        image_2d_operated.native, 1.0e-4
    )

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        image_2d_operated.native, 1.0e-4
    )


def test__unmasked_blurred_image_2d_list_from():
    psf = ag.Kernel2D.no_mask(
        values=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
        pixel_scales=1.0,
    )

    mask = ag.Mask2D(
        mask=[[True, True, True], [True, False, True], [True, True, True]],
        pixel_scales=1.0,
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    lp_0 = ag.lp.Sersic(intensity=1.0)
    lp_1 = ag.lp.Sersic(intensity=2.0)

    padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

    manual_blurred_image_0 = lp_0.image_2d_from(grid=padded_grid)
    manual_blurred_image_0 = psf.convolved_array_from(array=manual_blurred_image_0)

    manual_blurred_image_1 = lp_1.image_2d_from(grid=padded_grid)
    manual_blurred_image_1 = psf.convolved_array_from(array=manual_blurred_image_1)

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    unmasked_blurred_image_2d_list = gal.unmasked_blurred_image_2d_list_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d_list[0].native == pytest.approx(
        manual_blurred_image_0.native[1:4, 1:4], 1.0e-4
    )

    assert unmasked_blurred_image_2d_list[1].native == pytest.approx(
        manual_blurred_image_1.native[1:4, 1:4], 1.0e-4
    )


def test__visibilities_list_from(grid_2d_7x7, transformer_7x7_7):
    lp_0 = ag.lp.Sersic(intensity=1.0)
    lp_1 = ag.lp.Sersic(intensity=2.0)

    lp_0_image = lp_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=grid_2d_7x7)

    lp_0_visibilities = transformer_7x7_7.visibilities_from(image=lp_0_image)
    lp_1_visibilities = transformer_7x7_7.visibilities_from(image=lp_1_image)

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    visibilities_list = gal.visibilities_list_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (lp_0_visibilities == visibilities_list[0]).all()
    assert (lp_1_visibilities == visibilities_list[1]).all()


def test__galaxy_blurred_image_2d_dict_from(
    grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7
):
    lp_0 = ag.lp.Sersic(intensity=1.0)

    g0 = ag.Galaxy(redshift=0.5, light_profile=lp_0)
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    light_profile_operated = ag.lp_operated.Gaussian(intensity=3.0)

    g2 = ag.Galaxy(redshift=0.5, light_profile_operated=light_profile_operated)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    blurred_image_2d_list = galaxies.blurred_image_2d_list_from(
        grid=grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    blurred_image_dict = galaxies.galaxy_blurred_image_2d_dict_from(
        grid=grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert (blurred_image_dict[g0] == blurred_image_2d_list[1]).all()
    assert (blurred_image_dict[g1] == blurred_image_2d_list[0]).all()
    assert (blurred_image_dict[g2] == blurred_image_2d_list[2]).all()

    image_2d = lp_0.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d = lp_0.image_2d_from(grid=blurring_grid_2d_7x7)

    image_2d_convolved = convolver_7x7.convolve_image(
        image=image_2d, blurring_image=blurring_image_2d
    )

    assert (blurred_image_dict[g0] == image_2d_convolved).all()

    image_2d_operated = light_profile_operated.image_2d_from(grid=grid_2d_7x7)

    assert (blurred_image_dict[g2] == image_2d_operated).all()


def test__galaxy_visibilities_dict_from(grid_2d_7x7, transformer_7x7_7):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=3.0))

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    visibilities_list = galaxies.visibilities_list_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    visibilities_dict = galaxies.galaxy_visibilities_dict_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (visibilities_dict[g0] == visibilities_list[1]).all()
    assert (visibilities_dict[g1] == visibilities_list[0]).all()
    assert (visibilities_dict[g2] == visibilities_list[2]).all()
