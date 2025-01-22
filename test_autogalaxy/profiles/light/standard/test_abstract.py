from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag


def luminosity_from_radius_and_profile(radius, profile):
    x = profile.sersic_constant * (
        (radius / profile.effective_radius) ** (1.0 / profile.sersic_index)
    )

    return (
        profile.intensity
        * profile.effective_radius**2
        * 2
        * math.pi
        * profile.sersic_index
        * (
            (math.e**profile.sersic_constant)
            / (profile.sersic_constant ** (2 * profile.sersic_index))
        )
        * scipy.special.gamma(2 * profile.sersic_index)
        * scipy.special.gammainc(2 * profile.sersic_index, x)
    )


def test__luminosity_within_centre__compare_to_gridded_calculations():
    sersic = ag.lp.SersicSph(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

    luminosity_analytic = luminosity_from_radius_and_profile(radius=0.5, profile=sersic)

    luminosity_integral = sersic.luminosity_within_circle_from(radius=0.5)

    assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    luminosity_grid = luminosity_from_radius_and_profile(radius=1.0, profile=sersic)

    luminosity_integral = sersic.luminosity_within_circle_from(radius=1.0)

    assert luminosity_grid == pytest.approx(luminosity_integral, 0.02)


def test__image_1d_from__grid_2d_in__returns_1d_image_via_projected_quantities():
    grid_2d = ag.Grid2D.uniform(
        shape_native=(5, 5),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    image_1d = lp.image_1d_from(grid=grid_2d)
    image_2d = lp.image_2d_from(grid=grid_2d)

    assert image_1d[0] == pytest.approx(image_2d.native[2, 2], 1.0e-4)
    assert image_1d[1] == pytest.approx(image_2d.native[2, 3], 1.0e-4)
    assert image_1d[2] == pytest.approx(image_2d.native[2, 4], 1.0e-4)

    lp = ag.lp.Gaussian(
        centre=(0.2, 0.2), ell_comps=(0.3, 0.3), intensity=1.0, sigma=1.0
    )

    image_1d = lp.image_1d_from(grid=grid_2d)

    grid_2d_projected = grid_2d.grid_2d_radial_projected_from(
        centre=lp.centre, angle=lp.angle + 90.0
    )

    image_projected = lp.image_2d_from(grid=grid_2d_projected)

    assert image_1d == pytest.approx(image_projected, 1.0e-4)
    assert (image_1d.grid_radial == np.array([0.0, 1.0, 2.0])).all()


def test__decorator__oversample_uniform__numerical_values(gal_x1_lp):
    mask = ag.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    lp = ag.lp.Sersic(intensity=1.0)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image = lp.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.15987224303572964, 1.0e-6)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=2)

    image = lp.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.17481917, 1.0e-6)
    assert image[1] == pytest.approx(0.39116856, 1.0e-6)

    lp = ag.lp.Sersic(centre=(3.0, 3.0), intensity=1.0)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image = lp.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.006719704400094508, 1.0e-6)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=2)

    image = lp.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.00681791, 1.0e-6)
    assert image[1] == pytest.approx(0.01332332, 1.0e-6)


def test__regression__centre_of_profile_in_right_place():
    grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

    lp = ag.lp.Sersic(centre=(2.0, 1.0), intensity=1.0)
    image = lp.image_2d_from(grid=grid)
    max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
    assert max_indexes == (1, 4)

    lp = ag.lp.SersicSph(centre=(2.0, 1.0), intensity=1.0)
    image = lp.image_2d_from(grid=grid)
    max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
    assert max_indexes == (1, 4)
