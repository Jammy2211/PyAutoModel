import numpy as np
import pytest
from skimage import measure

import autogalaxy as ag

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autogalaxy.galaxy.galaxies import plane_image_from

from autogalaxy import exc


def test__image_2d_from(grid_2d_7x7, gal_x1_lp):
    light_profile = gal_x1_lp.cls_list_from(cls=ag.LightProfile)[0]

    lp_image = light_profile.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[gal_x1_lp])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert (image == lp_image).all()

    galaxy_image = gal_x1_lp.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[gal_x1_lp])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image == pytest.approx(galaxy_image, 1.0e-4)

    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)

    g1_image = g1.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image == pytest.approx(g0_image + g1_image, 1.0e-4)


def test__image_2d_list_from(grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    lp0 = g0.cls_list_from(cls=ag.LightProfile)[0]
    lp1 = g1.cls_list_from(cls=ag.LightProfile)[0]

    lp0_image = lp0.image_2d_from(grid=grid_2d_7x7)
    lp1_image = lp1.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image[0] == pytest.approx(lp0_image[0] + lp1_image[0], 1.0e-4)
    assert image[1] == pytest.approx(lp0_image[1] + lp1_image[1], 1.0e-4)

    image_of_galaxies = galaxies.image_2d_list_from(grid=grid_2d_7x7)

    assert image_of_galaxies[0][0] == lp0_image[0]
    assert image_of_galaxies[0][1] == lp0_image[1]
    assert image_of_galaxies[1][0] == lp1_image[0]
    assert image_of_galaxies[1][1] == lp1_image[1]


def test__image_2d_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=False)
    assert image_2d == pytest.approx(image_2d_not_operated, 1.0e-4)

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=True)
    assert image_2d == pytest.approx(3.0 * image_2d_operated, 1.0e-4)

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=None)
    assert image_2d == pytest.approx(
        image_2d_not_operated + 3.0 * image_2d_operated, 1.0e-4
    )


def test__image_2d_list_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=False)
    assert image_2d_list[0] == pytest.approx(image_2d_not_operated, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(np.zeros((9)), 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((9)), 1.0e-4)

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=True)
    assert image_2d_list[0] == pytest.approx(image_2d_operated, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(2.0 * image_2d_operated, 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((9)), 1.0e-4)

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=None)
    assert image_2d_list[0] + image_2d_list[1] == pytest.approx(
        image_2d_not_operated + 3.0 * image_2d_operated, 1.0e-4
    )


def test__galaxy_image_2d_dict_from(grid_2d_7x7):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )

    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp_operated.Gaussian(intensity=3.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=grid_2d_7x7)
    g2_image = g2.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(grid=grid_2d_7x7)

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=True
    )

    assert (galaxy_image_2d_dict[g0] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g1] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=False
    )

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == np.zeros(shape=(9,))).all()


def test__convergence_2d_from(grid_2d_7x7):
    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_convergence = g0.convergence_2d_from(grid=grid_2d_7x7)

    g1_convergence = g1.convergence_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    convergence = galaxies.convergence_2d_from(grid=grid_2d_7x7)

    assert convergence == pytest.approx(g0_convergence + g1_convergence, 1.0e-8)


def test__potential_2d_from(grid_2d_7x7):
    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_potential = g0.potential_2d_from(grid=grid_2d_7x7)

    g1_potential = g1.potential_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    potential = galaxies.potential_2d_from(grid=grid_2d_7x7)

    assert potential == pytest.approx(g0_potential + g1_potential, 1.0e-8)


def test__deflections_yx_2d_from(grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_deflections = g0.deflections_yx_2d_from(grid=grid_2d_7x7)

    g1_deflections = g1.deflections_yx_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    deflections = galaxies.deflections_yx_2d_from(grid=grid_2d_7x7)

    assert deflections == pytest.approx(g0_deflections + g1_deflections, 1.0e-4)


def test__has():
    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5)])
    assert galaxies.has(cls=ag.LightProfile) is False

    galaxies = ag.Galaxies(
        galaxies=[ag.Galaxy(redshift=0.5, light_profile=ag.LightProfile())],
    )
    assert galaxies.has(cls=ag.LightProfile) is True

    galaxies = ag.Galaxies(
        galaxies=[
            ag.Galaxy(redshift=0.5, light_profile=ag.LightProfile()),
            ag.Galaxy(redshift=0.5),
        ],
    )
    assert galaxies.has(cls=ag.LightProfile) is True


def test__cls_list_from():
    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5)])

    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == []

    sis_0 = ag.mp.IsothermalSph(einstein_radius=1.0)
    sis_1 = ag.mp.IsothermalSph(einstein_radius=2.0)
    sis_2 = ag.mp.IsothermalSph(einstein_radius=3.0)

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass_profile=sis_0)])
    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == [sis_0]

    galaxies = ag.Galaxies(
        galaxies=[
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_0, mass_profile_1=sis_1),
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_2, mass_profile_1=sis_1),
        ],
    )
    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == [sis_0, sis_1, sis_2, sis_1]

    pixelization = ag.m.MockPixelization(mapper=1)

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    galaxies = ag.Galaxies(galaxies=[galaxy_pix])

    assert galaxies.cls_list_from(cls=ag.Pixelization)[0].mapper == 1

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    pixelization = ag.m.MockPixelization(mapper=2)

    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    galaxies = ag.Galaxies(galaxies=[galaxy_pix_0, galaxy_pix_1])

    assert galaxies.cls_list_from(cls=ag.Pixelization)[0].mapper == 1
    assert galaxies.cls_list_from(cls=ag.Pixelization)[1].mapper == 2

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    galaxies = ag.Galaxies(galaxies=[galaxy_no_pix])

    assert galaxies.cls_list_from(cls=ag.Pixelization) == []


def test__plane_image_from(grid_2d_7x7):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    plane_image = plane_image_from(grid=grid_2d_7x7, galaxies=[galaxy], buffer=0.1)

    assert plane_image[0] == pytest.approx(12.5595, 1.0e-3)
