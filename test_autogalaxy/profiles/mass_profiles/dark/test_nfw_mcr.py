import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_and_concentration_consistent_with_normal_nfw():

    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    nfw_mass = ag.mp.SphNFWMCRDuffy(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = nfw_mass.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = nfw_mass.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.SphNFW(
        centre=(1.0, 2.0),
        kappa_s=nfw_mass.kappa_s,
        scale_radius=nfw_mass.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the SphNFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(nfw_mass.kappa_s, float)

    assert nfw_mass.centre == (1.0, 2.0)

    assert nfw_mass.axis_ratio == 1.0
    assert isinstance(nfw_mass.axis_ratio, float)

    assert nfw_mass.angle == 0.0
    assert isinstance(nfw_mass.angle, float)

    assert nfw_mass.inner_slope == 1.0
    assert isinstance(nfw_mass.inner_slope, float)

    assert nfw_mass.scale_radius == pytest.approx(0.273382, 1.0e-4)


def test__mass_and_concentration_consistent_with_normal_nfw__scatter_0():

    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    nfw_mass = ag.mp.SphNFWMCRLudlow(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = nfw_mass.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = nfw_mass.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.SphNFW(
        centre=(1.0, 2.0),
        kappa_s=nfw_mass.kappa_s,
        scale_radius=nfw_mass.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the SphNFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(nfw_mass.kappa_s, float)

    assert nfw_mass.centre == (1.0, 2.0)

    assert nfw_mass.axis_ratio == 1.0
    assert isinstance(nfw_mass.axis_ratio, float)

    assert nfw_mass.angle == 0.0
    assert isinstance(nfw_mass.angle, float)

    assert nfw_mass.inner_slope == 1.0
    assert isinstance(nfw_mass.inner_slope, float)

    assert nfw_mass.scale_radius == pytest.approx(0.21157, 1.0e-4)

    deflections_ludlow = nfw_mass.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_elliptical():

    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    nfw_mass = ag.mp.EllNFWMCRLudlow(
        centre=(1.0, 2.0),
        elliptical_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = nfw_mass.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = nfw_mass.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.EllNFW(
        centre=(1.0, 2.0),
        elliptical_comps=(0.1, 0.2),
        kappa_s=nfw_mass.kappa_s,
        scale_radius=nfw_mass.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the SphNFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(nfw_mass.kappa_s, float)

    assert nfw_mass.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        elliptical_comps=(0.1, 0.2)
    )

    assert nfw_mass.axis_ratio == axis_ratio
    assert isinstance(nfw_mass.axis_ratio, float)

    assert nfw_mass.angle == angle
    assert isinstance(nfw_mass.angle, float)

    assert nfw_mass.inner_slope == 1.0
    assert isinstance(nfw_mass.inner_slope, float)

    assert nfw_mass.scale_radius == pytest.approx(0.211578, 1.0e-4)

    deflections_ludlow = nfw_mass.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_generalized_elliptical():

    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    nfw_mass = ag.mp.EllNFWGeneralizedMCRLudlow(
        centre=(1.0, 2.0),
        elliptical_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        inner_slope=2.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = nfw_mass.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = nfw_mass.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.EllNFWGeneralized(
        centre=(1.0, 2.0),
        elliptical_comps=(0.1, 0.2),
        kappa_s=nfw_mass.kappa_s,
        scale_radius=nfw_mass.scale_radius,
        inner_slope=2.0,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the SphNFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(nfw_mass.kappa_s, float)

    assert nfw_mass.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        elliptical_comps=(0.1, 0.2)
    )

    assert nfw_mass.axis_ratio == axis_ratio
    assert isinstance(nfw_mass.axis_ratio, float)

    assert nfw_mass.angle == angle
    assert isinstance(nfw_mass.angle, float)

    assert nfw_mass.inner_slope == 2.0
    assert isinstance(nfw_mass.inner_slope, float)

    assert nfw_mass.scale_radius == pytest.approx(0.21157, 1.0e-4)

    deflections_ludlow = nfw_mass.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()
