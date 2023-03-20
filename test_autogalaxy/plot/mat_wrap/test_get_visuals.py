from os import path
import pytest

import autogalaxy as ag
import autogalaxy.plot as aplt

from autogalaxy.plot.get_visuals.one_d import GetVisuals1D
from autogalaxy.plot.get_visuals.two_d import GetVisuals2D

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__1d__via_light_obj_from(lp_0):

    visuals_1d = aplt.Visuals1D()
    include_1d = aplt.Include1D(half_light_radius=True)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_light_obj_from(light_obj=lp_0)

    assert visuals_1d_via.half_light_radius == lp_0.half_light_radius

    include_1d = aplt.Include1D(half_light_radius=False)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_light_obj_from(light_obj=lp_0)

    assert visuals_1d_via.half_light_radius is None


def test__1d__via_light_obj_list_from(lp_0):

    visuals_1d = aplt.Visuals1D()
    include_1d = aplt.Include1D(half_light_radius=True)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_light_obj_list_from(
        light_obj_list=[lp_0, lp_0], low_limit=1.0
    )

    assert visuals_1d_via.half_light_radius == lp_0.half_light_radius
    assert visuals_1d_via.half_light_radius_errors[0][0] == lp_0.half_light_radius

    include_1d = aplt.Include1D(half_light_radius=False)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_light_obj_list_from(
        light_obj_list=[lp_0, lp_0], low_limit=1.0
    )

    assert visuals_1d_via.half_light_radius is None
    assert visuals_1d_via.half_light_radius_errors is None


def test__1d__via_mass_obj_from(mp_0, grid_2d_7x7):

    visuals_1d = aplt.Visuals1D()
    include_1d = aplt.Include1D(einstein_radius=True)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_mass_obj_from(mass_obj=mp_0, grid=grid_2d_7x7)

    assert visuals_1d_via.einstein_radius == mp_0.einstein_radius_from(grid=grid_2d_7x7)

    include_1d = aplt.Include1D(einstein_radius=False)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_mass_obj_from(mass_obj=mp_0, grid=grid_2d_7x7)

    assert visuals_1d_via.einstein_radius is None


def test__1d__via_mass_obj_list_from(mp_0, grid_2d_7x7):

    visuals_1d = aplt.Visuals1D()
    include_1d = aplt.Include1D(einstein_radius=True)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_mass_obj_list_from(
        mass_obj_list=[mp_0, mp_0], grid=grid_2d_7x7, low_limit=1.0
    )

    assert visuals_1d_via.einstein_radius == mp_0.einstein_radius_from(grid=grid_2d_7x7)
    assert visuals_1d_via.einstein_radius_errors[0][0] == mp_0.einstein_radius_from(
        grid=grid_2d_7x7
    )

    include_1d = aplt.Include1D(einstein_radius=False)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_mass_obj_list_from(
        mass_obj_list=[mp_0, mp_0], grid=grid_2d_7x7, low_limit=1.0
    )

    assert visuals_1d_via.einstein_radius is None
    assert visuals_1d_via.einstein_radius_errors is None


def test__2d__via_light_obj_from(lp_0, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(vectors=2)
    include_2d = aplt.Include2D(
        origin=True, mask=True, border=True, light_profile_centres=True
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_light_obj_from(light_obj=lp_0, grid=grid_2d_7x7)

    assert visuals_2d_via.origin.in_list == [(0.0, 0.0)]
    assert (visuals_2d_via.mask == grid_2d_7x7.mask).all()
    assert (
        visuals_2d_via.border == grid_2d_7x7.mask.derive_grid.border_sub_1.binned
    ).all()
    assert visuals_2d_via.light_profile_centres.in_list == [lp_0.centre]
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=False, mask=False, border=False, light_profile_centres=False
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_light_obj_from(light_obj=lp_0, grid=grid_2d_7x7)

    assert visuals_2d_via.origin is None
    assert visuals_2d_via.mask is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.light_profile_centres is None
    assert visuals_2d_via.vectors == 2


def test__2d__via_mass_obj(mp_0, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(vectors=2)
    include_2d = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        mass_profile_centres=True,
        tangential_critical_curves=True,
        radial_critical_curves=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mass_obj_from(mass_obj=mp_0, grid=grid_2d_7x7)

    assert visuals_2d_via.origin.in_list == [(0.0, 0.0)]
    assert (visuals_2d_via.mask == grid_2d_7x7.mask).all()
    assert (
        visuals_2d_via.border == grid_2d_7x7.mask.derive_grid.border_sub_1.binned
    ).all()
    assert visuals_2d_via.mass_profile_centres.in_list == [mp_0.centre]
    assert (
        visuals_2d_via.tangential_critical_curves[0]
        == mp_0.tangential_critical_curve_list_from(grid=grid_2d_7x7)[0]
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=False,
        mask=False,
        border=False,
        mass_profile_centres=False,
        tangential_critical_curves=False,
        radial_critical_curves=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mass_obj_from(mass_obj=mp_0, grid=grid_2d_7x7)

    assert visuals_2d_via.origin is None
    assert visuals_2d_via.mask is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.mass_profile_centres is None
    assert visuals_2d_via.tangential_critical_curves is None
    assert visuals_2d_via.radial_critical_curves is None
    assert visuals_2d_via.vectors == 2


def test__2d__via_light_mass_obj(gal_x1_lp_x1_mp, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(vectors=2)
    include_2d = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        tangential_critical_curves=True,
        radial_critical_curves=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_light_mass_obj_from(
        light_mass_obj=gal_x1_lp_x1_mp, grid=grid_2d_7x7
    )

    assert visuals_2d_via.origin.in_list == [(0.0, 0.0)]
    assert (visuals_2d_via.mask == grid_2d_7x7.mask).all()
    assert (
        visuals_2d_via.border == grid_2d_7x7.mask.derive_grid.border_sub_1.binned
    ).all()
    assert visuals_2d_via.light_profile_centres.in_list == [
        gal_x1_lp_x1_mp.light_profile_0.centre
    ]
    assert visuals_2d_via.mass_profile_centres.in_list == [
        gal_x1_lp_x1_mp.mass_profile_0.centre
    ]
    assert (
        visuals_2d_via.tangential_critical_curves[0]
        == gal_x1_lp_x1_mp.tangential_critical_curve_list_from(grid=grid_2d_7x7)[0]
    ).all()
    assert (
        visuals_2d_via.radial_critical_curves[0]
        == gal_x1_lp_x1_mp.radial_critical_curve_list_from(grid=grid_2d_7x7)[0]
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=False,
        mask=False,
        border=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        tangential_critical_curves=False,
        radial_critical_curves=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_light_mass_obj_from(
        light_mass_obj=gal_x1_lp_x1_mp, grid=grid_2d_7x7
    )

    assert visuals_2d_via.origin is None
    assert visuals_2d_via.mask is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.light_profile_centres is None
    assert visuals_2d_via.mass_profile_centres is None
    assert visuals_2d_via.tangential_critical_curves is None
    assert visuals_2d_via.radial_critical_curves is None
    assert visuals_2d_via.vectors == 2


def test__via_fit_imaging_from(fit_imaging_x2_galaxy_7x7, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)
    include_2d = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        tangential_critical_curves=True,
        radial_critical_curves=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_x2_galaxy_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == fit_imaging_x2_galaxy_7x7.mask).all()
    assert (
        visuals_2d_via.border
        == fit_imaging_x2_galaxy_7x7.mask.derive_grid.border_sub_1.binned
    ).all()
    assert visuals_2d_via.light_profile_centres.in_list == [(0.0, 0.0), (0.0, 0.0)]
    assert visuals_2d_via.mass_profile_centres.in_list == [(0.0, 0.0)]
    assert (
        visuals_2d_via.tangential_critical_curves[0]
        == fit_imaging_x2_galaxy_7x7.plane.tangential_critical_curve_list_from(
            grid=grid_2d_7x7
        )[0]
    ).all()
    assert (
        visuals_2d_via.radial_critical_curves[0]
        == fit_imaging_x2_galaxy_7x7.plane.radial_critical_curve_list_from(
            grid=grid_2d_7x7
        )[0]
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=False,
        mask=False,
        border=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        tangential_critical_curves=False,
        radial_critical_curves=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_x2_galaxy_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.light_profile_centres is None
    assert visuals_2d_via.mass_profile_centres is None
    assert visuals_2d_via.tangential_critical_curves is None
    assert visuals_2d_via.radial_critical_curves is None
    assert visuals_2d_via.vectors == 2
