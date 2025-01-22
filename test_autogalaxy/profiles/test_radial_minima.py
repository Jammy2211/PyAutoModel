from __future__ import division, print_function

from os import path
import autogalaxy as ag

import pytest

directory = path.dirname(path.realpath(__file__))


def test__grid_2d__moves_radial_coordinates__does_not_double_transform():
    grid_2d = ag.Grid2D.no_mask(values=[[[0.0, 0.0]]], pixel_scales=1.0)
    grid_2d_offset = ag.Grid2D.no_mask(values=[[[0.0001, 0.0001]]], pixel_scales=1.0)

    isothermal = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_offset)

    assert convergence_0 == pytest.approx(convergence_1, 1.0e-8)

    grid_2d = ag.Grid2D.no_mask(
        values=[[[0.5, 0.5]]], pixel_scales=1.0, origin=(0.5, 0.5)
    )
    grid_2d_offset = ag.Grid2D.no_mask(
        values=[[[0.5001, 0.5001]]], pixel_scales=1.0, origin=(0.5001, 0.5001)
    )

    isothermal = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_offset)

    assert convergence_0 != pytest.approx(convergence_1, 1.0e-8)

    isothermal = ag.mp.Isothermal(centre=(0.5, 0.5), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_offset)

    assert convergence_0 == pytest.approx(convergence_1, 1.0e-5)


def test__grid_2d_irrergular__moves_radial_coordinates__does_not_double_transform():
    grid_2d_irregular = ag.Grid2DIrregular(values=[[0.0, 0.0]])
    grid_2d_irregular_offset = ag.Grid2DIrregular(values=[[0.0001, 0.0001]])

    isothermal = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d_irregular)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_irregular_offset)

    assert convergence_0 == pytest.approx(convergence_1, 1.0e-8)

    grid_2d_irregular = ag.Grid2DIrregular(values=[[0.5, 0.5]])
    grid_2d_irregular_offset = ag.Grid2DIrregular(values=[[0.5001, 0.5001]])

    isothermal = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d_irregular)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_irregular_offset)

    assert convergence_0 != pytest.approx(convergence_1, 1.0e-8)

    isothermal = ag.mp.Isothermal(centre=(0.5, 0.5), einstein_radius=1.0)

    convergence_1 = isothermal.convergence_2d_from(grid=grid_2d_irregular)
    convergence_0 = isothermal.convergence_2d_from(grid=grid_2d_irregular_offset)

    assert convergence_0 == pytest.approx(convergence_1, 1.0e-8)
