import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_via_integral_from():
    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    deflections = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(2.6134, 1e-3)
    assert deflections[0, 1] == pytest.approx(1.80719, 1e-3)


def test__deflections_2d_via_mge_from():
    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_mge = sersic.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)

    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_mge = sersic.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__deflections_2d_via_cse_from():
    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = sersic.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-4)

    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = sersic.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-3)

    sersic = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=1.5,
        mass_to_light_ratio=2.0,
    )

    deflections_via_integral = sersic.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = sersic.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-3)


def test__deflections_yx_2d_from():
    gaussian = ag.mp.Sersic()

    deflections = gaussian.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = gaussian.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)

    gaussian = ag.mp.SersicSph()

    deflections = gaussian.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = gaussian.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)

    elliptical = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SersicSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    np.testing.assert_almost_equal(
        elliptical.deflections_2d_via_integral_from(grid=grid),
        spherical.deflections_2d_via_integral_from(grid=grid),
    )


def test__convergence_2d_via_mge_from():
    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    convergence = sersic.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert convergence == pytest.approx(5.38066670129, 1e-3)


def test__convergence_2d_via_cse_from():
    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    convergence = sersic.convergence_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]])
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert convergence == pytest.approx(5.38066670129, 1e-3)


def test__convergence_2d_from():
    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    convergence = sersic.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)

    sersic = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = sersic.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.38066670129, 1e-3)

    elliptical = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SersicSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d, 1.0e-4)
