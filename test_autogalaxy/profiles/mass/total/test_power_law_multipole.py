import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-0.036120991, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.0476260676, 1e-3)

    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-0.096376665, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.1298677210, 1e-3)


def test__convergence_2d_from():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.25958037, 1e-3)

    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.2875647, 1e-3)


def test__potential_2d_from():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert potential[0] == pytest.approx(0.0, 1e-3)
