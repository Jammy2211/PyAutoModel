from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from():

    shapelet = ag.lp_shapelets.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.0, 0.0), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.1397, 0.0708009]), 1e-4)

    shapelet = ag.lp_shapelets.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.2, 0.4), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.23733, -0.07913]), 1e-4)


def test__elliptical__image_2d_from():

    shapelet = ag.lp_shapelets.ShapeletCartesianEll(
        n_y=2, n_x=3, centre=(0.0, 0.0), ell_comps=(0.1, 0.2), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.13444, 0.122273]), 1e-4)

    shapelet = ag.lp_shapelets.ShapeletCartesianEll(
        n_y=2, n_x=3, centre=(0.0, 0.0), ell_comps=(0.2, 0.3), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.12993, 0.13719]), 1e-4)
