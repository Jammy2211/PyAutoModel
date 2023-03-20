from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from():

    shapelet = ag.lp_shapelets.ShapeletPolar(n=2, m=0, centre=(0.0, 0.0), beta=1.0)

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.0, 0.33177]), 1e-4)

    shapelet = ag.lp_shapelets.ShapeletPolar(n=2, m=0, centre=(0.2, 0.4), beta=1.0)

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.27715, 0.47333]), 1e-4)


def test__elliptical__image_2d_from():

    shapelet = ag.lp_shapelets.ShapeletPolarEll(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.1, 0.2), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.0, 0.33177]), 1e-4)

    shapelet = ag.lp_shapelets.ShapeletPolarEll(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.5, 0.7), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.0, 0.33177]), 1e-4)
