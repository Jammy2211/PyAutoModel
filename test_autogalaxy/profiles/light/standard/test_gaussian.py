from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    gaussian = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

    assert image == pytest.approx(0.60653, 1e-2)

    gaussian = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
    )

    image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

    assert image == pytest.approx(2.0 * 0.60653, 1e-2)

    gaussian = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

    assert image == pytest.approx(0.88249, 1e-2)

    gaussian = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    image = gaussian.image_2d_from(grid=np.array([[0.0, 3.0]]))

    assert image == pytest.approx(0.3246, 1e-2)

    elliptical = ag.lp.Gaussian(ell_comps=(0.0, 0.0), intensity=3.0, sigma=2.0)
    spherical = ag.lp.GaussianSph(intensity=3.0, sigma=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
