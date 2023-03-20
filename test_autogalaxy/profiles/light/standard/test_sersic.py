from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    sersic = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
    )

    image = sersic.image_2d_from(grid=np.array([[1.0, 0.0]]))

    assert image == pytest.approx(0.351797, 1e-3)

    sersic = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )
    # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

    image = sersic.image_2d_from(grid=np.array([[1.5, 0.0]]))

    assert image == pytest.approx(4.90657319276, 1e-3)

    sersic = ag.lp.Sersic(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    image = sersic.image_2d_from(grid=np.array([[1.0, 0.0]]))

    assert image == pytest.approx(5.38066670129, 1e-3)

    elliptical = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    spherical = ag.lp.SersicSph(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)

    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
