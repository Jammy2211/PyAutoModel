from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():

    core_sersic = ag.lp.SersicCore(
        ell_comps=(0.0, 0.333333),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    image = core_sersic.image_2d_from(grid=np.array([[0.0, 0.1]]))

    assert image == pytest.approx(0.0255173, 1.0e-4)

    elliptical = ag.lp.SersicCore(
        ell_comps=(0.0, 0.0),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    spherical = ag.lp.SersicCoreSph(
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    image_elliptical = elliptical.image_2d_from(grid=grid)

    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
