from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    moffat = ag.lp.Moffat(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        alpha=1.0,
        beta=1.0,
    )

    image = moffat.image_2d_from(grid=np.array([[0.0, 1.0]]))

    assert image == pytest.approx(0.5, 1e-4)

    moffat = ag.lp.Moffat(
        centre=(0.0, 0.0),
        ell_comps=(0.2, 0.2),
        intensity=1.0,
        alpha=1.8,
        beta=0.75,
    )

    image = moffat.image_2d_from(grid=np.array([[0.0, 2.0]]))

    assert image == pytest.approx(0.7340746, 1e-4)

    moffat = ag.lp.MoffatSph(centre=(0.0, 0.0), intensity=1.0, alpha=1.8, beta=0.75)

    image = moffat.image_2d_from(grid=np.array([[0.0, 2.0]]))

    assert image == pytest.approx(0.5471480213, 1e-4)
