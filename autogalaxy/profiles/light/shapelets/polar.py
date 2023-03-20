import numpy as np
from scipy.special import factorial, genlaguerre
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.shapelets.abstract import AbstractShapelet


class ShapeletPolarEll(AbstractShapelet):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n
            The n order of the shapelets basis function.
        m
            The m order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n = n
        self.m = m

        super().__init__(centre=centre, ell_comps=ell_comps, beta=beta)

    @aa.grid_dec.grid_2d_to_structure
    @check_operated_only
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns the Polar Shapelet light profile's 2D image from a 2D grid of Polar (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Polar Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """

        radial = (grid[:, 0] ** 2 + grid[:, 1] ** 2) / self.beta**2.0
        theta = np.arctan(grid[:, 1] / grid[:, 0])

        laguerre = genlaguerre(n=(self.n - np.abs(self.m)) / 2.0, alpha=np.abs(self.m))

        shapelet = laguerre(radial)

        const = (
            ((-1) ** ((self.n - np.abs(self.m)) / 2))
            * np.sqrt(
                factorial((self.n - np.abs(self.m)) / 2)
                / factorial((self.n + np.abs(self.m)) / 2)
            )
            / self.beta
            / np.sqrt(np.pi)
        )
        gauss = np.exp(-radial / 2.0)

        return np.abs(
            const
            * radial ** (np.abs(self.m / 2.0))
            * shapelet
            * gauss
            * np.exp(0.0 + 1j * -self.m * theta)
        )


class ShapeletPolar(ShapeletPolarEll):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n_y
            The order of the shapelets basis function in the y-direction.
        n_x
            The order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(n=n, m=m, centre=centre, ell_comps=(0.0, 0.0), beta=beta)
