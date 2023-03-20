import numpy as np
from scipy.special import factorial, genlaguerre
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.shapelets.abstract import AbstractShapelet


class ShapeletExponentialEll(AbstractShapelet):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to an Exponential using a polar (r,theta) grid of
        coordinates.

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
        Returns the Exponential Shapelet light profile's 2D image from a 2D grid of Exponential (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Exponential Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """

        radial = (grid[:, 0] ** 2 + grid[:, 1] ** 2) / self.beta
        theta = np.arctan(grid[:, 1] / grid[:, 0])

        prefactor = (
            1.0
            / np.sqrt(2 * np.pi)
            / self.beta
            * (self.n + 0.5) ** (-1 - np.abs(self.m))
            * (-1) ** (self.n + self.m)
            * np.sqrt(
                factorial(self.n - np.abs(self.m)) / 2 * self.n
                + 1 / factorial(self.n + np.abs(self.m))
            )
        )

        laguerre = genlaguerre(n=self.n - np.abs(self.m), alpha=2 * np.abs(self.m))
        shapelet = laguerre(radial / (self.n + 0.5))

        return np.abs(
            prefactor
            * np.exp(-radial / (2 * self.n + 1))
            * radial ** (np.abs(self.m))
            * shapelet
            * np.cos(self.m * theta)
            + -1.0j * np.sin(self.m * theta)
        )


class ShapeletExponential(ShapeletExponentialEll):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Exponential (r,theta) grid of coordinates.

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
