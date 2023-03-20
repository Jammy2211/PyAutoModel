import numpy as np
from scipy.special import hermite, factorial
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.shapelets.abstract import AbstractShapelet


class ShapeletCartesianEll(AbstractShapelet):
    def __init__(
        self,
        n_y: int,
        n_x: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Cartesian (y,x) grid of coordinates.

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
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n_y = n_y
        self.n_x = n_x

        super().__init__(centre=centre, ell_comps=ell_comps, beta=beta)

    @aa.grid_dec.grid_2d_to_structure
    @check_operated_only
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns the Cartesian Shapelet light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Cartesian Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """

        hermite_y = hermite(n=self.n_y)
        hermite_x = hermite(n=self.n_x)

        y = grid[:, 0]
        x = grid[:, 1]

        shapelet_y = hermite_y(y / self.beta)
        shapelet_x = hermite_x(x / self.beta)

        return (
            shapelet_y
            * shapelet_x
            * np.exp(-0.5 * (y**2 + x**2) / (self.beta**2))
            / self.beta
            / (
                np.sqrt(
                    2 ** (self.n_x + self.n_y)
                    * (np.pi)
                    * factorial(self.n_y)
                    * factorial(self.n_x)
                )
            )
        )


class ShapeletCartesian(ShapeletCartesianEll):
    def __init__(
        self,
        n_y: int,
        n_x: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Cartesian (y,x) grid of coordinates.

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

        super().__init__(
            n_y=n_y, n_x=n_x, centre=centre, ell_comps=(0.0, 0.0), beta=beta
        )
