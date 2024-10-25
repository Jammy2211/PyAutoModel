import os

if os.environ.get("USE_JAX", "0") == "1":
    USING_JAX = True
    import jax.numpy as np
    from .jax_utils import omega
else:
    USING_JAX = False
    import numpy as np

from scipy import special
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law_core import PowerLawCore


class PowerLaw(PowerLawCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=0.0,
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        alpha = self.deflections_yx_2d_from(aa.Grid2DIrregular(grid), **kwargs)

        alpha_x = alpha[:, 1]
        alpha_y = alpha[:, 0]

        x = grid[:, 1] - self.centre[1]
        y = grid[:, 0] - self.centre[0]

        return (x * alpha_x + y * alpha_y) / (3 - self.slope)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        slope = self.slope - 1.0
        einstein_radius = (
            2.0 / (self.axis_ratio**-0.5 + self.axis_ratio**0.5)
        ) * self.einstein_radius

        factor = np.divide(1.0 - self.axis_ratio, 1.0 + self.axis_ratio)
        b = np.multiply(einstein_radius, np.sqrt(self.axis_ratio))
        angle = np.arctan2(
            grid[:, 0], np.multiply(self.axis_ratio, grid[:, 1])
        )  # Note, this angle is not the position angle
        z = np.add(
            np.multiply(np.cos(angle), 1 + 0j), np.multiply(np.sin(angle), 0 + 1j)
        )

        if USING_JAX:
            # offset radius so calculation is finite at (0, 0)
            R = np.sqrt((self.axis_ratio * grid[:, 1]) ** 2 + grid[:, 0] ** 2 + 1e-16)
            zh = omega(z, slope, factor, n_terms=20)
        else:
            R = np.sqrt(
                np.add(
                    np.multiply(self.axis_ratio**2, grid[:, 1] ** 2), grid[:, 0] ** 2
                )
            )
            zh = z * special.hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -factor * z**2)

        complex_angle = (
            2.0 * b / (1.0 + self.axis_ratio) * (b / R) ** (slope - 1.0) * zh
        )

        deflection_y = complex_angle.imag
        deflection_x = complex_angle.real

        rescale_factor = (self.ellipticity_rescale) ** (slope - 1)

        deflection_y *= rescale_factor
        deflection_x *= rescale_factor

        return self.rotated_grid_from_reference_frame_from(
            grid=np.vstack((deflection_y, deflection_x)).T
        )

    def convergence_func(self, grid_radius: float) -> float:
        if grid_radius > 0.0:
            return self.einstein_radius_rescaled * grid_radius ** (-(self.slope - 1))
        return np.inf

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (_eta_u / u)
            * ((3.0 - slope) * _eta_u) ** -1.0
            * _eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )


class PowerLawSph(PowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        eta = self.radial_grid_from(grid=grid, **kwargs)
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled
            * np.divide(
                np.power(eta, (3.0 - self.slope)), np.multiply((3.0 - self.slope), eta)
            )
        )

        return self._cartesian_grid_via_radial_from(grid=grid, radius=deflection_r)
