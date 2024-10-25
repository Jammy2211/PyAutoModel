import os

if os.environ.get("USE_JAX", "0") == "1":
    USING_JAX = True
    import jax.numpy as np
else:
    USING_JAX = False
    import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law import PowerLaw


def psi_from(grid, axis_ratio, core_radius):
    """
    Returns the $\Psi$ term in expressions for the calculation of the deflection of an elliptical isothermal mass
    distribution. This is used in the `Isothermal` and `Chameleon` `MassProfile`'s.

    The expression for Psi is:

    $\Psi = \sqrt(q^2(s^2 + x^2) + y^2)$

    Parameters
    ----------
    grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius
        The radius of the inner core

    Returns
    -------
    float
        The value of the Psi term.

    """
    if USING_JAX:
        return np.sqrt(
            (axis_ratio**2.0 * (grid[:, 1] ** 2.0 + core_radius**2.0))
            + grid[:, 0] ** 2.0
            + 1e-16
        )
    else:
        return np.sqrt(
            np.add(
                np.multiply(
                    axis_ratio**2.0, np.add(np.square(grid[:, 1]), core_radius**2.0)
                ),
                np.square(grid[:, 0]),
            )
        )


class Isothermal(PowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope = 2.0.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    @property
    def axis_ratio(self):
        axis_ratio = super().axis_ratio
        return min(axis_ratio, 0.99999)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio**2)
        )

        psi = psi_from(grid=grid, axis_ratio=self.axis_ratio, core_radius=0.0)

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio**2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio**2), grid[:, 1]), psi)
        )
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(factor, np.vstack((deflection_y, deflection_x)).T),
            **kwargs
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def shear_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the (gamma_y, gamma_x) shear vector field on a grid of (y,x) arc-second coordinates.

        The result is returned as a `ShearYX2D` dats structure, which has shape [total_shear_vectors, 2], where
        entries for [:,0] are the gamma_2 values and entries for [:,1] are the gamma_1 values.

        Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
        entries are the gamma_1 values.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        convergence = self.convergence_2d_from(grid=grid, **kwargs)

        gamma_2 = (
            -2
            * convergence
            * np.divide(grid[:, 1] * grid[:, 0], grid[:, 1] ** 2 + grid[:, 0] ** 2)
        )
        gamma_1 = -convergence * np.divide(
            grid[:, 1] ** 2 - grid[:, 0] ** 2, grid[:, 1] ** 2 + grid[:, 0] ** 2
        )

        shear_field = self.rotated_grid_from_reference_frame_from(
            grid=np.vstack((gamma_2, gamma_1)).T, angle=self.angle * 2
        )

        return aa.VectorYX2DIrregular(values=shear_field, grid=grid)


class IsothermalSph(Isothermal):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        """
        super().__init__(
            centre=centre, ell_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    @property
    def axis_ratio(self):
        return 1.0

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.elliptical_radii_grid_from(grid=grid, **kwargs)
        return 2.0 * self.einstein_radius_rescaled * eta

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self._cartesian_grid_via_radial_from(
            grid=grid,
            radius=np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled),
            **kwargs
        )
