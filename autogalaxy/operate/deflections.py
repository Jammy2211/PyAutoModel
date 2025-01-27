from functools import wraps
import logging
import numpy as np
from typing import List, Tuple, Union

from autoconf import conf

import autoarray as aa

from autogalaxy.util.shear_field import ShearYX2D
from autogalaxy.util.shear_field import ShearYX2DIrregular

logger = logging.getLogger(__name__)


def grid_scaled_2d_for_marching_squares_from(
    grid_pixels_2d: aa.Grid2D,
    shape_native: Tuple[int, int],
    mask: aa.Mask2D,
) -> aa.Grid2DIrregular:
    pixel_scales = mask.pixel_scales
    origin = mask.origin

    grid_scaled_1d = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels_2d,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_scaled_1d[:, 0] -= pixel_scales[0] / 2.0
    grid_scaled_1d[:, 1] += pixel_scales[1] / 2.0

    return aa.Grid2DIrregular(values=grid_scaled_1d)


def precompute_jacobian(func):
    @wraps(func)
    def wrapper(lensing_obj, grid, jacobian=None):
        if jacobian is None:
            jacobian = lensing_obj.jacobian_from(grid=grid)

        return func(lensing_obj, grid, jacobian)

    return wrapper


def evaluation_grid(func):
    @wraps(func)
    def wrapper(
        lensing_obj, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        if hasattr(grid, "is_evaluation_grid"):
            if grid.is_evaluation_grid:
                return func(lensing_obj, grid, pixel_scale)

        pixel_scale_ratio = grid.pixel_scale / pixel_scale

        zoom_shape_native = grid.mask.zoom_shape_native
        shape_native = (
            int(pixel_scale_ratio * zoom_shape_native[0]),
            int(pixel_scale_ratio * zoom_shape_native[1]),
        )

        max_evaluation_grid_size = conf.instance["general"]["grid"][
            "max_evaluation_grid_size"
        ]

        # This is a hack to prevent the evaluation gird going beyond 1000 x 1000 pixels, which slows the code
        # down a lot. Need a better moe robust way to set this up for any general lens.

        if shape_native[0] > max_evaluation_grid_size:
            pixel_scale = pixel_scale_ratio / (
                shape_native[0] / float(max_evaluation_grid_size)
            )
            shape_native = (max_evaluation_grid_size, max_evaluation_grid_size)

        grid = aa.Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=(pixel_scale, pixel_scale),
            origin=grid.mask.zoom_offset_scaled,
        )

        grid.is_evaluation_grid = True

        return func(lensing_obj, grid, pixel_scale)

    return wrapper


class OperateDeflections:
    """
    Packages methods which manipulate the 2D deflection angle map returned from the `deflections_yx_2d_from` function
    of a mass object (e.g. a `MassProfile`, `Galaxy`).

    The majority of methods are those which from the 2D deflection angle map compute lensing quantities like a 2D
    shear field, magnification map or the Einstein Radius.

    The methods in `CalcLens` are passed to the mass object to provide a concise API.

    Parameters
    ----------
    deflections_yx_2d_from
        The function which returns the mass object's 2D deflection angles.
    """

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    @precompute_jacobian
    def tangential_eigen_value_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the tangential eigen values of lensing jacobian, which are given by the expression:

        `tangential_eigen_value = 1 - convergence - shear`

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = self.convergence_2d_via_jacobian_from(
            grid=grid, jacobian=jacobian
        )

        shear_yx = self.shear_yx_2d_via_jacobian_from(grid=grid, jacobian=jacobian)

        return aa.Array2D(values=1 - convergence - shear_yx.magnitudes, mask=grid.mask)

    @precompute_jacobian
    def radial_eigen_value_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the radial eigen values of lensing jacobian, which are given by the expression:

        radial_eigen_value = 1 - convergence + shear

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = self.convergence_2d_via_jacobian_from(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_yx_2d_via_jacobian_from(grid=grid, jacobian=jacobian)

        return aa.Array2D(values=1 - convergence + shear.magnitudes, mask=grid.mask)

    def magnification_2d_from(self, grid) -> aa.Array2D:
        """
        Returns the 2D magnification map of lensing object, which is computed as the inverse of the determinant of the
        jacobian.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        """
        jacobian = self.jacobian_from(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return aa.Array2D(values=1 / det_jacobian, mask=grid.mask)

    def hessian_from(self, grid, buffer: float = 0.01, deflections_func=None) -> Tuple:
        """
        Returns the Hessian of the lensing object, where the Hessian is the second partial derivatives of the
        potential (see equation 55 https://inspirehep.net/literature/419263):

        `hessian_{i,j} = d^2 / dtheta_i dtheta_j`

        The Hessian is computed by evaluating the 2D deflection angles around every (y,x) coordinate on the input 2D
        grid map in four directions (positive y, negative y, positive x, negative x), exploiting how the deflection
        angles are the derivative of the potential.

        By using evaluating the deflection angles around each grid coordinate, the Hessian can therefore be computed
        using uniform or irregular 2D grids of (y,x). This can be slower, because x4 more deflection angle calculations
        are required, however it is more flexible in and therefore used throughout **PyAutoLens** by default.

        The Hessian is returned as a 4 entry tuple, which reflect its structure as a 2x2 matrix.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """
        if deflections_func is None:
            deflections_func = self.deflections_yx_2d_from

        grid_shift_y_up = aa.Grid2DIrregular(values=np.zeros(grid.shape))
        grid_shift_y_up[:, 0] = grid[:, 0] + buffer
        grid_shift_y_up[:, 1] = grid[:, 1]

        grid_shift_y_down = aa.Grid2DIrregular(values=np.zeros(grid.shape))
        grid_shift_y_down[:, 0] = grid[:, 0] - buffer
        grid_shift_y_down[:, 1] = grid[:, 1]

        grid_shift_x_left = aa.Grid2DIrregular(values=np.zeros(grid.shape))
        grid_shift_x_left[:, 0] = grid[:, 0]
        grid_shift_x_left[:, 1] = grid[:, 1] - buffer

        grid_shift_x_right = aa.Grid2DIrregular(values=np.zeros(grid.shape))
        grid_shift_x_right[:, 0] = grid[:, 0]
        grid_shift_x_right[:, 1] = grid[:, 1] + buffer

        deflections_up = deflections_func(grid=grid_shift_y_up)
        deflections_down = deflections_func(grid=grid_shift_y_down)
        deflections_left = deflections_func(grid=grid_shift_x_left)
        deflections_right = deflections_func(grid=grid_shift_x_right)

        hessian_yy = 0.5 * (deflections_up[:, 0] - deflections_down[:, 0]) / buffer
        hessian_xy = 0.5 * (deflections_up[:, 1] - deflections_down[:, 1]) / buffer
        hessian_yx = 0.5 * (deflections_right[:, 0] - deflections_left[:, 0]) / buffer
        hessian_xx = 0.5 * (deflections_right[:, 1] - deflections_left[:, 1]) / buffer

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def convergence_2d_via_hessian_from(
        self, grid, buffer: float = 0.01
    ) -> aa.ArrayIrregular:
        """
        Returns the convergence of the lensing object, which is computed from the 2D deflection angle map via the
        Hessian using the expression (see equation 56 https://inspirehep.net/literature/419263):

        `convergence = 0.5 * (hessian_{0,0} + hessian_{1,1}) = 0.5 * (hessian_xx + hessian_yy)`

        By going via the Hessian, the convergence can be calculated at any (y,x) coordinate therefore using either a
        2D uniform or irregular grid.

        This calculation of the convergence is independent of analytic calculations defined within `MassProfile` objects
        and can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer
        )

        return aa.ArrayIrregular(values=0.5 * (hessian_yy + hessian_xx))

    def shear_yx_2d_via_hessian_from(
        self, grid, buffer: float = 0.01
    ) -> ShearYX2DIrregular:
        """
        Returns the 2D (y,x) shear vectors of the lensing object, which are computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 57 https://inspirehep.net/literature/419263):

        `shear_y = hessian_{1,0} =  hessian_{0,1} = hessian_yx = hessian_xy`
        `shear_x = 0.5 * (hessian_{0,0} - hessian_{1,1}) = 0.5 * (hessian_xx - hessian_yy)`

        By going via the Hessian, the shear vectors can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the shear vectors is independent of analytic calculations defined within `MassProfile`
        objects and can therefore be used as a cross-check.

        The result is returned as a `ShearYX2D` dats structure, which has shape [total_shear_vectors, 2], where
        entries for [:,0] are the gamma_2 values and entries for [:,1] are the gamma_1 values.

        Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
        entries are the gamma_1 values.

        Parameters
        ----------
        grids
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer
        )

        gamma_1 = 0.5 * (hessian_xx - hessian_yy)
        gamma_2 = hessian_xy

        shear_yx_2d = np.zeros(shape=(grid.shape_slim, 2))

        shear_yx_2d[:, 0] = gamma_2
        shear_yx_2d[:, 1] = gamma_1

        return ShearYX2DIrregular(values=shear_yx_2d, grid=grid)

    def magnification_2d_via_hessian_from(
        self, grid, buffer: float = 0.01, deflections_func=None
    ) -> aa.ArrayIrregular:
        """
        Returns the 2D magnification map of lensing object, which is computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 60 https://inspirehep.net/literature/419263):

        `magnification = 1.0 / det(Jacobian) = 1.0 / abs((1.0 - convergence)**2.0 - shear**2.0)`
        `magnification = (1.0 - hessian_{0,0}) * (1.0 - hessian_{1, 1)) - hessian_{0,1}*hessian_{1,0}`
        `magnification = (1.0 - hessian_xx) * (1.0 - hessian_yy)) - hessian_xy*hessian_yx`

        By going via the Hessian, the magnification can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the magnification is independent of calculations using the Jacobian and can therefore be
        used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer, deflections_func=deflections_func
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.ArrayIrregular(values=1.0 / det_A)

    def contour_list_from(self, grid, contour_array):
        grid_contour = aa.Grid2DContour(
            grid=grid,
            pixel_scales=grid.pixel_scales,
            shape_native=grid.shape_native,
            contour_array=contour_array.native,
        )

        return grid_contour.contour_list

    @evaluation_grid
    def tangential_critical_curve_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential critical curves of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        tangential_eigen_values = self.tangential_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=tangential_eigen_values)

    @evaluation_grid
    def radial_critical_curve_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial critical curves of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        radial_eigen_values = self.radial_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=radial_eigen_values)

    @evaluation_grid
    def tangential_caustic_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential caustics of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the tangential critical curve
           contours and ray-trace it to the source-plane, therefore forming the tangential caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        tangential_caustic_list = []

        for tangential_critical_curve in tangential_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=tangential_critical_curve
            )

            tangential_caustic_list.append(
                tangential_critical_curve - deflections_critical_curve
            )

        return tangential_caustic_list

    @evaluation_grid
    def radial_caustic_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial caustics of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the radial critical curve
           contours and ray-trace it to the source-plane, therefore forming the radial caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        radial_critical_curve_list = self.radial_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        radial_caustic_list = []

        for radial_critical_curve in radial_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=radial_critical_curve
            )

            radial_caustic_list.append(
                radial_critical_curve - deflections_critical_curve
            )

        return radial_caustic_list

    @evaluation_grid
    def radial_critical_curve_area_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float]
    ) -> List[float]:
        """
        Returns the surface area within each radial critical curve as a list, the calculation of which is described in
        the function `radial_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.


        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the radial critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        radial_critical_curve_list = self.radial_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        return self.area_within_curve_list_from(curve_list=radial_critical_curve_list)

    @evaluation_grid
    def tangential_critical_curve_area_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[float]:
        """
        Returns the surface area within each tangential critical curve as a list, the calculation of which is
        described in the function `tangential_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        return self.area_within_curve_list_from(
            curve_list=tangential_critical_curve_list
        )

    def area_within_curve_list_from(
        self, curve_list: List[aa.Grid2DIrregular]
    ) -> List[float]:
        area_within_each_curve_list = []

        for curve in curve_list:
            x, y = curve[:, 0], curve[:, 1]
            area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))
            area_within_each_curve_list.append(area)

        return area_within_each_curve_list

    @evaluation_grid
    def einstein_radius_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        """
        Returns a list of the Einstein radii corresponding to the area within each tangential critical curve.

        Each Einstein radius is defined as the radius of the circle which contains the same area as the area within
        each tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        try:
            area_list = self.tangential_critical_curve_area_list_from(
                grid=grid, pixel_scale=pixel_scale
            )
            return [np.sqrt(area / np.pi) for area in area_list]
        except TypeError:
            raise TypeError("The grid input was unable to estimate the Einstein Radius")

    @evaluation_grid
    def einstein_radius_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The Einstein radius is defined as the radius of the circle which contains the same area as the area within
        the tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        If there are multiple tangential critical curves (e.g. because the mass distribution is complex) this function
        raises an error, and the function `einstein_radius_list_from()` should be used instead.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential
            critical curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        einstein_radii_list = self.einstein_radius_list_from(grid=grid)

        if len(einstein_radii_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein radius is the sum of
                all of them. Check the `einstein_radius_list_from` function for the individual Einstein.
            """
            )

        return sum(einstein_radii_list)

    @evaluation_grid
    def einstein_mass_angular_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[float]:
        """
        Returns a list of the angular Einstein massses corresponding to the area within each tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_radius_list = self.einstein_radius_list_from(
            grid=grid, pixel_scale=pixel_scale
        )
        return [
            np.pi * einstein_radius**2 for einstein_radius in einstein_radius_list
        ]

    @evaluation_grid
    def einstein_mass_angular_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> float:
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_mass_angular_list = self.einstein_mass_angular_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        if len(einstein_mass_angular_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein mass is the sum of
                all of them. Check the `einstein_mass_list_from` function for the individual Einstein.
            """
            )

        return einstein_mass_angular_list[0]

    def jacobian_from(self, grid, plane_i=0, plane_j=-1):
        """
        Returns the Jacobian of the lensing object, which is computed by taking the gradient of the 2D deflection
        angle map in four direction (positive y, negative y, positive x, negative x).

        By using the `np.gradient` method the Jacobian can therefore only be computed using uniform 2D grids of (y,x)
        coordinates, and does not support irregular grids. For this reason, calculations by default use the Hessian,
        which is slower to compute because more deflection angle calculations are necessary but more flexible in
        general.

        The Jacobian is returned as a list of lists, which reflect its structure as a 2x2 matrix.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        """

        #deflections = self.deflections_yx_2d_from(grid=grid)
        deflections = self.deflections_between_planes_from(grid=grid, plane_i=plane_i, plane_j=plane_j)

        # TODO : Can probably make this work on irregular grid? Is there any point?

        a11 = aa.Array2D(
            values=1.0
            - np.gradient(deflections.native[:, :, 1], grid.native[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a12 = aa.Array2D(
            values=-1.0
            * np.gradient(deflections.native[:, :, 1], grid.native[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        a21 = aa.Array2D(
            values=-1.0
            * np.gradient(deflections.native[:, :, 0], grid.native[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a22 = aa.Array2D(
            values=1
            - np.gradient(deflections.native[:, :, 0], grid.native[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        return [[a11, a12], [a21, a22]]

    @precompute_jacobian
    def convergence_2d_via_jacobian_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the convergence of the lensing object, which is computed from the 2D deflection angle map via the
        Jacobian using the expression (see equation 58 https://inspirehep.net/literature/419263):

        `convergence = 1.0 - 0.5 * (jacobian_{0,0} + jacobian_{1,1}) = 0.5 * (jacobian_xx + jacobian_yy)`

        By going via the Jacobian, the convergence must be calculated using 2D uniform grid.

        This calculation of the convergence is independent of analytic calculations defined within `MassProfile`
        objects and the calculation via the Hessian. It can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])

        return aa.Array2D(values=convergence, mask=grid.mask)

    @precompute_jacobian
    def shear_yx_2d_via_jacobian_from(
        self, grid, jacobian=None
    ) -> Union[ShearYX2D, ShearYX2DIrregular]:
        """
        Returns the 2D (y,x) shear vectors of the lensing object, which are computed from the 2D deflection angle map
        via the Jacobian using the expression (see equation 58 https://inspirehep.net/literature/419263):

        `shear_y = -0.5 * (jacobian_{0,1} + jacobian_{1,0} = -0.5 * (jacobian_yx + jacobian_xy)`
        `shear_x = 0.5 * (jacobian_{1,1} + jacobian_{0,0} = 0.5 * (jacobian_yy + jacobian_xx)`

        By going via the Jacobian, the convergence must be calculated using 2D uniform grid.

        This calculation of the shear vectors is independent of analytic calculations defined within `MassProfile`
        objects and the calculation via the Hessian. It can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """

        shear_yx_2d = np.zeros(shape=(grid.shape_slim, 2))
        shear_yx_2d[:, 0] = -0.5 * (jacobian[0][1] + jacobian[1][0])
        shear_yx_2d[:, 1] = 0.5 * (jacobian[1][1] - jacobian[0][0])

        if isinstance(grid, aa.Grid2DIrregular):
            return ShearYX2DIrregular(values=shear_yx_2d, grid=grid)
        return ShearYX2D(values=shear_yx_2d, grid=grid, mask=grid.mask)
