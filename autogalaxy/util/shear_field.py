import logging
from matplotlib.patches import Ellipse
import numpy as np
import typing

import autoarray as aa

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractShearField:
    @property
    def ellipticities(self) -> aa.ArrayIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the galaxy ellipticity each vector
        corresponds too.
        """
        return aa.ArrayIrregular(
            values=np.sqrt(self.slim[:, 0] ** 2 + self.slim[:, 1] ** 2.0)
        )

    @property
    def semi_major_axes(self) -> aa.ArrayIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-major axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return aa.ArrayIrregular(values=3 * (1 + self.ellipticities))

    @property
    def semi_minor_axes(self) -> aa.ArrayIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-minor axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return aa.ArrayIrregular(values=3 * (1 - self.ellipticities))

    @property
    def phis(self) -> aa.ArrayIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the position angle defined
        counter clockwise from the positive x-axis of each galaxy ellipticity that each vector corresponds too.
        """
        return aa.ArrayIrregular(
            values=np.arctan2(self.slim[:, 0], self.slim[:, 1]) * 180.0 / np.pi / 2.0
        )

    @property
    def elliptical_patches(self) -> typing.List[Ellipse]:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the elliptical patch representing
        each galaxy ellipticity. This patch is used for visualizing an ellipse of each galaxy in an image.
        """

        return [
            Ellipse(
                xy=(x, y), width=semi_major_axis, height=semi_minor_axis, angle=angle
            )
            for x, y, semi_major_axis, semi_minor_axis, angle in zip(
                self.grid.slim[:, 1],
                self.grid.slim[:, 0],
                self.semi_major_axes,
                self.semi_minor_axes,
                self.phis,
            )
        ]


class ShearYX2D(aa.VectorYX2D, AbstractShearField):
    """
    A regular shear field, which is collection of (y,x) vectors which are located on a uniform grid
    of (y,x) coordinates.

    The structure of this data structure is described in
    `autoarray.structures.vectors.uniform.VectorYX2D`

    This class extends `VectorYX2D` to include methods that are specific to a shear field, typically
    used for weak lensing calculations.

    This data structure is of shape [total_vectors, 2], where entries for [:,0] are the gamma_2 values and
    entries for [:,1] are the gamma_1 values.

    Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
    entries are the gamma_1 values.

    Parameters
    ----------
    vectors
        The 2D (y,x) vectors on an irregular grid that represent the vector-field.
    grid
        The irregular grid of (y,x) coordinates where each vector is located.
    """


class ShearYX2DIrregular(aa.VectorYX2DIrregular, AbstractShearField):
    """
    An irregular shear field, which is collection of (y,x) vectors which are located on an irregular grid
    of (y,x) coordinates.

    The structure of this data structure is described in
    `autoarray.structures.vectors.irregular.VectorYX2DIrregular`

    This class extends `VectorYX2DIrregular` to include methods that are specific to a shear field, typically
    used for weak lensing calculations.


    This data structure is of shape [total_vectors, 2], where entries for [:,0] are the gamma_2 values and
    entries for [:,1] are the gamma_1 values.

    Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
    entries are the gamma_1 values.

    Parameters
    ----------
    vectors
        The 2D (y,x) vectors on an irregular grid that represent the vector-field.
    grid
        The irregular grid of (y,x) coordinates where each vector is located.
    """
