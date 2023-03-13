import math

import json
import jax.numpy as np
from typing import Dict, List, Optional, Tuple, Type

import autoarray as aa

from autoconf.dictable import Dictable

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.hyper import HyperGalaxy
from autogalaxy.profiles.light.basis import Basis
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.snr import LightProfileSNR
from autogalaxy.operate.image import OperateImageGalaxies
from autogalaxy.operate.deflections import OperateDeflections

from autogalaxy import exc
from autogalaxy.util import plane_util


class Plane(OperateImageGalaxies, OperateDeflections, Dictable):
    def __init__(
        self,
        galaxies,
        redshift: Optional[float] = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        A plane of galaxies where all galaxies are at the same redshift.

        Parameters
        ----------
        redshift or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of galaxies in this plane.
        """

        if redshift is None:

            if not galaxies:
                raise exc.PlaneException(
                    "No redshift and no galaxies were input to a Plane. A redshift for the Plane therefore cannot be"
                    "determined"
                )
            redshift = sum(galaxy.redshift for galaxy in galaxies) / len(galaxies)

        self.redshift = redshift
        self.galaxies = galaxies

        self.profiling_dict = profiling_dict

    def dict(self) -> Dict:
        plane_dict = super().dict()
        plane_dict["galaxies"] = [galaxy.dict() for galaxy in self.galaxies]
        return plane_dict

    def output_to_json(self, file_path: str):

        with open(file_path, "w+") as f:
            json.dump(self.dict(), f, indent=4)

    @property
    def galaxy_redshifts(self) -> List[float]:
        return [galaxy.redshift for galaxy in self.galaxies]

    def has(self, cls: Tuple[Type]) -> bool:
        if self.galaxies is not None:
            return any(list(map(lambda galaxy: galaxy.has(cls=cls), self.galaxies)))
        return False

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the plane which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the plane is returned.

        Returns
        -------
            The list of objects in the plane that inherit from input `cls`.
        """
        cls_list = []

        for galaxy in self.galaxies:
            if galaxy.has(cls=cls):
                for cls_galaxy in galaxy.cls_list_from(cls=cls):
                    cls_list.append(cls_galaxy)

        return cls_list

    def galaxies_with_cls_list_from(self, cls: Type) -> List[Galaxy]:
        return list(filter(lambda galaxy: galaxy.has(cls=cls), self.galaxies))

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Returns the profile-image plane image of the list of galaxies of the plane's sub-grid, by summing the
        individual images of each galaxy's light profile.

        If the `operated_only` input is included, the function omits light profiles which are parents of
        the `LightProfileOperated` object, which signifies that the light profile represents emission that has
        already had the instrument operations (e.g. PSF convolution, a Fourier transform) applied to it.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            By default, the image is the sum of light profile images (irrespective of whether they have been operatd on
            or not). If this input is included as a bool, only images which are or are not already operated are summed
            and returned.
        """
        if self.galaxies:
            return sum(self.image_2d_list_from(grid=grid, operated_only=operated_only))
        return np.zeros((grid.shape[0],))

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        return [
            galaxy.image_2d_from(grid=grid, operated_only=operated_only)
            for galaxy in self.galaxies
        ]

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> {Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object in the `Plane` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for hyper-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy in the plane with its corresponding 2D image.
        """

        galaxy_image_2d_dict = dict()

        image_2d_list = self.image_2d_list_from(grid=grid, operated_only=operated_only)

        for (galaxy_index, galaxy) in enumerate(self.galaxies):
            galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict

    def plane_image_2d_from(self, grid: aa.type.Grid2DLike) -> "PlaneImage":
        return plane_util.plane_image_of_galaxies_from(
            shape=grid.mask.shape,
            grid=grid.mask.derive_grid.all_false_sub_1,
            galaxies=self.galaxies,
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        if self.galaxies:
            return sum(
                map(lambda g: g.deflections_yx_2d_from(grid=grid), self.galaxies)
            )
        return np.zeros(shape=(grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the convergence of the list of galaxies of the plane's sub-grid, by summing the individual convergences \
        of each galaxy's mass profile.

        The convergence is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

        Parameters
        ----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self.galaxies:
            return sum(map(lambda g: g.convergence_2d_from(grid=grid), self.galaxies))
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the potential of the list of galaxies of the plane's sub-grid, by summing the individual potentials \
        of each galaxy's mass profile.

        The potential is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

        Parameters
        ----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self.galaxies:
            return sum(map(lambda g: g.potential_2d_from(grid=grid), self.galaxies))
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def traced_grid_from(self, grid: aa.type.Grid2DLike) -> aa.type.Grid2DLike:
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""
        return grid - self.deflections_yx_2d_from(grid=grid)

    @property
    def hyper_galaxies_with_pixelization_image_list(self) -> List[aa.Array2D]:
        return [
            galaxy.hyper_galaxy_image
            for galaxy in self.galaxies_with_cls_list_from(cls=aa.Pixelization)
        ]

    def hyper_noise_map_from(self, noise_map) -> aa.Array2D:
        hyper_noise_maps = self.hyper_noise_map_list_from(noise_map=noise_map)
        return sum(hyper_noise_maps)

    def hyper_noise_map_list_from(self, noise_map) -> List[aa.Array2D]:
        """
        For a contribution map and noise-map, use the model hyper_galaxy galaxies to compute a hyper noise-map.

        Parameters
        ----------
        noise_map : imaging.NoiseMap or ndarray
            An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        hyper_noise_map_list = []

        for galaxy in self.galaxies:

            if galaxy.has(cls=HyperGalaxy):

                contribution_map = galaxy.hyper_galaxy.contribution_map_from(
                    hyper_model_image=galaxy.hyper_model_image,
                    hyper_galaxy_image=galaxy.hyper_galaxy_image,
                )

                hyper_noise_map = galaxy.hyper_galaxy.hyper_noise_map_from(
                    contribution_map=contribution_map, noise_map=noise_map,
                )

                hyper_noise_map_list.append(hyper_noise_map)

            else:

                hyper_noise_map = aa.Array2D(
                    values=np.zeros(noise_map.mask.derive_mask.sub_1.pixels_in_mask),
                    mask=noise_map.mask.derive_mask.sub_1,
                )

                hyper_noise_map_list.append(hyper_noise_map)

        return hyper_noise_map_list

    @property
    def contribution_map(self) -> Optional[aa.Array2D]:

        contribution_map_list = self.contribution_map_list

        contribution_map_list = [i for i in contribution_map_list if i is not None]

        if contribution_map_list:
            return sum(contribution_map_list)
        else:
            return None

    @property
    def contribution_map_list(self) -> List[aa.Array2D]:

        contribution_map_list = []

        for galaxy in self.galaxies:

            if galaxy.hyper_galaxy is not None:

                contribution_map_list.append(galaxy.contribution_map)

            else:

                contribution_map_list.append(None)

        return contribution_map_list

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies in the `model_obj` have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        if self.has(cls=(aa.Pixelization, LightProfileLinear)):
            return True
        elif self.has(cls=Basis):
            basis_list = self.cls_list_from(cls=Basis)
            for basis in basis_list:
                for light_profile in basis.light_profile_list:
                    if isinstance(light_profile, LightProfileLinear):
                        return True

        return False

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in `Plane` as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a plane has a galaxy which two light profiles and we want its axis-ratios, the following:

        `plane.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want their centres, the following:

        `plane.extract_attribute(cls=MassProfile, name="centres")`

        would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """

        def extract(value, name):

            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for galaxy in self.galaxies
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    def extract_attributes_of_galaxies(self, cls, attr_name, filter_nones=False):
        """
        Returns an attribute of a class in the plane as a list of `ValueIrregular` or `Grid2DIrregular` objects,
        where the list indexes correspond to each galaxy in the plane..

        For example, if a plane has two galaxies which each have a light profile the following:

        `plane.extract_attributes_of_galaxies(cls=LightProfile, name="axis_ratio")`

        would return:

        [ArrayIrregular(values=[axis_ratio_0]), ArrayIrregular(values=[axis_ratio_1])]

        If a plane has two galaxies, the first with a mass profile and the second with two mass profiles ,the following:

        `plane.extract_attributes_of_galaxies(cls=MassProfile, name="centres")`

        would return:
        [
            Grid2DIrregular(values=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(values=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)])
        ]

        If a Profile does not have a certain entry, it is replaced with a None. Nones can be removed by
        setting `filter_nones=True`.

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """
        if filter_nones:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
                if galaxy.extract_attribute(cls=cls, attr_name=attr_name) is not None
            ]

        else:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
            ]

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Optional[aa.Kernel2D] = None,
    ):
        """
        Iterate over every `LightProfileSNR` in the plane and set their `intensity` values to values which give
        their input `signal_to_noise_ratio` value, which is performed as follows:

        - Evaluate the image of each light profile on the input grid.
        - Blur this image with a PSF, if included.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
          what value of `intensity` gives the desired signal to noise ratio for the image.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            The psf of the simulated imaging which can change the S/N of the light profile due to spreading out
            the emission.
        """
        for galaxy in self.galaxies:
            for light_profile in galaxy.cls_list_from(cls=LightProfile):
                if isinstance(light_profile, LightProfileSNR):
                    light_profile.set_intensity_from(
                        grid=grid,
                        exposure_time=exposure_time,
                        background_sky_level=background_sky_level,
                        psf=psf,
                    )


class PlaneImage:
    def __init__(self, array, grid):
        self.array = array
        self.grid = grid
