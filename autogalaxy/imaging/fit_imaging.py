import numpy as np
from typing import Dict, List, Optional

from autoconf import conf
from autoconf import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.hyper.hyper_data import HyperImageSky
from autogalaxy.hyper.hyper_data import HyperBackgroundNoise
from autogalaxy.plane.plane import Plane
from autogalaxy.plane.to_inversion import PlaneToInversion
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear


class FitImaging(aa.FitImaging, AbstractFitInversion):
    def __init__(
        self,
        dataset: aa.Imaging,
        plane: Plane,
        hyper_image_sky: Optional[HyperImageSky] = None,
        hyper_background_noise: Optional[HyperBackgroundNoise] = None,
        use_hyper_scaling: bool = True,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Fits an imaging dataset using a `Plane` object.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the `Plane`.

        2) Blur this with the imaging PSF to created the `blurred_image`.

        3) Subtract this image from the `data` to create the `profile_subtracted_image`.

        4) If the `Plane` has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_image` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `blurred_image` and `reconstructed_data` of the inversion (if
           an inversion is not performed the `model_data` is only the `blurred_image`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including additional terms describing the linear
           algebra solution, is computed).

        When performing a `model-fit`via an `AnalysisImaging` object the `figure_of_merit` of this `FitImaging` object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The imaging dataset which is fitted by the galaxies in the plane.
        plane
            The plane of galaxies whose light profile images are used to fit the imaging data.
        hyper_image_sky
            If included, accounts for the background sky in the fit.
        hyper_background_noise
            If included, adds a noise-scaling term to the background to account for an inaacurate background sky model.
        use_hyper_scaling
            If set to False, the hyper scaling functions (e.g. the `hyper_image_sky` / `hyper_background_noise`) are
            omitted irrespective of their inputs.
        settings_pixelization
            Settings controlling how a pixelization is fitted for example if a border is used when creating the
            pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        preloads
            Contains preloaded calculations (e.g. linear algebra matrices) which can skip certain calculations in
            the fit.
        profiling_dict
            A dictionary which if passed to the fit records how long fucntion calls which have the `profile_func`
            decorator take to run.
        """

        self.plane = plane
        self.preloads = preloads

        super().__init__(
            dataset=dataset,
            profiling_dict=profiling_dict,
        )
        AbstractFitInversion.__init__(
            self=self, model_obj=plane, settings_inversion=settings_inversion
        )

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.use_hyper_scaling = use_hyper_scaling

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion

    @property
    def galaxies(self) -> List[Galaxy]:
        return self.plane.galaxies

    @property
    def grid(self) -> aa.type.Grid2DLike:
        return self.imaging.grid

    @property
    def data(self) -> aa.Array2D:
        """
        Returns the imaging data, which may have a hyper scaling performed which rescales the background sky level
        in order to account for uncertainty in the background sky subtraction.
        """
        if self.use_hyper_scaling:

            return hyper_image_from(
                image=self.dataset.image, hyper_image_sky=self.hyper_image_sky
            )

        return self.dataset.data

    @property
    def noise_map(self) -> aa.Array2D:
        """
        Returns the imaging noise-map, which may have a hyper scaling performed which increase the noise in regions of
        the data that are poorly fitted in order to avoid overfitting.
        """
        if self.use_hyper_scaling:

            return hyper_noise_map_from(
                noise_map=self.dataset.noise_map,
                model_obj=self.plane,
                hyper_background_noise=self.hyper_background_noise,
            )

        return self.dataset.noise_map

    @property
    def blurred_image(self) -> aa.Array2D:
        """
        Returns the image of the light profiles of all galaxies in the fit's plane, convolved with the
        imaging dataset's PSF.

        If the plane does not have any light profiles, the image is computed bypassing the convolution routine
        altogether.
        """

        if not self.plane.has(cls=LightProfile):

            return self.plane.image_2d_from(
                grid=self.dataset.grid,
            )

        return self.plane.blurred_image_2d_from(
            grid=self.dataset.grid,
            convolver=self.dataset.convolver,
            blurring_grid=self.dataset.blurring_grid,
        )

    @property
    def profile_subtracted_image(self) -> aa.Array2D:
        """
        Returns the dataset's image data with all blurred light profile images in the fit's plane subtracted.
        """
        return self.image - self.blurred_image

    @property
    def model_data(self) -> aa.Array2D:
        """
        Returns the model-image that is used to fit the data.

        If the plane does not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images blurred with the PSF.

        If a inversion is included it is the sum of this image and the inversion's reconstruction of the image.
        """
        return self.blurred_image

    @property
    def plane_to_inversion(self) -> PlaneToInversion:
        return PlaneToInversion(
            plane=self.plane,
            dataset=self.dataset,
            data=self.profile_subtracted_image,
            noise_map=self.noise_map,
            w_tilde=self.w_tilde,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )

    @cached_property
    def inversion(self) -> Optional[aa.AbstractInversion]:
        """
        If the plane has linear objects which are used to fit the data (e.g. a linear light profile / pixelization)
        this function returns a linear inversion, where the flux values of these objects (e.g. the `intensity`
        of linear light profiles) are computed via linear matrix algebra.

        The data passed to this function is the dataset's image with all light profile images of the plane subtracted,
        ensuring that the inversion only fits the data with ordinary light profiles subtracted.
        """

        if self.perform_inversion:
            return self.plane_to_inversion.inversion

    @property
    def model_data(self) -> aa.Array2D:
        """
        Returns the model-image that is used to fit the data.

        If the plane does not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images blurred with the PSF.

        If a inversion is included it is the sum of this image and the inversion's reconstruction of the image.
        """

        if self.perform_inversion:

            return self.blurred_image + self.inversion.mapped_reconstructed_data

        return self.blurred_image

    @property
    def galaxy_model_image_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the plane with its `model_image`.

        This image is the image of the sum of:

        - The images of all ordinary light profiles in that plane summed and convolved with the imaging data's PSF.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        For modeling, this dictionary is used to set up the `hyper_images` that adapt certain pixelizations to the
        data being fitted.
        """

        galaxy_blurred_image_2d_dict = self.plane.galaxy_blurred_image_2d_dict_from(
            grid=self.grid,
            convolver=self.imaging.convolver,
            blurring_grid=self.imaging.blurring_grid,
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_image=True
        )

        return {**galaxy_blurred_image_2d_dict, **galaxy_linear_obj_image_dict}

    @property
    def model_images_of_galaxies_list(self) -> List:
        """
        A list of the model images of each galaxy in the plane.
        """
        return list(self.galaxy_model_image_dict.values())

    @property
    def subtracted_images_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        A list of the subtracted image of every galaxy.

        A subtracted image of a galaxy is the data where all other galaxy images are subtracted from it, therefore
        showing how a galaxy appears in the data in the absence of all other galaxies.

        This is used to visualize the contribution of each galaxy in the data.
        """

        subtracted_images_of_galaxies_list = []

        model_images_of_galaxies_list = self.model_images_of_galaxies_list

        for galaxy_index in range(len(self.galaxies)):

            other_galaxies_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_galaxies_list)
                if i != galaxy_index
            ]

            subtracted_image = self.image - sum(other_galaxies_model_images)

            subtracted_images_of_galaxies_list.append(subtracted_image)

        return subtracted_images_of_galaxies_list

    @property
    def unmasked_blurred_image(self) -> aa.Array2D:
        """
        The blurred image of the overall fit that would be evaluated without a mask being used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.plane.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.plane.unmasked_blurred_image_2d_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def unmasked_blurred_image_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        The blurred image of every galaxy int he plane used in this fit, that would be evaluated without a mask being
        used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.plane.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.plane.unmasked_blurred_image_2d_list_from(
            grid=self.grid, psf=self.imaging.psf
        )

    @property
    def plane_linear_light_profiles_to_light_profiles(self) -> Plane:
        """
        The `Plane` where all linear light profiles have been converted to ordinary light profiles, where their
        `intensity` values are set to the values inferred by this fit.

        This is typically used for visualization, because linear light profiles cannot be used in `LightProfilePlotter`
        or `GalaxyPlotter` objects.
        """
        return self.model_obj_linear_light_profiles_to_light_profiles

    def refit_with_new_preloads(
        self,
        preloads: Preloads,
        settings_inversion: Optional[aa.SettingsInversion] = None,
    ) -> "FitImaging":
        """
        Returns a new fit which uses the dataset, plane and other objects of this fit, but uses a different set of
        preloads input into this function.

        This is used when setting up the preloads objects, to concisely test how using different preloads objects
        changes the attributes of the fit.

        Parameters
        ----------
        preloads
            The new preloads which are used to refit the data using the
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

        Returns
        -------
        A new fit which has used new preloads input into this function but the same dataset, plane and other settings.
        """
        profiling_dict = {} if self.profiling_dict is not None else None

        settings_inversion = (
            self.settings_inversion
            if settings_inversion is None
            else settings_inversion
        )

        return FitImaging(
            dataset=self.imaging,
            plane=self.plane,
            hyper_image_sky=self.hyper_image_sky,
            hyper_background_noise=self.hyper_background_noise,
            use_hyper_scaling=self.use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )


def hyper_image_from(image: aa.Array2D, hyper_image_sky: HyperImageSky) -> aa.Array2D:
    """
    Returns a `hyper_image` from the input data's `image` used in a fit.

    The `hyper_image` is the image with its background sky subtraction changed based on the `hyper_image_sky` object,
    which scales the sky level using a `sky_scale` parameter.

    Parameters
    ----------
    image
        The image of the data whose background sky is changed.
    hyper_image_sky
        The model image describing how much the background sky level is scaled.

    Returns
    -------
    The image whose background sky is scaled.
    """
    if hyper_image_sky is not None:
        return hyper_image_sky.hyper_image_from(image=image)
    return image


def hyper_noise_map_from(
    noise_map: aa.Array2D, model_obj, hyper_background_noise: HyperBackgroundNoise
) -> aa.Array2D:
    """
    Returns a `hyper_noise_map` from the input data's `noise_map` used in a fit.

    The `hyper_noise_map` is the noise-map with a background value added or subtracted from it, based on
    the `hyper_background_noise` object, which scales the noise level using a `noise_scale` parameter.

    Parameters
    ----------
    image
        The image of the data whose background sky is changed.
    hyper_image_sky
        The model image describing how much the background sky level is scaled.

    Returns
    -------
    The image whose background sky is scaled.
    """
    hyper_noise_map = model_obj.hyper_noise_map_from(noise_map=noise_map)

    if hyper_background_noise is not None:
        noise_map = hyper_background_noise.hyper_noise_map_from(noise_map=noise_map)

    if hyper_noise_map is not None:
        noise_map = noise_map + hyper_noise_map
        noise_map_limit = conf.instance["general"]["hyper"]["hyper_noise_limit"]
        noise_map[noise_map > noise_map_limit] = noise_map_limit

    return noise_map
