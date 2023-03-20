from typing import List

import autoarray as aa

from autogalaxy.analysis.result import ResultDataset
from autogalaxy.plane.plane import Plane
from autogalaxy.imaging.fit_imaging import FitImaging


class ResultImaging(ResultDataset):
    """
    After the non-linear search of a fit to an imaging dataset is complete it creates this `ResultImaging`, object
    which includes:

    - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
    the maximum likelihood model, posteriors and other properties.

    - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
    an instance of the maximum log likelihood model).

    - The non-linear search used to perform the model fit.

    This class contains a number of methods which use the above objects to create the max log likelihood `Plane`,
    `FitImaging`, hyper-galaxy images,etc.

    Parameters
    ----------
    samples
        A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
        run of samples of the nested sampler.
    model
        The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
        the imaging data.
    search
        The non-linear search used to perform this model-fit.

    Returns
    -------
    ResultImaging
        The result of fitting the model to the imaging dataset, via a non-linear search.
    """

    @property
    def max_log_likelihood_fit(self) -> FitImaging:
        """
        An instance of a `FitImaging` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """
        hyper_image_sky = self.analysis.hyper_image_sky_via_instance_from(
            instance=self.instance_copy
        )

        hyper_background_noise = self.analysis.hyper_background_noise_via_instance_from(
            instance=self.instance_copy
        )

        instance = self.analysis.instance_with_associated_hyper_images_from(
            instance=self.instance_copy
        )

        plane = self.analysis.plane_via_instance_from(instance=instance)

        return self.analysis.fit_imaging_via_plane_from(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def max_log_likelihood_plane(self) -> Plane:
        """
        An instance of a `Plane` corresponding to the maximum log likelihood model inferred by the non-linear search.

        The `Plane` is computed from the `max_log_likelihood_fit`, as this ensures that all linear light profiles
        are converted to normal light profiles with their `intensity` values updated.
        """
        return (
            self.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
        )

    @property
    def unmasked_model_image(self) -> aa.Array2D:
        """
        The model image of the maximum log likelihood model, creating without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_galaxies(self) -> List[aa.Array2D]:
        """
        A list of the model image of every galaxy in the maximum log likelihood model, whereas all images are created
        without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_galaxies_list
