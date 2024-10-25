import logging
import numpy as np
from typing import Dict, Optional, Tuple

from autoconf.dictable import to_dict

import autofit as af
import autoarray as aa

from autoarray.exc import PixelizationException

from autogalaxy.analysis.adapt_images.adapt_image_maker import AdaptImageMaker
from autogalaxy.analysis.analysis.dataset import AnalysisDataset
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.interferometer.model.result import ResultInterferometer
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer
from autogalaxy import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisDataset):
    Result = ResultInterferometer
    Visualizer = VisualizerInterferometer

    def __init__(
        self,
        dataset: aa.Interferometer,
        adapt_image_maker: Optional[AdaptImageMaker] = None,
        cosmology: LensingCosmology = Planck15(),
        settings_inversion: aa.SettingsInversion = None,
        title_prefix: str = None,
    ):
        """
        Fits a galaxy model to an interferometer dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This Analysis class is used for all model-fits which fit galaxies to an interferometer dataset.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The interferometer dataset that the model is fitted too.
        adapt_image_maker
            Makes the adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings_inversion
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(
            dataset=dataset,
            adapt_image_maker=adapt_image_maker,
            cosmology=cosmology,
            settings_inversion=settings_inversion,
            title_prefix=title_prefix,
        )

    @property
    def interferometer(self):
        return self.dataset

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is
        being resumed from a previous run, and it visualizes objects which do not change throughout the model fit
        like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        super().modify_before_fit(paths=paths, model=model)

        return self

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the interferometer dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like scaling the background background noise.

        3) Extracts all galaxies from the model instance.

        4) Use the galaxies and other attributes to create a `FitInterferometer` object, which performs steps such as
           creating model images of every galaxy, transforming them to the uv-plane via a Fourier transform
           and computing residuals, a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the interferometer data.
        """

        try:
            return self.fit_from(instance=instance).figure_of_merit
        except (
            PixelizationException,
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            ValueError,
            np.linalg.LinAlgError,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def fit_from(
        self,
        instance: af.ModelInstance,
        run_time_dict: Optional[Dict] = None,
    ) -> FitInterferometer:
        """
        Given a model instance create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the interferometer data and compute
        the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        run_time_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitInterferometer
            The fit of the galaxies to the interferometer dataset, which includes the log likelihood.
        """
        galaxies = self.galaxies_via_instance_from(
            instance=instance, run_time_dict=run_time_dict
        )

        adapt_images = self.adapt_images_via_instance_from(instance=instance)

        return FitInterferometer(
            dataset=self.dataset,
            galaxies=galaxies,
            adapt_images=adapt_images,
            settings_inversion=self.settings_inversion,
            run_time_dict=run_time_dict,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
         Before the model-fit begins, this routine saves attributes of the `Analysis` object to the `files` folder
         such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The interferometer dataset (data / noise-map / uv-wavelengths / settings / etc.).
         - The real space mask defining how the images appear in real-space and used for the FFT.
         - The settings associated with the inversion.
         - The settings associated with the pixelization.
         - The Cosmology.
         - The adapt image's model image and galaxy images, if used.

         This function also outputs attributes specific to an interferometer dataset:

        - Its uv-wavelengths
        - Its real space mask.

         It is common for these attributes to be loaded by many of the template aggregator functions given in the
         `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
         the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
         output by this function.

         Parameters
         ----------
         paths
             The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
             and the pickled objects used by the aggregator output by this function.
        """
        super().save_attributes(paths=paths)

        hdu = aa.util.array_2d.hdu_for_output_from(
            array_2d=self.dataset.uv_wavelengths,
        )
        paths.save_fits(name="uv_wavelengths", hdu=hdu, prefix="dataset")
        paths.save_fits(
            name="real_space_mask",
            hdu=self.dataset.real_space_mask.hdu_for_output,
            prefix="dataset",
        )
        paths.save_json(
            "transformer_class", to_dict(self.dataset.transformer.__class__), "dataset"
        )

    def profile_log_likelihood_function(
        self, instance: af.ModelInstance, paths: Optional[af.DirectoryPaths] = None
    ) -> Tuple[Dict, Dict]:
        """
        This function is optionally called throughout a model-fit to profile the log likelihood function.

        All function calls inside the `log_likelihood_function` that are decorated with the `profile_func` are timed
        with their times stored in a dictionary called the `run_time_dict`.

        An `info_dict` is also created which stores information on aspects of the model and dataset that dictate
        run times, so the profiled times can be interpreted with this context.

        The results of this profiling are then output to hard-disk in the `profiling` folder of the model-fit results,
        which they can be inspected to ensure run-times are as expected.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.

        Returns
        -------
        Two dictionaries, the profiling dictionary and info dictionary, which contain the profiling times of the
        `log_likelihood_function` and information on the model and dataset used to perform the profiling.
        """
        run_time_dict, info_dict = super().profile_log_likelihood_function(
            instance=instance,
        )

        info_dict["number_of_visibilities"] = self.dataset.data.shape[0]
        info_dict["transformer_cls"] = self.dataset.transformer.__class__.__name__

        self.output_profiling_info(
            paths=paths, run_time_dict=run_time_dict, info_dict=info_dict
        )

        return run_time_dict, info_dict
