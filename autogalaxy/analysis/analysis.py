import json
import logging
import numpy as np
from typing import Optional, Union
from os import path
import os

from autoconf import conf
import autofit as af
import autoarray as aa

from autogalaxy import exc
from autogalaxy.analysis.maker import FitMaker
from autogalaxy.analysis.preloads import Preloads
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15
from autogalaxy.hyper.hyper_data import HyperImageSky
from autogalaxy.hyper.hyper_data import HyperBackgroundNoise
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane
from autogalaxy.analysis.result import ResultDataset

from autogalaxy.analysis import model_util

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Analysis(af.Analysis):
    def __init__(self, cosmology: LensingCosmology = Planck15):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        This abstract Analysis class for all model-fits which fit galaxies (or objects containing galaxies like a
        plane), but does not perform a model-fit by itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and hyper datasets used for certain model classes.

        Parameters
        ----------
        cosmology
            The AstroPy Cosmology assumed for this analysis.
        """
        self.cosmology = cosmology

    def plane_via_instance_from(self, instance: af.ModelInstance) -> Plane:
        """
        Create a `Plane` from the galaxies contained in a model instance.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        An instance of the Plane class that is used to then fit the dataset.
        """
        if hasattr(instance, "clumps"):
            return Plane(galaxies=instance.galaxies + instance.clumps)
        return Plane(galaxies=instance.galaxies)


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        hyper_dataset_result: ResultDataset = None,
        cosmology: LensingCosmology = Planck15(),
        settings_pixelization: aa.SettingsPixelization = None,
        settings_inversion: aa.SettingsInversion = None,
    ):
        """
        Abstract Analysis class for all model-fits which fit galaxies (or objects containing galaxies like a plane)
        to a dataset, like imaging or interferometer data.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and hyper datasets used for certain model
        classes.

        Parameters
        ----------
        dataset
            The dataset that is the model is fitted too.
        hyper_dataset_result
            The hyper-model image and hyper galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings_pixelization
            settings controlling how a pixelization is fitted during the model-fit, for example if a border is used
            when creating the pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        """
        super().__init__(cosmology=cosmology)

        self.dataset = dataset
        self.hyper_dataset_result = hyper_dataset_result

        if self.hyper_dataset_result is not None:

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

        self.settings_pixelization = settings_pixelization or aa.SettingsPixelization()
        self.settings_inversion = settings_inversion or aa.SettingsInversion()

        self.preloads = self.preloads_cls()

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

        This function:

        - Checks if the model has a pixelization which uses a KMeans clustering algorithm (e.g. DelaunayBrightnessImage,
          VoronoiBrightnessImage) and makes sure that the upper limit on the prior on its `pixels` is below the number
          pixels in the mask. If it is not, the `pixels` prior upper limit is reduced.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        self.check_and_replace_hyper_images(paths=paths)

        model_util.set_upper_limit_of_pixelization_pixels_prior(
            model=model, pixels_in_mask=self.dataset.mask.pixels_in_mask
        )

    def set_hyper_dataset(self, result: ResultDataset) -> None:
        """
        Using a the result of a previous model-fit, set the hyper-dataset for this analysis. This is used to adapt
        aspects of the model (e.g. the pixelization, regularization scheme) to the properties of the dataset being
        fitted.

        This passes the hyper model image and hyper galaxy images of the previous fit. These represent where different
        galaxies in the dataset are located and thus allows the fit to adapt different aspects of the model to different
        galaxies in the data.

        Parameters
        ----------
        result
            The result of a previous model-fit which contains the model image and model galaxy images of a fit to
            the dataset, which set up the hyper dataset. These are used by certain classes for adapting the analysis
            to the properties of the dataset.
        """
        hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
        hyper_model_image = result.hyper_model_image

        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
        self.hyper_model_image = hyper_model_image

    @property
    def preloads_cls(self):
        return Preloads

    @property
    def fit_maker_cls(self):
        return FitMaker

    def set_preloads(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        It is common for the model to have components whose parameters are all fixed, and thus the way that component
        fits the data does not change. For example, if all parameter associated with the light profiles of galaxies
        in the model are fixed, the image generated from these galaxies will not change irrespective of the model
        parameters chosen by the non-linear search.

        Preloading exploits this to speed up the log likelihood function, by inspecting the model and storing in memory
        quantities that do not change. For the example above, the image of all galaxies would be stored in memory and
        to perform every fit in the `log_likelihood_funtion`.

        This function sets up all preload quantities, which are described fully in the `preloads` modules. This
        occurs directly before the non-linear search begins, to ensure the model parameterization is fixed.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        logger.info(
            "PRELOADS - Setting up preloads, may take a few minutes for fits using an inversion."
        )

        os.makedirs(paths.profile_path, exist_ok=True)

        fit_maker = self.fit_maker_cls(model=model, fit_func=self.fit_func)

        fit_0 = fit_maker.fit_via_model_from(unit_value=0.45)
        fit_1 = fit_maker.fit_via_model_from(unit_value=0.55)

        if fit_0 is None or fit_1 is None:
            self.preloads = self.preloads_cls(failed=True)
        else:
            self.preloads = self.preloads_cls.setup_all_via_fits(
                fit_0=fit_0, fit_1=fit_1
            )

            if conf.instance["general"]["test"]["check_preloads"]:

                self.preloads.check_via_fit(fit=fit_0)

        self.preloads.output_info_to_summary(file_path=paths.profile_path)

    def modify_after_fit(
        self, paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ) -> "AnalysisDataset":
        """
        Call functions that perform tasks after a model-fit is completed, for example ensuring the figure of merit
        has not changed from previous estimates and resetting preloads.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        result
            The result of the model fit that has just been completed.
        """

        self.output_or_check_figure_of_merit_sanity(paths=paths, result=result)
        self.preloads.reset_all()

        return self

    def hyper_image_sky_via_instance_from(
        self, instance: af.ModelInstance
    ) -> Optional[HyperImageSky]:
        """
        If the model instance contains a `HyperImageSky` attribute, which adds a free parameter to the model that
        scales the background sky, return this attribute. Otherwise a None is returned.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        An instance of the hyper image sky class that scales the sky background.
        """
        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    def hyper_background_noise_via_instance_from(
        self, instance: af.ModelInstance
    ) -> Optional[HyperBackgroundNoise]:
        """
        If the model instance contains a `HyperBackgroundNoise` attribute, which adds a free parameter to the model that
        scales the background noise, return this attribute. Otherwise a None is returned.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        An instance of the hyper background noise class that scales the background noise.
        """
        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def instance_with_associated_hyper_images_from(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Using the model image and galaxy images that were set up as the hyper dataset, associate the galaxy images
        of that result with the galaxies in this model fit.

        Association is performed based on galaxy names, whereby if the name of a galaxy in this search matches the
        full-path name of galaxies in the hyper dataset the galaxy image is passed.

        If the galaxy collection has a different name then an association is not made.

        For example, `galaxies.lens` will match with:
            `galaxies.lens`
        but not with:
            `galaxies.source`

        Parameters
        ----------
        instance
        An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search), which has 0 or more galaxies in its tree.

        Returns
        -------
        instance
           The input instance with images associated with galaxies where possible.
        """

        if self.hyper_galaxy_image_path_dict is not None:

            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                if galaxy_path in self.hyper_galaxy_image_path_dict:

                    galaxy.hyper_model_image = self.hyper_model_image

                    galaxy.hyper_galaxy_image = self.hyper_galaxy_image_path_dict[
                        galaxy_path
                    ]

        return instance

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be load after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The dataset's data.
        - The dataset's noise-map.
        - The settings associated with the dataset.
        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The hyper dataset's model image and galaxy images, if used.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
            and the pickled objects used by the aggregator output by this function.
        """
        paths.save_object("data", self.dataset.data)
        paths.save_object("noise_map", self.dataset.noise_map)
        paths.save_object("settings_dataset", self.dataset.settings)
        paths.save_object("settings_inversion", self.settings_inversion)
        paths.save_object("settings_pixelization", self.settings_pixelization)

        paths.save_object("cosmology", self.cosmology)

        if self.hyper_model_image is not None:
            paths.save_object("hyper_model_image", self.hyper_model_image)

        if self.hyper_galaxy_image_path_dict is not None:
            paths.save_object(
                "hyper_galaxy_image_path_dict", self.hyper_galaxy_image_path_dict
            )

    def check_and_replace_hyper_images(self, paths: af.DirectoryPaths):
        """
        Using a the result of a previous model-fit, a hyper-dataset can be set up which adapts aspects of the model
        (e.g. the pixelization, regularization scheme) to the properties of the dataset being fitted.

        If the model-fit is being resumed from a previous run, this function checks that the model image and galaxy
        images used to set up the hyper-dataset are identical to those used previously. If they are not, it replaces
        them with the previous hyper image. This ensures consistency in the log likelihood function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        """
        try:
            hyper_model_image = paths.load_object("hyper_model_image")

            if np.max(abs(hyper_model_image - self.hyper_model_image)) > 1e-8:

                logger.info(
                    "ANALYSIS - Hyper image loaded from pickle different to that set in Analysis class."
                    "Overwriting hyper images with values loaded from pickles."
                )

                self.hyper_model_image = hyper_model_image

                hyper_galaxy_image_path_dict = paths.load_object(
                    "hyper_galaxy_image_path_dict"
                )
                self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict

        except (
            FileNotFoundError,
            AttributeError,
            KeyError,
            ModuleNotFoundError,
            TypeError,
        ):
            pass

    def output_or_check_figure_of_merit_sanity(
        self, paths: af.DirectoryPaths, result: af.Result
    ):
        """
        Changes to the PyAutoGalaxy source code may inadvertantly change the numerics of how a log likelihood is
        computed. Equally, one may set off a model-fit that resumes from previous results, but change the settings of
        the pixelization or inversion in a way that changes the log likelihood function.

        This function performs an optional sanity check, which raises an exception if the log likelihood calculation
        changes, to ensure a model-fit is not resumed with a different likelihood calculation to the previous run.

        If the model-fit has not been performed before (e.g. it is not a resume) this function outputs
        the `figure_of_merit` (e.g. the log likelihood) of the maximum log likelihood model at the end of the model-fit.

        If the model-fit is a resume, it loads this `figure_of_merit` and compares it against a new value computed for
        the resumed run (again using the maximum log likelihood model inferred). If the two likelihoods do not agree
        and therefore the log likelihood function has changed, an exception is raised and the code execution terminated.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and pickled objects used by the database and aggregator.
        result
            The result containing the maximum log likelihood fit of the model.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        if conf.instance["general"]["test"]["bypass_figure_of_merit_sanity"]:
            return

        figure_of_merit = result.max_log_likelihood_fit.figure_of_merit

        figure_of_merit_sanity_file = path.join(
            paths.output_path, "figure_of_merit_sanity.json"
        )

        if not path.exists(figure_of_merit_sanity_file):

            with open(figure_of_merit_sanity_file, "w+") as f:
                json.dump(figure_of_merit, f)

        else:

            with open(figure_of_merit_sanity_file) as json_file:
                figure_of_merit_sanity = json.load(json_file)

            if conf.instance["general"]["test"]["check_figure_of_merit_sanity"]:

                if not np.isclose(figure_of_merit, figure_of_merit_sanity):

                    raise exc.AnalysisException(
                        "Figure of merit sanity check failed. "
                        ""
                        "This means that the existing results of a model fit used a different "
                        "likelihood function compared to the one implemented now.\n\n"
                        f"Old Figure of Merit = {figure_of_merit_sanity}\n"
                        f"New Figure of Merit = {figure_of_merit}"
                    )

    @property
    def fit_func(self):
        raise NotImplementedError
