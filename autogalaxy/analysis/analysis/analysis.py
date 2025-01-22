import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from os import path
import os
import time

from autoconf import conf
import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Analysis(af.Analysis):
    def __init__(self, cosmology: LensingCosmology = Planck15):
        """
        Fits a model to a dataset via a non-linear search.

        This abstract Analysis class for all model-fits which fit galaxies, but does not perform a model-fit by
        itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and adapt images used for certain model classes.

        Parameters
        ----------
        cosmology
            The Cosmology assumed for this analysis.
        """
        self.cosmology = cosmology

    def galaxies_via_instance_from(
        self, instance: af.ModelInstance, run_time_dict: Optional[Dict] = None
    ) -> List[Galaxy]:
        """
        Create a list of galaxies from a model instance, which is used to fit the dataset.

        The instance may only contain galaxies, in which case this function is redundant. However, if extra galaxies
        are included, the instance will contain both galaxies and extra galaxies, and they should be added to create
        the single list of galaxies used to fit the dataset.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        A list of galaxies that is used to then fit the dataset.
        """
        if hasattr(instance, "extra_galaxies"):
            if getattr(instance, "extra_galaxies", None) is not None:
                return Galaxies(
                    galaxies=instance.galaxies + instance.extra_galaxies,
                    run_time_dict=run_time_dict,
                )

        return Galaxies(galaxies=instance.galaxies, run_time_dict=run_time_dict)

    def dataset_model_via_instance_from(
        self, instance: af.ModelInstance
    ) -> aa.DatasetModel:
        """
        Create a dataset model from a model instance, which is used to fit the dataset.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        A dataset_model that is used to then fit the dataset.
        """
        if hasattr(instance, "dataset_model"):
            return instance.dataset_model

    def make_result(
        self,
        samples_summary: af.SamplesSummary,
        paths: af.AbstractPaths,
        samples: Optional[af.SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[af.Analysis] = None,
    ) -> af.Result:
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultImaging` object contains a number of methods which use the above objects to create the max
        log likelihood galaxies `FitImaging`, adapt-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        search
            The non-linear search used to perform this model-fit.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
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
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.

        Returns
        -------
        Two dictionaries, the profiling dictionary and info dictionary, which contain the profiling times of the
        `log_likelihood_function` and information on the model and dataset used to perform the profiling.
        """

        if isinstance(paths, af.DatabasePaths):
            return

        run_time_dict = {}
        info_dict = {}

        repeats = conf.instance["general"]["profiling"]["repeats"]
        info_dict["repeats"] = repeats

        # Ensure numba functions are compiled before profiling begins.

        fit = self.fit_from(instance=instance)
        fit.figure_of_merit

        start = time.time()

        for _ in range(repeats):
            try:
                fit = self.fit_from(instance=instance)
                fit.figure_of_merit
            except Exception:
                logger.info(
                    "Profiling failed. Returning without outputting information."
                )
                return

        fit_time = (time.time() - start) / repeats

        run_time_dict["fit_time"] = fit_time

        fit = self.fit_from(instance=instance, run_time_dict=run_time_dict)
        fit.figure_of_merit

        try:
            info_dict["image_pixels"] = self.dataset.grids.lp.shape_slim
            info_dict[
                "sub_total_light_profiles"
            ] = self.dataset.grids.lp.over_sampler.sub_total
        except AttributeError:
            pass

        if fit.model_obj.has(cls=aa.Pixelization):
            info_dict["use_w_tilde"] = fit.inversion.settings.use_w_tilde
            try:
                info_dict[
                    "sub_total_pixelization"
                ] = self.dataset.grids.pixelization.over_sampler.sub_total
            except AttributeError:
                pass
            info_dict["use_positive_only_solver"] = (
                fit.inversion.settings.use_positive_only_solver
            )
            info_dict["force_edge_pixels_to_zeros"] = (
                fit.inversion.settings.force_edge_pixels_to_zeros
            )
            info_dict["use_w_tilde_numpy"] = fit.inversion.settings.use_w_tilde_numpy
            info_dict["source_pixels"] = len(fit.inversion.reconstruction)

            if hasattr(fit.inversion, "w_tilde"):
                info_dict["w_tilde_curvature_preload_size"] = (
                    fit.inversion.w_tilde.curvature_preload.shape[0]
                )

        self.output_profiling_info(
            paths=paths, run_time_dict=run_time_dict, info_dict=info_dict
        )

        return run_time_dict, info_dict

    def output_profiling_info(
        self, paths: Optional[af.DirectoryPaths], run_time_dict: Dict, info_dict: Dict
    ):
        """
        Output the log likelihood function profiling information to hard-disk as a json file.

        This function is separate from the `profile_log_likelihood_function` function above such that it can be
        called by children `Analysis` classes that profile additional aspects of the model-fit and therefore add
        extra information to the `run_time_dict` and `info_dict`.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        run_time_dict
            A dictionary containing the profiling times of the functions called by the `log_likelihood_function`.
        info_dict
            A dictionary containing information on the model and dataset used to perform the profiling, where these
            settings typically control the overall run-time.
        """

        if paths is None:
            return

        os.makedirs(paths.profile_path, exist_ok=True)

        with open(path.join(paths.profile_path, "run_time_dict.json"), "w+") as f:
            json.dump(run_time_dict, f, indent=4)

        with open(path.join(paths.profile_path, "info_dict.json"), "w+") as f:
            json.dump(info_dict, f, indent=4)
