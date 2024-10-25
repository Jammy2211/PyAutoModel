import logging
import time
from typing import Dict, List, Optional, Tuple

from autoconf.dictable import to_dict

import autofit as af
import autoarray as aa

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.model.result import ResultEllipse
from autogalaxy.ellipse.model.visualizer import VisualizerEllipse

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisEllipse(af.Analysis):
    Result = ResultEllipse
    Visualizer = VisualizerEllipse

    def __init__(self, dataset: aa.Imaging, title_prefix: str = None):
        """
        Fits a model made of ellipses to an imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit ellipses to an imaging dataset.

        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model containing ellipses is fitted to.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        self.dataset = dataset
        self.title_prefix = title_prefix

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extract all ellipses from the model instance.

        2) Use the ellipses to create a list of `FitEllipse` objects, which fits each ellipse to the data and noise-map
        via interpolation and subtracts these values from their mean values in order to quantify how well the ellipse
        traces around the data.

        Certain models will fail to fit the dataset and raise an exception. For example the ellipse parameters may be
        ill defined and raise an Exception. In such circumstances the model is discarded and its likelihood value is
        passed to the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """
        fit_list = self.fit_list_from(instance=instance)

        return sum(fit.log_likelihood for fit in fit_list)

    def fit_list_from(self, instance: af.ModelInstance) -> List[FitEllipse]:
        """
        Given a model instance create a list of `FitEllipse` objects.

        This function unpacks the `instance`, specifically the `ellipses` and (in input) the `multipoles` and uses
        them to create a list of `FitEllipse` objects that are used to fit the model to the imaging data.

        This function is used in the `log_likelihood_function` to fit the model containing ellipses to the imaging data
        and compute the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The fit of the ellipses to the imaging dataset, which includes the log likelihood.
        """
        fit_list = []

        for i in range(len(instance.ellipses)):
            ellipse = instance.ellipses[i]

            try:
                multipole_list = instance.multipoles[i]
            except AttributeError:
                multipole_list = None

            fit = FitEllipse(
                dataset=self.dataset, ellipse=ellipse, multipole_list=multipole_list
            )

            fit_list.append(fit)

        return fit_list

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

        The `ResultEllipse` object contains a number of methods which use the above objects to create the max
        log likelihood galaxies `FitEllipse`, etc.

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
            The result of fitting the ellipse model to the imaging dataset, via a non-linear search.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
         Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
         folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The imaging dataset (data / noise-map / etc.).
         - The mask applied to the dataset.
         - The Cosmology.

         This function also outputs attributes specific to an imaging dataset:

        - Its mask.

         It is common for these attributes to be loaded by many of the template aggregator functions given in the
         `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
         the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
         output by this function.

         Parameters
         ----------
         paths
             The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
             visualization, and the pickled objects used by the aggregator output by this function.
        """
        paths.save_fits(
            name="data",
            hdu=self.dataset.data.hdu_for_output,
            prefix="dataset",
        )
        paths.save_fits(
            name="noise_map",
            hdu=self.dataset.noise_map.hdu_for_output,
            prefix="dataset",
        )
        paths.save_fits(
            name="mask",
            hdu=self.dataset.mask.hdu_for_output,
            prefix="dataset",
        )
        paths.save_json(
            name="over_sampling",
            object_dict=to_dict(self.dataset.over_sampling),
            prefix="dataset",
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
        repeats = 1

        if isinstance(paths, af.DatabasePaths):
            return

        run_time_dict = {}

        # Ensure numba functions are compiled before profiling begins.

        self.log_likelihood_function(instance=instance)

        start = time.time()

        for _ in range(repeats):
            try:
                self.log_likelihood_function(instance=instance)
            except Exception:
                logger.info(
                    "Profiling failed. Returning without outputting information."
                )
                return

        fit_time = (time.time() - start) / repeats

        run_time_dict["fit_time"] = fit_time

        return run_time_dict, {}
