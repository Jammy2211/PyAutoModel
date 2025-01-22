from os import path
from typing import List

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot.fit_imaging_plotters import FitImagingPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


class PlotterInterfaceImaging(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        """
        Output visualization of an `Imaging` dataset, typically before a model-fit is performed.

        Images are output to the `image` folder of the `image_path` in a subfolder called `dataset`. When used with
        a non-linear search the `image_path` is the output folder of the non-linear search.
        `.
        Visualization includes individual images of attributes of the dataset (e.g. the image, noise map, PSF) and a
        subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` header.

        Parameters
        ----------
        dataset
            The imaging dataset which is visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="dataset")

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        dataset_plotter.figures_2d(
            data=should_plot("data"),
            noise_map=should_plot("noise_map"),
            psf=should_plot("psf"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            over_sample_size_lp=should_plot("over_sample_size_lp"),
            over_sample_size_pixelization=should_plot("over_sample_size_pixelization"),
        )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_imaging(
        self, fit: FitImaging, during_analysis: bool, subfolders: str = "fit_dataset"
    ):
        """
        Visualizes a `FitImaging` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path` in a subfolder called `fit`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitImaging` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitImaging` (e.g. the model data, residual map)
        and a subplot of all `FitImaging`'s images on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [fit] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitImaging` of the non-linear search which is used to plot the fit.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_plotter = FitImagingPlotter(
            fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_plotter.figures_2d(
            data=should_plot("data"),
            noise_map=should_plot("noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            model_image=should_plot("model_data"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
        )

        fit_plotter.figures_2d_of_galaxies(
            subtracted_image=should_plot("subtracted_images_of_galaxies"),
            model_image=should_plot("model_images_of_galaxies"),
        )

        if should_plot("subplot_fit"):
            fit_plotter.subplot_fit()

        if should_plot("subplot_of_galaxies"):
            fit_plotter.subplot_of_galaxies()

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(subfolders=path.join(subfolders, "end"))

            fit_plotter = FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            fit_plotter.figures_2d(
                data=True,
                noise_map=True,
                signal_to_noise_map=True,
                model_image=True,
                residual_map=True,
                normalized_residual_map=True,
                chi_squared_map=True,
            )

            fit_plotter.figures_2d_of_galaxies(subtracted_image=True, model_image=True)

        if not during_analysis and should_plot("all_at_end_fits"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join(subfolders, "fits"), format="fits"
            )

            fit_plotter = FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            fit_plotter.figures_2d(
                data=True,
                noise_map=True,
                signal_to_noise_map=True,
                model_image=True,
                residual_map=True,
                normalized_residual_map=True,
                chi_squared_map=True,
            )

            fit_plotter.figures_2d_of_galaxies(
                subtracted_image=True,
                model_image=True,
            )

    def imaging_combined(self, dataset_list: List[aa.Imaging]):
        """
        Output visualization of all `Imaging` datasets in a summed combined analysis, typically before a model-fit
        is performed.

        Images are output to the `image` folder of the `image_path` in a subfolder called `dataset_combined`. When
        used with a non-linear search the `image_path` is the output folder of the non-linear search.
        `.
        Visualization includes individual images of attributes of each dataset (e.g. the image, noise map, PSF) on
        a single subplot, such that the full suite of multiple datasets can be viewed on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `dataset` header.

        Parameters
        ----------
        dataset
            The list of imaging datasets which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="combined")

        dataset_plotter_list = [
            aplt.ImagingPlotter(
                dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )
            for dataset in dataset_list
        ]

        subplot_shape = (len(dataset_list), 4)

        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=dataset_plotter_list, subplot_shape=subplot_shape
        )

        if should_plot("subplot_dataset"):
            multi_plotter.subplot_of_figures_multi(
                func_name_list=["figures_2d"] * 4,
                figure_name_list=["data", "noise_map", "signal_to_noise_map", "psf"],
                filename_suffix="dataset",
            )

            for plotter in multi_plotter.plotter_list:
                plotter.mat_plot_2d.use_log10 = True

            multi_plotter.subplot_of_figures_multi(
                func_name_list=["figures_2d"] * 4,
                figure_name_list=["data", "noise_map", "signal_to_noise_map", "psf"],
                filename_suffix="dataset_log10",
            )

    def fit_imaging_combined(self, fit_list: List[FitImaging]):
        """
        Output visualization of all `FitImaging` objects in a summed combined analysis, typically during or after a
        model-fit is performed.

        Images are output to the `image` folder of the `image_path` in a subfolder called `combined`. When used
        with a non-linear search the `image_path` is the output folder of the non-linear search.
        `.
        Visualization includes individual images of attributes of each fit (e.g. data, normalized residual-map) on
        a single subplot, such that the full suite of multiple datasets can be viewed on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `fit` header.

        Parameters
        ----------
        fit
            The list of imaging fits which are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="combined")

        fit_plotter_list = [
            FitImagingPlotter(
                fit=fit, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )
            for fit in fit_list
        ]

        subplot_shape = (len(fit_list), 5)

        multi_plotter = aplt.MultiFigurePlotter(
            plotter_list=fit_plotter_list, subplot_shape=subplot_shape
        )

        if should_plot("subplot_fit"):

            def make_subplot_fit(filename_suffix):
                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"] * 4,
                    figure_name_list=[
                        "data",
                        "signal_to_noise_map",
                        "model_image",
                        "normalized_residual_map",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * 5,
                    close_subplot=False,
                )

                for plotter in multi_plotter.plotter_list:
                    plotter.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
                    plotter.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

                multi_plotter.subplot_of_figures_multi(
                    func_name_list=["figures_2d"],
                    figure_name_list=[
                        "normalized_residual_map",
                    ],
                    filename_suffix=filename_suffix,
                    number_subplots=len(fit_list) * 5,
                    subplot_index_offset=4,
                    open_subplot=False,
                )

                for plotter in multi_plotter.plotter_list:
                    plotter.mat_plot_2d.cmap.kwargs["vmin"] = None
                    plotter.mat_plot_2d.cmap.kwargs["vmax"] = None

            make_subplot_fit(filename_suffix="fit")

            for plotter in multi_plotter.plotter_list:
                plotter.mat_plot_2d.use_log10 = True

            make_subplot_fit(filename_suffix="fit_log10")
