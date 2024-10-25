from typing import List
from os import path

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.plot.fit_ellipse_plotters import FitEllipsePlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


class PlotterInterfaceEllipse(PlotterInterface):
    def imaging(self, dataset: aa.Imaging):
        """
        Visualizes an `Imaging` dataset object.

        Images are output to the `image` folder of the `image_path` in a subfolder called `imaging`. When used with
        a non-linear search the `image_path` points to the search's results folder.
        `.
        Visualization includes individual images of attributes of the dataset (e.g. the image, noise map, PSF) and a
        subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [dataset] header.

        Parameters
        ----------
        dataset
            The imaging dataset whose attributes are visualized.
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
            signal_to_noise_map=should_plot("signal_to_noise_map"),
        )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        dataset_plotter = aplt.ImagingPlotter(
            dataset=dataset, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_dataset"):
            dataset_plotter.subplot_dataset()

    def fit_ellipse(
        self,
        fit_list: List[FitEllipse],
        during_analysis: bool,
        subfolders: str = "fit_dataset",
    ):
        """
        Visualizes a `FitEllipse` object, which fits an imaging dataset.

        Images are output to the `image` folder of the `image_path` in a subfolder called `fit`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `FitEllipse` inferred by the search so far.

        Visualization includes individual images of attributes of the `FitEllipse` (e.g. the model data, residual map)
        and a subplot of all `FitEllipse`'s images on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [fit] header.

        Parameters
        ----------
        fit
            The maximum log likelihood `FitEllipse` of the non-linear search which is used to plot the fit.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        visuals_2d
            An object containing attributes which may be plotted over the figure (e.g. the centres of mass and light
            profiles).
        """

        def should_plot(name):
            return plot_setting(section=["fit", "fit_ellipse"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        fit_plotter = FitEllipsePlotter(
            fit_list=fit_list, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        fit_plotter.figures_2d(data=should_plot("data"))

        if should_plot("data_no_ellipse"):
            fit_plotter.figures_2d(
                data=True,
                disable_data_contours=True,
            )

        fit_plotter.mat_plot_2d.use_log10 = True

        fit_plotter.figures_2d(data=should_plot("data"))

        if should_plot("data_no_ellipse"):
            fit_plotter.figures_2d(
                data=True,
                disable_data_contours=True,
            )

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(subfolders=path.join(subfolders, "end"))

            fit_plotter = FitEllipsePlotter(
                fit_list=fit_list, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            fit_plotter.figures_2d(data=True)
            fit_plotter.figures_2d(
                data=True,
                disable_data_contours=True,
            )
