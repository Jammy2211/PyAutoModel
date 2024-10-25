from os import path

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot.fit_imaging_plotters import FitImagingPlotter
from autogalaxy.analysis.plotter_interface import PlotterInterface

from autogalaxy.analysis.plotter_interface import plot_setting


class PlotterInterfaceImaging(PlotterInterface):
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
            psf=should_plot("psf"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            over_sampling=should_plot("over_sampling"),
            over_sampling_non_uniform=should_plot("over_sampling_non_uniform"),
            over_sampling_pixelization=should_plot("over_sampling_pixelization"),
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
