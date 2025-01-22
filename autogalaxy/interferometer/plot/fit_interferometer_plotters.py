from typing import List

import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_interferometer_plotters import FitInterferometerPlotterMeta

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.plot.include.one_d import Include1D
from autogalaxy.plot.include.two_d import Include2D

from autogalaxy.galaxy.plot.galaxies_plotters import GalaxiesPlotter


class FitInterferometerPlotter(Plotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitInterferometer` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 2D plots.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            get_visuals_2d_real_space=self.get_visuals_2d_real_space,
            mat_plot_1d=self.mat_plot_1d,
            include_1d=self.include_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot = self._fit_interferometer_meta_plotter.subplot
        #  self.subplot_fit = self._fit_interferometer_meta_plotter.subplot_fit
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    def get_visuals_2d_real_space(self) -> Visuals2D:
        return self.get_2d.via_mask_from(mask=self.fit.dataset.real_space_mask)

    @property
    def galaxies(self) -> List[Galaxy]:
        return self.fit.galaxies_linear_light_profiles_to_light_profiles

    def galaxies_plotter_from(self, galaxies: List[Galaxy]) -> GalaxiesPlotter:
        """
        Returns a `GalaxiesPlotter` corresponding to an input galaxies list.

        Returns
        -------
        galaxies
            The galaxies used to make the `GalaxiesPlotter`.
        """
        return GalaxiesPlotter(
            galaxies=galaxies,
            grid=self.fit.grids.lp,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self) -> aplt.InversionPlotter:
        """
        Returns an `InversionPlotter` corresponding to the `Inversion` of the fit.

        Returns
        -------
        InversionPlotter
            An object that plots inversions which is used for plotting attributes of the inversion.
        """
        return aplt.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.get_visuals_2d_real_space(),
            include_2d=self.include_2d,
        )

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitImaging` object.
        """

        self.open_subplot_figure(number_subplots=9)

        self.figures_2d(amplitudes_vs_uv_distances=True)

        self.mat_plot_1d.subplot_index = 2
        self.mat_plot_2d.subplot_index = 2

        self.figures_2d(dirty_image=True)
        self.figures_2d(dirty_signal_to_noise_map=True)

        self.mat_plot_1d.subplot_index = 4
        self.mat_plot_2d.subplot_index = 4

        self.figures_2d(dirty_model_image=True)

        self.mat_plot_1d.subplot_index = 5
        self.mat_plot_2d.subplot_index = 5

        self.figures_2d(normalized_residual_map_real=True)
        self.figures_2d(normalized_residual_map_imag=True)

        self.mat_plot_1d.subplot_index = 7
        self.mat_plot_2d.subplot_index = 7

        self.figures_2d(dirty_normalized_residual_map=True)

        self.mat_plot_2d.cmap.kwargs["vmin"] = -1.0
        self.mat_plot_2d.cmap.kwargs["vmax"] = 1.0

        self.set_title(label="Normalized Residual Map (1 sigma)")
        self.figures_2d(dirty_normalized_residual_map=True)
        self.set_title(label=None)

        self.mat_plot_2d.cmap.kwargs.pop("vmin")
        self.mat_plot_2d.cmap.kwargs.pop("vmax")

        self.figures_2d(dirty_chi_squared_map=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_fit")
        self.close_subplot_figure()

    def subplot_fit_real_space(self):
        """
        Standard subplot of the real-space attributes of the plotter's `FitInterferometer` object.

        Depending on whether `LightProfile`'s or an `Inversion` are used to represent galaxies, different
        methods are called to create these real-space images.
        """
        if not self.galaxies.has(cls=aa.Pixelization):
            galaxies_plotter = self.galaxies_plotter_from(galaxies=self.galaxies)

            galaxies_plotter.subplot(image=True, auto_filename="subplot_fit_real_space")

        elif self.galaxies.has(cls=aa.Pixelization):
            self.open_subplot_figure(number_subplots=6)

            mapper_index = 0

            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=mapper_index, reconstructed_image=True
            )
            self.inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=mapper_index, reconstruction=True
            )

            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_fit_real_space"
            )

            self.close_subplot_figure()
