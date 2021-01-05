from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.plotters import inversion_plotters
from autogalaxy.plot.mat_wrap import lensing_mat_plot, lensing_include, lensing_visuals
from autogalaxy.plot.plotters import plane_plotters
from autoarray.plot.plotters import fit_interferometer_plotters
from autogalaxy.fit import fit as f


class FitInterferometerPlotter(fit_interferometer_plotters.FitInterferometerPlotter):
    def __init__(
        self,
        fit: f.FitInterferometer,
        mat_plot_1d: lensing_mat_plot.MatPlot1D = lensing_mat_plot.MatPlot1D(),
        visuals_1d: lensing_visuals.Visuals1D = lensing_visuals.Visuals1D(),
        include_1d: lensing_include.Include1D = lensing_include.Include1D(),
        mat_plot_2d: lensing_mat_plot.MatPlot2D = lensing_mat_plot.MatPlot2D(),
        visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
        include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    @property
    def plane(self):
        return self.fit.plane

    @property
    def visuals_with_include_2d(self):
        visuals_2d = super(FitInterferometerPlotter, self).visuals_with_include_2d
        return visuals_2d + lensing_visuals.Visuals2D()

    def plane_plotter_from(self, plane):
        return plane_plotters.PlanePlotter(
            plane=plane,
            grid=self.fit.masked_interferometer.grid,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d,
            include_2d=self.include_2d,
        )

    @property
    def inversion_plotter(self):
        return inversion_plotters.InversionPlotter(
            inversion=self.fit.inversion,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_with_include_2d,
            include_2d=self.include_2d,
        )

    @abstract_plotters.for_subplot
    def subplot_fit_real_space(self):

        number_subplots = 2

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        if self.fit.inversion is None:

            plane_plotter = self.plane_plotter_from(plane=self.plane)

            plane_plotter.figure_image()

            self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

            plane_plotter.figure_plane_image()

        elif self.fit.inversion is not None:

            self.inversion_plotter.figure_reconstructed_image()

            aspect_inv = self.mat_plot_2d.figure.aspect_for_subplot_from_grid(
                grid=self.fit.inversion.mapper.source_full_grid
            )

            self.setup_subplot(
                number_subplots=number_subplots,
                subplot_index=2,
                aspect=float(aspect_inv),
            )

            self.inversion_plotter.figure_reconstruction()

        self.mat_plot_2d.output.subplot_to_figure()

        self.mat_plot_2d.figure.close()
