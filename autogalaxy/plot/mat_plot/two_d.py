from typing import List, Optional, Union

import autoarray.plot as aplt
from autogalaxy.plot import wrap as w


class MatPlot2D(aplt.MatPlot2D):
    def __init__(
        self,
        units: Optional[aplt.Units] = None,
        figure: Optional[aplt.Figure] = None,
        axis: Optional[aplt.Axis] = None,
        cmap: Optional[aplt.Cmap] = None,
        colorbar: Optional[aplt.Colorbar] = None,
        colorbar_tickparams: Optional[aplt.ColorbarTickParams] = None,
        tickparams: Optional[aplt.TickParams] = None,
        yticks: Optional[aplt.YTicks] = None,
        xticks: Optional[aplt.XTicks] = None,
        title: Optional[aplt.Title] = None,
        ylabel: Optional[aplt.YLabel] = None,
        xlabel: Optional[aplt.XLabel] = None,
        text: Optional[Union[aplt.Text, List[aplt.Text]]] = None,
        annotate: Optional[Union[aplt.Annotate, List[aplt.Annotate]]] = None,
        legend: Optional[aplt.Legend] = None,
        output: Optional[aplt.Output] = None,
        array_overlay: Optional[aplt.ArrayOverlay] = None,
        grid_scatter: Optional[aplt.GridScatter] = None,
        grid_plot: Optional[aplt.GridPlot] = None,
        vector_yx_quiver: Optional[aplt.VectorYXQuiver] = None,
        patch_overlay: Optional[aplt.PatchOverlay] = None,
        interpolated_reconstruction: Optional[aplt.InterpolatedReconstruction] = None,
        voronoi_drawer: Optional[aplt.VoronoiDrawer] = None,
        origin_scatter: Optional[aplt.OriginScatter] = None,
        mask_scatter: Optional[aplt.MaskScatter] = None,
        border_scatter: Optional[aplt.BorderScatter] = None,
        positions_scatter: Optional[aplt.PositionsScatter] = None,
        index_scatter: Optional[aplt.IndexScatter] = None,
        mesh_grid_scatter: Optional[aplt.MeshGridScatter] = None,
        light_profile_centres_scatter: Optional[w.LightProfileCentresScatter] = None,
        mass_profile_centres_scatter: Optional[w.MassProfileCentresScatter] = None,
        multiple_images_scatter: Optional[w.MultipleImagesScatter] = None,
        tangential_critical_curves_plot: Optional[
            w.TangentialCriticalCurvesPlot
        ] = None,
        radial_critical_curves_plot: Optional[w.RadialCriticalCurvesPlot] = None,
        tangential_caustics_plot: Optional[w.TangentialCausticsPlot] = None,
        radial_caustics_plot: Optional[w.RadialCausticsPlot] = None,
    ):
        """
        Visualizes data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and
        customize the appearance of the plots of the data structure. If the values of these matplotlib wrapper
        objects are not manually specified, they assume the default values provided in
        the `config.visualize.mat_*` `.ini` config files.

        The following data structures can be plotted using the following matplotlib functions:

        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `MapperVoronoiNoInterp`: using `plt.fill`.

        Parameters
        ----------
        units
          The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure
          Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
          via `plt.close`.
        axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap
          Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
          as `colors.Normalize` and `colors.LogNorm`.
        colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams
          Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks
          Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.yticks`.
        xticks
          Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
          `plt.xticks`.
        title
          Sets the figure title and customizes its appearance using `plt.title`.
        ylabel
          Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel
          Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        text
            Sets any text on the figure and customizes its appearance using `plt.text`.
        annotate
            Sets any annotations on the figure and customizes its appearance using `plt.annotate`.
        legend
          Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
          Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay
          Overlays an input `Array2D` over the figure using `plt.imshow`.
        grid_scatter
          Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot
          Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_yx_quiver
          Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay
          Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer
          Interpolations the reconstruction of a `Mapper` object from its irregular grid (e.g. Delaunay, Voronoi) to a
          uniform 2D array and plots it via `plt.imshow()`.
        voronoi_drawer
          Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter
          Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter
          Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter
          Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter
          Scatters specific (y,x) coordinates input as a `Grid2DIrregular` object over the figure.
        index_scatter
          Scatters specific coordinates of an input `Grid2D` based on input values of the `Grid2D`'s 1D or 2D indexes.
        mesh_grid_scatter
          Scatters the `PixelizationGrid` of a `Pixelization` object.
        light_profile_centres_scatter
          Scatters the (y,x) centres of all `LightProfile`'s in the plotted object (e.g. a `Tracer`).
        mass_profile_centres_scatter
          Scatters the (y,x) centres of all `MassProfile`'s in the plotted object (e.g. a `Tracer`).
        light_profile_centres_scatter
          Scatters the (y,x) coordinates of the multiple image locations of the lens mass model.
        tangential_critical_curves_plot
            Plots the tangential critical curves of the lens mass model as colored lines.
        radial_critical_curves_plot
            Plots the radial critical curves of the lens mass model as colored lines.
        tangential_caustics_plot
            Plots the tangential caustics of the lens mass model as colored lines.
        radial_caustics_plot
            Plots the radial caustics of the lens mass model as colored lines.
        """

        self.light_profile_centres_scatter = (
            light_profile_centres_scatter
            or w.LightProfileCentresScatter(is_default=True)
        )
        self.mass_profile_centres_scatter = (
            mass_profile_centres_scatter or w.MassProfileCentresScatter(is_default=True)
        )
        self.multiple_images_scatter = (
            multiple_images_scatter or w.MultipleImagesScatter(is_default=True)
        )
        self.tangential_critical_curves_plot = (
            tangential_critical_curves_plot
            or w.TangentialCriticalCurvesPlot(is_default=True)
        )
        self.radial_critical_curves_plot = (
            radial_critical_curves_plot or w.RadialCriticalCurvesPlot()
        )
        self.tangential_caustics_plot = (
            tangential_caustics_plot or w.TangentialCausticsPlot(is_default=True)
        )
        self.radial_caustics_plot = radial_caustics_plot or w.RadialCausticsPlot()

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            legend=legend,
            title=title,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            ylabel=ylabel,
            xlabel=xlabel,
            text=text,
            annotate=annotate,
            output=output,
            origin_scatter=origin_scatter,
            mask_scatter=mask_scatter,
            border_scatter=border_scatter,
            grid_scatter=grid_scatter,
            positions_scatter=positions_scatter,
            index_scatter=index_scatter,
            mesh_grid_scatter=mesh_grid_scatter,
            vector_yx_quiver=vector_yx_quiver,
            patch_overlay=patch_overlay,
            array_overlay=array_overlay,
            grid_plot=grid_plot,
            interpolated_reconstruction=interpolated_reconstruction,
            voronoi_drawer=voronoi_drawer,
        )
