import os
from os import path
from typing import Dict, List, Union

from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.plot.galaxy_plotters import GalaxyPlotter
from autogalaxy.galaxy.plot.galaxies_plotters import GalaxiesPlotter
from autogalaxy.galaxy.plot.adapt_plotters import AdaptPlotter

from autogalaxy.plot.include.two_d import Include2D
from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


def setting(section: Union[List[str], str], name: str):
    if isinstance(section, str):
        return conf.instance["visualize"]["plots"][section][name]

    for sect in reversed(section):
        try:
            return conf.instance["visualize"]["plots"][sect][name]
        except KeyError:
            continue

    return conf.instance["visualize"]["plots"][section[0]][name]


def plot_setting(section: Union[List[str], str], name: str) -> bool:
    return setting(section, name)


class PlotterInterface:
    def __init__(self, image_path: str, title_prefix: str = None):
        """
        Provides an interface between an output path and all plotter objects.

        This is used to visualize the results of a model-fit, where the `image_path` points to the
        folder where the results of the model-fit are stored on your hard-disk, which is typically the `image` folder
        of a non-linear search.

        The `PlotterInterface` is typically used in the `Analysis` class of a non-linear search to visualize the maximum
        log likelihood model of the model-fit so far.

        The methods of the `PlotterInterface` are called throughout a non-linear search using the `Analysis.Visualizer`
        classes `visualize` method.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        image_path
            The path on the hard-disk to the `image` folder of the non-linear searches results.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        self.image_path = image_path
        self.title_prefix = title_prefix

        self.include_2d = Include2D()

        os.makedirs(image_path, exist_ok=True)

    def mat_plot_1d_from(self, subfolders: str, format: str = "png") -> MatPlot1D:
        """
        Returns a 1D matplotlib plotting object whose `Output` class uses the `image_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Parameters
        ----------
        subfolders
            Subfolders between the `image` folder of the non-linear search and where the images are output. For example,
            images associsted with a fit are output to the subfolder `fit`.
        format
            The format images are output as, e.g. `.png` files.

        Returns
        -------
        MatPlot1D
            The 1D matplotlib plotter object.
        """
        return MatPlot1D(
            title=aplt.Title(prefix=self.title_prefix),
            output=aplt.Output(
                path=path.join(self.image_path, subfolders), format=format
            ),
        )

    def mat_plot_2d_from(self, subfolders, format="png") -> MatPlot2D:
        """
        Returns a 2D matplotlib plotting object whose `Output` class uses the `image_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Parameters
        ----------
        subfolders
            Subfolders between the `image` folder of the non-linear search and where the images are output. For example,
            images associsted with a fit are output to the subfolder `fit`.
        format
            The format images are output as, e.g. `.png` files.

        Returns
        -------
        MatPlot2D
            The 2D matplotlib plotter object.
        """
        return MatPlot2D(
            title=aplt.Title(prefix=self.title_prefix),
            output=aplt.Output(
                path=path.join(self.image_path, subfolders), format=format
            ),
        )

    def galaxies(
        self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike, during_analysis: bool
    ):
        """
        Visualizes a list of galaxies.

        Images are output to the `image` folder of the `image_path` in a subfolder called `galaxies`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood galaxies inferred by the search so far.

        Visualization includes individual images of attributes of the galaxies (e.g. its image, convergence, deflection
        angles) and a subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [galaxies] header.

        Parameters
        ----------
        galaxies
            The maximum log likelihood galaxies of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        galaxies = Galaxies(galaxies=galaxies)

        def should_plot(name):
            return plot_setting(section="galaxies", name=name)

        subfolders = "galaxies"

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        plotter = GalaxiesPlotter(
            galaxies=galaxies,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_galaxy_images"):
            plotter.subplot_galaxy_images()

        plotter.figures_2d(
            image=should_plot("image"),
            convergence=should_plot("convergence"),
            potential=should_plot("potential"),
            deflections_y=should_plot("deflections"),
            deflections_x=should_plot("deflections"),
            magnification=should_plot("magnification"),
        )

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join(subfolders, "end"),
            )

            plotter = GalaxiesPlotter(
                galaxies=galaxies,
                grid=grid,
                mat_plot_2d=mat_plot_2d,
                include_2d=self.include_2d,
            )

            plotter.figures_2d(
                image=True,
                convergence=True,
                potential=True,
                deflections_y=True,
                deflections_x=True,
                magnification=True,
            )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        plotter = GalaxiesPlotter(
            galaxies=galaxies,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
            include_2d=self.include_2d,
        )

        if should_plot("subplot_galaxies"):
            plotter.subplot()

        if not during_analysis and should_plot("all_at_end_fits"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join(subfolders, "fits"), format="fits"
            )

            plotter = GalaxiesPlotter(
                galaxies=galaxies,
                grid=grid,
                mat_plot_2d=mat_plot_2d,
                include_2d=self.include_2d,
            )

            plotter.figures_2d(
                image=True,
                convergence=True,
                potential=True,
                deflections_y=True,
                deflections_x=True,
                magnification=True,
            )

    def galaxies_1d(
        self, galaxies: [List[Galaxy]], grid: aa.type.Grid2DLike, during_analysis: bool
    ):
        """
        Visualizes a list of `Galaxy` objects.

        Images are output to the `image` folder of the `image_path` in a subfolder called `galaxies`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Galaxy`'s inferred by the search so far.

        Visualization includes individual images of attributes of each galaxy (e.g. 1D plots of their image,
        convergence) and a subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [galaxies] header.

        Parameters
        ----------
        galaxies
            A list of the maximum log likelihood `Galaxy`'s of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="galaxies_1d", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders="galaxies_1d")

        for galaxy in galaxies:
            galaxy_plotter = GalaxyPlotter(
                galaxy=galaxy,
                grid=grid,
                mat_plot_1d=mat_plot_1d,
                include_2d=self.include_2d,
            )

            try:
                galaxy_plotter.figures_1d_decomposed(
                    image=should_plot("image"),
                    convergence=should_plot("convergence"),
                    potential=should_plot("potential"),
                )
            except OverflowError:
                pass

    def inversion(self, inversion: aa.Inversion, during_analysis: bool):
        """
        Visualizes an `Inversion` object.

        Images are output to the `image` folder of the `image_path` in a subfolder called `inversion`. When
        used with a non-linear search the `image_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Inversion` inferred by the search so far.

        Visualization includes individual images of attributes of the dataset (e.g. the reconstructed image, the
        reconstruction) and a subplot of all these attributes on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [inversion] header.

        Parameters
        ----------
        inversion
            The inversion used to fit the dataset whose attributes are visualized.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        subfolders = "inversion"

        mat_plot_2d = self.mat_plot_2d_from(subfolders=subfolders)

        inversion_plotter = aplt.InversionPlotter(
            inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        inversion_plotter.figures_2d(
            reconstructed_image=should_plot("reconstructed_image")
        )

        inversion_plotter.figures_2d_of_pixelization(
            pixelization_index=0,
            data_subtracted=should_plot("data_subtracted"),
            reconstructed_image=should_plot("reconstructed_image"),
            reconstruction=should_plot("reconstruction"),
            mesh_pixels_per_image_pixels=should_plot("mesh_pixels_per_image_pixels"),
            errors=should_plot("errors"),
            regularization_weights=should_plot("regularization_weights"),
        )

        mat_plot_2d = self.mat_plot_2d_from(subfolders="")

        inversion_plotter = aplt.InversionPlotter(
            inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_inversion"):
            mapper_list = inversion.cls_list_from(cls=aa.AbstractMapper)

            for mapper_index in range(len(mapper_list)):
                inversion_plotter.subplot_of_mapper(mapper_index=mapper_index)

        if not during_analysis and should_plot("all_at_end_png"):
            mat_plot_2d = self.mat_plot_2d_from(subfolders=path.join(subfolders, "end"))

            inversion_plotter = aplt.InversionPlotter(
                inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            inversion_plotter.figures_2d(reconstructed_image=True)

            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0,
                reconstructed_image=True,
                reconstruction=True,
                errors=True,
                regularization_weights=True,
            )

        if not during_analysis and should_plot("all_at_end_fits"):
            mat_plot_2d = self.mat_plot_2d_from(
                subfolders=path.join(subfolders, "fits"), format="fits"
            )

            inversion_plotter = aplt.InversionPlotter(
                inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
            )

            inversion_plotter.figures_2d(reconstructed_image=True)

            inversion_plotter.figures_2d_of_pixelization(
                pixelization_index=0,
                reconstructed_image=True,
                reconstruction=True,
                errors=True,
                regularization_weights=True,
                interpolate_to_uniform=True,
            )

    def adapt_images(
        self,
        adapt_images: AdaptImages,
    ):
        """
        Visualizes the adapt images used by a model-fit for adaptive pixelization mesh's and regularization.

        Images are output to the `image` folder of the `image_path` in a subfolder called `adapt`. When
        used with a non-linear search the `image_path` points to the search's results folder.

        Visualization includes an image of the overall adapt model image and a subplot of all galaxy images on the same
        figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under the
        [adapt] header.

        Parameters
        ----------
        adapt_images
            The adapt images (e.g. overall model image, individual galaxy images).
        """

        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="adapt")

        adapt_plotter = AdaptPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            adapt_plotter.figure_model_image(model_image=adapt_images.model_image)

        if should_plot("images_of_galaxies"):
            adapt_plotter.subplot_images_of_galaxies(
                adapt_galaxy_name_image_dict=adapt_images.galaxy_image_dict
            )
