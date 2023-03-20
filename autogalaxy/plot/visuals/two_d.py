from matplotlib import patches as ptch
from typing import List, Union, Optional

import autoarray as aa
import autoarray.plot as aplt


class Visuals2D(aplt.Visuals2D):
    def __init__(
        self,
        origin: aa.Grid2D = None,
        border: aa.Grid2D = None,
        mask: aa.Mask2D = None,
        positions: Optional[Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]] = None,
        grid: Union[aa.Grid2D, aa.Grid2DSparse] = None,
        mesh_grid: aa.Grid2D = None,
        vectors: aa.VectorYX2DIrregular = None,
        patches: Union[ptch.Patch] = None,
        array_overlay: aa.Array2D = None,
        light_profile_centres: aa.Grid2DIrregular = None,
        mass_profile_centres: aa.Grid2DIrregular = None,
        multiple_images: aa.Grid2DIrregular = None,
        tangential_critical_curves: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        radial_critical_curves: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        tangential_caustics: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        radial_caustics: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        indexes: Union[List[int], List[List[int]]] = None,
        pix_indexes: Union[List[int], List[List[int]]] = None,
    ):

        super().__init__(
            mask=mask,
            positions=positions,
            grid=grid,
            mesh_grid=mesh_grid,
            vectors=vectors,
            patches=patches,
            array_overlay=array_overlay,
            origin=origin,
            border=border,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            indexes=indexes,
            pix_indexes=pix_indexes,
        )

        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images
        self.tangential_critical_curves = tangential_critical_curves
        self.radial_critical_curves = radial_critical_curves
        self.tangential_caustics = tangential_caustics
        self.radial_caustics = radial_caustics

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):

        super().plot_via_plotter(
            plotter=plotter, grid_indexes=grid_indexes, mapper=mapper
        )

        if self.light_profile_centres is not None:
            plotter.light_profile_centres_scatter.scatter_grid(
                grid=self.light_profile_centres
            )

        if self.mass_profile_centres is not None:
            plotter.mass_profile_centres_scatter.scatter_grid(
                grid=self.mass_profile_centres
            )

        if self.multiple_images is not None:
            plotter.multiple_images_scatter.scatter_grid(grid=self.multiple_images)

        if self.tangential_critical_curves is not None:
            try:
                plotter.tangential_critical_curves_plot.plot_grid(
                    grid=self.tangential_critical_curves
                )
            except TypeError:
                pass

        if self.radial_critical_curves is not None:
            try:
                plotter.radial_critical_curves_plot.plot_grid(grid=self.radial_critical_curves)
            except TypeError:
                pass

        if self.tangential_caustics is not None:
            try:
                plotter.tangential_caustics_plot.plot_grid(grid=self.tangential_caustics)
            except TypeError:
                pass

        if self.radial_caustics is not None:
            try:
                plotter.radial_caustics_plot.plot_grid(grid=self.radial_caustics)
            except TypeError:
                pass
