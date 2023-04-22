import shutil
from os import path
import pytest

import autogalaxy as ag
from autogalaxy.imaging.model.visualizer import VisualizerImaging

directory = path.dirname(path.abspath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_fit_imaging__uses_configs(
    masked_imaging_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = VisualizerImaging(visualize_path=plot_path)

    visualizer.visualize_fit_imaging(
        fit=fit_imaging_x2_galaxy_inversion_7x7, during_analysis=False
    )

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    assert path.join(plot_path, "subtracted_image_of_galaxy_1.png") in plot_patch.paths
    assert path.join(plot_path, "model_image_of_galaxy_1.png") not in plot_patch.paths

    image = ag.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "fits", "data.fits"), hdu=0
    )

    assert image.shape == (5, 5)
