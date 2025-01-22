import numpy as np
import os
from os import path
import pytest
import shutil

import autogalaxy as ag


def test_via_signal_to_noise_map(dataset_quantity_7x7_array_2d, mask_2d_7x7):
    data = ag.Array2D.no_mask(values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    signal_to_noise_map = ag.Array2D.no_mask(
        values=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        signal_to_noise_map, 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[1.0, 0.4], [0.2, 0.1]]), 1.0e-4
    )

    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], pixel_scales=1.0
    )
    signal_to_noise_map = ag.Array2D.no_mask(
        values=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        np.array([[1.0, 1.0], [5.0, 5.0], [15.0, 15.0], [40.0, 40.0]]), 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[[1.0, 1.0], [0.4, 0.4]], [[0.2, 0.2], [0.1, 0.1]]]), 1.0e-4
    )


def test__apply_mask__masks_dataset(
    dataset_quantity_7x7_array_2d, dataset_quantity_7x7_vector_yx_2d, mask_2d_7x7
):
    dataset_quantity_7x7 = dataset_quantity_7x7_array_2d.apply_mask(mask=mask_2d_7x7)

    assert (dataset_quantity_7x7.data.slim == np.ones(9)).all()
    assert (
        dataset_quantity_7x7.data.native == np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
    assert (
        dataset_quantity_7x7.noise_map.native
        == 2.0 * np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    dataset_quantity_7x7 = dataset_quantity_7x7_vector_yx_2d.apply_mask(
        mask=mask_2d_7x7
    )

    assert (dataset_quantity_7x7.data.slim == np.ones((9, 2))).all()
    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones((9, 2))).all()


def test__grid(
    dataset_quantity_7x7_array_2d,
    mask_2d_7x7,
    grid_2d_7x7,
    blurring_grid_2d_7x7,
):
    dataset = dataset_quantity_7x7_array_2d.apply_mask(mask=mask_2d_7x7)

    assert isinstance(dataset.grids.lp, ag.Grid2D)
    assert (dataset.grids.lp == grid_2d_7x7).all()

    dataset_quantity = ag.DatasetQuantity(
        data=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        noise_map=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
        over_sample_size_lp=4,
    )

    dataset = dataset_quantity.apply_mask(mask=mask_2d_7x7)

    assert (dataset.grids.lp == grid_2d_7x7).all()


def test__vector_data__y_x():
    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]],
        pixel_scales=1.0,
    )

    noise_map = ag.VectorYX2D.no_mask(
        values=[[[1.1, 5.1], [2.1, 6.1]], [[3.1, 7.1], [4.1, 8.1]]],
        pixel_scales=1.0,
    )

    dataset_quantity = ag.DatasetQuantity(data=data, noise_map=noise_map)

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.y.data.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (dataset_quantity.y.noise_map.slim == np.array([1.1, 2.1, 3.1, 4.1])).all()

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.x.data.slim == np.array([5.0, 6.0, 7.0, 8.0])).all()
    assert (dataset_quantity.x.noise_map.slim == np.array([5.1, 6.1, 7.1, 8.1])).all()


@pytest.fixture(name="test_data_path")
def make_test_data_path():
    test_data_path = path.join(
        "{}".format(os.path.dirname(os.path.realpath(__file__))),
        "files",
        "array",
        "output_test",
    )

    if os.path.exists(test_data_path):
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    return test_data_path


def test__output_to_fits(dataset_quantity_7x7_array_2d, test_data_path):
    dataset_quantity_7x7_array_2d.output_to_fits(
        data_path=path.join(test_data_path, "data.fits"),
        noise_map_path=path.join(test_data_path, "noise_map.fits"),
        overwrite=True,
    )

    data = ag.Array2D.from_fits(
        file_path=path.join(test_data_path, "data.fits"), hdu=0, pixel_scales=1.0
    )
    noise_map = ag.Array2D.from_fits(
        file_path=path.join(test_data_path, "noise_map.fits"), hdu=0, pixel_scales=1.0
    )

    assert (data.native == np.ones((7, 7))).all()
    assert (noise_map.native == 2.0 * np.ones((7, 7))).all()

    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]],
        pixel_scales=1.0,
    )

    noise_map = ag.VectorYX2D.no_mask(
        values=[[[1.1, 5.1], [2.1, 6.1]], [[3.1, 7.1], [4.1, 8.1]]],
        pixel_scales=1.0,
    )

    dataset_quantity = ag.DatasetQuantity(data=data, noise_map=noise_map)

    dataset_quantity.output_to_fits(
        data_path=path.join(test_data_path, "data.fits"),
        noise_map_path=path.join(test_data_path, "noise_map.fits"),
        overwrite=True,
    )

    data = ag.Array2D.from_fits(
        file_path=path.join(test_data_path, "data.fits"), hdu=0, pixel_scales=1.0
    )

    assert data[0, 0] == pytest.approx([1.0, 5.0], 1.0e-4)
