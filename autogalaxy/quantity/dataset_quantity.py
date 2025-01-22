import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import autoarray as aa
from autoarray.dataset.abstract.dataset import AbstractDataset

logger = logging.getLogger(__name__)


class DatasetQuantity(AbstractDataset):
    def __init__(
        self,
        data: Union[aa.Array2D, aa.VectorYX2D],
        noise_map: Union[aa.Array2D, aa.VectorYX2D],
        over_sample_size_lp: Union[int, aa.Array2D] = 4,
        over_sample_size_pixelization: Union[int, aa.Array2D] = 4,
    ):
        """
        A quantity dataset, which represents a derived quantity of a light profile, mass profile, galaxy or galaxies
        that can be fitted with a model via a non-linear.

        For example, the `DatasetQuantity` could represent the `convergence` of a mass profile, and this dataset could
        then be fit with a galaxies via the `AnalysisQuantity` class. The benefit of doing this is that the components
        of different models can be fitted to one another and matched for example, the power-law model whose convergence
        best matches the convergence of a dark matter profile could be inferred).

        The following quantities of the data are available and used for the following tasks:

        - `data`: The quantity data, which shows the signal that is analysed and fitted with a model data of a model
        object.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        Datasets also contains following properties:

        - `grid`: A grids of (y,x) coordinates which align with the image pixels, whereby each coordinate corresponds to
        the centre of an image pixel. This may be used in fits to calculate the model image of the imaging data.

        - `grids.pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `invserion` module, which often use different
        oversampling and sub-size values to the grid above.

        The `over_sampling` and `over_sampling_pixelization` define how over sampling is performed for these grids.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

        Parameters
        ----------
        data
            The data of the quantity (e.g. 2D convergence, 2D potential, 2D deflections) that is fitted.
        noise_map
            The 2D noise map of the quantity's data.
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit, which is often chosen in an arbitrary way for a quantity dataset given the quantities
            are not observed using real astronomical instruments.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """
        if data.shape != noise_map.shape:
            if data.shape[0:-1] == noise_map.shape[0:]:
                noise_map = aa.VectorYX2D.no_mask(
                    values=np.stack((noise_map, noise_map), axis=-1),
                    pixel_scales=data.pixel_scales,
                    shape_native=data.shape_native,
                    origin=data.origin,
                )

        self.unmasked = None

        super().__init__(
            data=data,
            noise_map=noise_map,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

    @classmethod
    def via_signal_to_noise_map(
        cls,
        data: Union[aa.Array2D, aa.VectorYX2D],
        signal_to_noise_map: Union[aa.Array2D],
        over_sample_size_lp: Union[int, aa.Array2D] = 4,
        over_sample_size_pixelization: Union[int, aa.Array2D] = 4,
    ):
        """
        Represents a derived quantity of a light profile, mass profile, galaxy or galaxies as a dataset that can be
        fitted with a model via a non-linear (see `DatasetQuantity.__init__`).

        This classmethod takes as input a signal-to-noise map, as opposed to the noise map used in the `__init__`
        constructor. The noise-map is then derived from this signal-to-noise map, such that this is the signal to
        noise of the `DatasetQuantity` that is returned.

        Parameters
        ----------
        data
            The data of the quantity (e.g. 2D convergence, 2D potential, 2D deflections) that is fitted.
        signal_to_noise_map
            The 2D signal to noise map of the quantity's data.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """
        try:
            noise_map = data / signal_to_noise_map
        except ValueError:
            noise_map = aa.VectorYX2D.zeros(
                shape_native=data.shape_native, pixel_scales=data.pixel_scales
            )
            noise_map = noise_map.apply_mask(mask=data.mask)

            signal_to_noise_map[signal_to_noise_map < 1e-8] = 1e-8

            noise_map[:, 0] = np.abs(data.slim[:, 0]) / signal_to_noise_map
            noise_map[:, 1] = np.abs(data.slim[:, 1]) / signal_to_noise_map

        return DatasetQuantity(
            data=data,
            noise_map=noise_map,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

    @property
    def y(self) -> "DatasetQuantity":
        """
        If the `DatasetQuantity` contains a `VectorYX2D` as its data, this property returns a new `DatasetQuantity`
        with just the y-values of the vectors as the data, alongside the noise-map.

        This is primarily used for visualizing a fit to the `DatasetQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return DatasetQuantity(
                data=self.data.y,
                noise_map=self.noise_map.y,
                over_sample_size_lp=self.over_sample_size_lp,
                over_sample_size_pixelization=self.over_sample_size_pixelization,
            )

    @property
    def x(self) -> "DatasetQuantity":
        """
        If the `DatasetQuantity` contains a `VectorYX2D` as its data, this property returns a new `DatasetQuantity`
        with just the x-values of the vectors as the data, alongside the noise-map.

        This is primarily used for visualizing a fit to the `DatasetQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return DatasetQuantity(
                data=self.data.x,
                noise_map=self.noise_map.x,
                over_sample_size_lp=self.over_sample_size_lp,
                over_sample_size_pixelization=self.over_sample_size_pixelization,
            )

    def apply_mask(self, mask: aa.Mask2D) -> "DatasetQuantity":
        """
        Apply a mask to the quantity dataset, whereby the mask is applied to the data and noise-map one-by-one.

        The original unmasked qunatity dataset is stored as the `self.unmasked` attribute. This is used to ensure that
        if the `apply_mask` function is called multiple times, every mask is always applied to the original unmasked
        imaging dataset.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.
        """
        if self.data.mask.is_all_false:
            unmasked_dataset = self
        else:
            unmasked_dataset = self.unmasked

        data = self.data.apply_mask(mask=mask)
        noise_map = self.noise_map.apply_mask(mask=mask)
        over_sample_size_lp = self.over_sample_size_lp.apply_mask(mask=mask)
        over_sample_size_pixelization = self.over_sample_size_pixelization.apply_mask(
            mask=mask
        )

        dataset = DatasetQuantity(
            data=data,
            noise_map=noise_map,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

        dataset.unmasked = unmasked_dataset

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return dataset

    @property
    def shape_native(self):
        return self.data.shape_native

    @property
    def pixel_scales(self):
        return self.data.pixel_scales

    def output_to_fits(
        self,
        data_path: Union[Path, str],
        noise_map_path: Optional[Union[Path, str]] = None,
        overwrite: bool = False,
    ):
        """
        Output a quantity dataset to multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`) the path to
        the .fits can be specified, with `hdu=0` assumed automatically.

        If the `data` has been masked, the masked data is output to .fits files. A mask can be separately output to
        a file `mask.fits` via the `Mask` objects `output_to_fits` method.

        Parameters
        ----------
        data_path
            The path to the data .fits file where the image data is output (e.g. '/path/to/data.fits').
        noise_map_path
            The path to the noise_map .fits where the noise_map is output (e.g. '/path/to/noise_map.fits').
        overwrite
            If `True`, the .fits files are overwritten if they already exist, if `False` they are not and an
            exception is raised.
        """
        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
