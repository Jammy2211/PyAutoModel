import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_interferometer"


def test__interferometer_generator_from_aggregator__analysis_has_single_dataset(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
    samples,
    model,
):
    interferometer_7 = ag.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        transformer_class=ag.TransformerDFT,
    )

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    dataset_agg = ag.agg.InterferometerAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == interferometer_7.data).all()
        assert (dataset_list[0].real_space_mask == mask_2d_7x7).all()
        assert isinstance(dataset_list[0].transformer, ag.TransformerDFT)

    clean(database_file=database_file)


def test__interferometer_generator_from_aggregator__analysis_multi(
    analysis_interferometer_7,
    samples,
    model,
):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_interferometer_7 + analysis_interferometer_7,
        model=model,
        samples=samples,
    )

    dataset_agg = ag.agg.InterferometerAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == analysis_interferometer_7.dataset.data).all()
        assert (dataset_list[1].data == analysis_interferometer_7.dataset.data).all()

    clean(database_file=database_file)
