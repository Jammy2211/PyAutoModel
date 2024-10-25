.. _overview_1_start_here:

Start Here
==========

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/hstcombined.png
  :width: 400
  :alt: Alternative text

**PyAutoGalaxy** has three core aims:

- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse Fitting, Irregular Meshes) that go beyond just simple Sersic fitting (which is supported too!).

- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be fitted independently or simultaneously.

- **Big Data**: Scaling automated analysis to extremely large datasets, using tools like an SQL database to build a scalable scientific workflow.

This page gives an overview of **PyAutoGalaxy**'s API, with follow up overview pages describing how to navigate the autogalaxy workspace and the advanced features of the software.

Imports
-------

Lets first import autogalaxy, its plotting module and the other libraries we'll need.

You'll see these imports in the majority of workspace examples.

.. code:: python

    import autogalaxy as ag
    import autogalaxy.plot as aplt

    import matplotlib.pyplot as plt
    from os import path

Lets illustrate a simple galaxy structure calculations creating an an image of a galaxy using a light profile.

Grid
----

The emission of light from a galaxy is described using the ``Grid2D`` data structure, which is two-dimensional
Cartesian grids of (y,x) coordinates where the light profile of the galaxy is evaluated on the grid.

We make and plot a uniform Cartesian grid:

.. code:: python

    grid = ag.Grid2D.uniform(
        shape_native=(150, 150),  # The [pixels x pixels] shape of the grid in 2D.
        pixel_scales=0.05,  # The pixel-scale describes the conversion from pixel units to arc-seconds.
    )

    grid_plotter = aplt.Grid2DPlotter(grid=grid)
    grid_plotter.figure_2d()

The ``Grid2D`` looks like this:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/0_grid.png
  :width: 600
  :alt: Alternative text

Light Profiles
--------------

Our aim is to create an image of the morphological structures that make up a galaxy.

This uses analytic functions representing a galaxy's light, referred to as ``LightProfile`` objects. 

The most common light profile in Astronomy is the elliptical Sersic, which we create an instance of below:

.. code:: python

    sersic_light_profile = ag.lp.Sersic(
        centre=(0.0, 0.0),  # The light profile centre [units of arc-seconds].
        ell_comps=(
            0.2,
            0.1,
        ),  # The light profile elliptical components [can be converted to axis-ratio and position angle].
        intensity=0.005,  # The overall intensity normalisation [units arbitrary and are matched to the data].
        effective_radius=2.0,  # The effective radius containing half the profile's total luminosity [units of arc-seconds].
        sersic_index=4.0,  # Describes the profile's shape [higher value -> more concentrated profile].
    )

By passing the light profile the ``grid``, we evaluate the light emitted at every (y,x) coordinate and therefore create 
an image of the Sersic light profile.

.. code:: python

    image = sersic_light_profile.image_2d_from(grid=grid)

Plotting
--------

The **PyAutoGalaxy** in-built plot module provides methods for plotting objects and their properties, like the image of
a light profile we just created.

By using a ``LightProfilePlotter`` to plot the light profile's image, the figured is improved. 

Its axis units are scaled to arc-seconds, a color-bar is added, its given a descriptive labels, etc.

The plot module is highly customizable and designed to make it straight forward to create clean and informative figures
for fits to large datasets.

.. code:: python

    light_profile_plotter = aplt.LightProfilePlotter(
        light_profile=sersic_light_profile, grid=grid
    )
    light_profile_plotter.figures_2d(image=True)

The light profile appears as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/1_image_2d.png
  :width: 600
  :alt: Alternative text

Galaxy
------

A ``Galaxy`` object is a collection of light profiles at a specific redshift.

This object is highly extensible and is what ultimately allows us to fit complex models to galaxy images.

Below, we combine the Sersic light profile above with an Exponential light profile to create a galaxy containing both
a bulge and disk component.

.. code:: python

    exponential_light_profile = ag.lp.Exponential(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.0), intensity=0.1, effective_radius=0.5
    )

    galaxy = ag.Galaxy(
        redshift=0.5, bulge=sersic_light_profile, disk=exponential_light_profile
    )


The ``GalaxyPlotter`` object plots the image of the galaxy, which is the sum of its bulge and disk light profiles.

.. code:: python

    galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy, grid=grid)
    galaxy_plotter.figures_2d(image=True)

The galaxy, with both a bulge and disk, appears as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/2_image_2d.png
  :width: 600
  :alt: Alternative text

One example of the plotter's customizability is the ability to plot the individual light profiles of the galaxy
on a subplot.

.. code:: python

    galaxy_plotter.subplot_of_light_profiles(image=True)

The light profiles appear as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/3_subplot_image.png
  :width: 600
  :alt: Alternative text


Galaxies
--------

The ``Galaxies`` object is a collection of galaxies at the same redshift.

In a moment, we will see it is integral to the model-fitting API. 

For now, lets use it to create an image of a pair of merging galaxies, noting that a more concise API for creating
the galaxy is used below where the ``Sersic`` is passed directly to the ``Galaxy`` object.

.. code:: python

    galaxy_1 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(
            centre=(0.5, 0.2), intensity=1.0, effective_radius=1.0, sersic_index=2.0
        ),
    )

    galaxies = ag.Galaxies(
        galaxies=[galaxy, galaxy_1],
    )

    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/4_image_2d.png
  :width: 600
  :alt: Alternative text

Extensibility
-------------

All of the objects we've introduced so far are highly extensible, for example a galaxy can be made up of any number of
light profiles and many galaxy objects can be combined into a galaxies object.

To further illustrate this, we create a merging galaxy system with 4 star forming clumps of light, using a 
``SersicSph`` profile to make each spherical.

.. code:: python

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
            intensity=0.2,
            effective_radius=0.8,
            sersic_index=4.0,
        ),
        disk=ag.lp.Exponential(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
            intensity=0.1,
            effective_radius=1.6,
        ),
        clump_0=ag.lp.SersicSph(centre=(1.0, 1.0), intensity=0.5, effective_radius=0.2),
        clump_1=ag.lp.SersicSph(centre=(0.5, 0.8), intensity=0.5, effective_radius=0.2),
        clump_2=ag.lp.SersicSph(centre=(-1.0, -0.7), intensity=0.5, effective_radius=0.2),
        clump_3=ag.lp.SersicSph(centre=(-1.0, 0.4), intensity=0.5, effective_radius=0.2),
    )

    galaxy_1 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(
            centre=(0.0, 1.0),
            ell_comps=(0.0, 0.1),
            intensity=0.1,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
    )

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

The image of the merging galaxy system appears as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/5_image_2d.png
  :width: 600
  :alt: Alternative text

Simulating Data
---------------

The galaxy images above are **not** what we would observe if we looked at the sky through a telescope.

In reality, images of galaxies are observed using a telescope and detector, for example a CCD Imaging device attached
to the Hubble Space Telescope.

To make images that look like realistic Astronomy data, we must account for the effects like how the length of the
exposure time change the signal-to-noise, how the optics of the telescope blur the galaxy's light and that
there is a background sky which also contributes light to the image and adds noise.

The ``SimulatorImaging`` object simulates this process, creating realistic CCD images of galaxies using the ``Imaging``
object.

.. code:: python

    simulator = ag.SimulatorImaging(
        exposure_time=300.0,
        background_sky_level=1.0,
        psf=ag.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=0.05),
        add_poisson_noise=True,
    )


Once we have a simulator, we can use it to create an imaging dataset which consists of an image, noise-map and 
Point Spread Function (PSF) by passing it a galaxies and grid.

This uses the galaxies above to create the image of the galaxy and then add the effects that occur during data
acquisition.

This data is used below to illustrate model-fitting, so lets simulate a very simple image of a galaxy using
just a single Sersic light profile.

.. code:: python

    galaxies = ag.Galaxies(
        galaxies=[
            ag.Galaxy(
                redshift=0.5,
                bulge=ag.lp.Sersic(
                    centre=(0.0, 0.0),
                    ell_comps=(0.1, 0.2),
                    intensity=1.0,
                    effective_radius=0.8,
                    sersic_index=2.0,
                ),
            )
        ]
    )

    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)


Observed Dataset
----------------

We now have an ``Imaging`` object, which is a realistic representation of the data we observe with a telescope.

We use the ``ImagingPlotter`` to plot the dataset, showing that it contains the observed image, but also other
import dataset attributes like the noise-map and PSF.

.. code:: python

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.figures_2d(data=True)

The observed dataset appears as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/6_data.png
  :width: 600
  :alt: Alternative text

If you have come to **PyAutoGalaxy** to perform interferometry, the API above is easily adapted to use
a ``SimulatorInterferometer`` object to simulate an ``Interferometer`` dataset instead.

However, you should finish reading this notebook before moving on to the interferometry examples, to get a full
overview of the core **PyAutoGalaxy** API.

Masking
-------

We are about to fit the data with a model, but first must define a mask, which defines the regions of the image that 
are used to fit the data and which regions are not.

We create a ``Mask2D`` object which is a 3.0" circle, whereby all pixels within this 3.0" circle are used in the 
model-fit and all pixels outside are omitted. 

Inspection of the dataset above shows that no signal from the galaxy is observed outside of this radius, so this is a 
sensible mask.

.. code:: python

    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native,  # The mask's shape must match the dataset's to be applied to it.
        pixel_scales=dataset.pixel_scales,  # It must also have the same pixel scales.
        radius=3.0,  # The mask's circular radius [units of arc-seconds].
    )

Combine the imaging dataset with the mask.

.. code:: python

    dataset = dataset.apply_mask(mask=mask)

When we plot a masked dataset, the removed regions of the image (e.g. outside the 3.0") are automatically set to zero
and the plot axis automatically zooms in around the mask.

.. code:: python

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.figures_2d(data=True)

Here is the masked dataset:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/7_data.png
  :width: 600
  :alt: Alternative text

Fitting
-------

We are now at the point a scientist would be after observing a galaxy - we have an image of it, have used to a mask to 
determine where we observe signal from the galaxy, but cannot make any quantitative statements about its morphology.

We therefore must now fit a model to the data. This model is a representation of the galaxy's light, and we seek a way
to determine whether a given model provides a good fit to the data.

A fit is performing using a ``FitImaging`` object, which takes a dataset and galaxies object as input and determine if 
the galaxies are a good fit to the data.

.. code:: python

    fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

The fit creates ``model_data``, which is the image of the galaxy including effects which change its appearance
during data acquisition.

For example, by plotting the fit's ``model_data`` and comparing it to the image of the galaxies obtained via
the ``GalaxiesPlotter``, we can see the model data has been blurred by the dataset's PSF.

.. code:: python

    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=fit.galaxies, grid=grid)
    galaxies_plotter.figures_2d(image=True)

    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.figures_2d(model_image=True)

The image and model image appear as follows:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/8_image_2d.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/9_model_image.png
  :width: 400
  :alt: Alternative text

The fit also creates the following:

 - The ``residual_map``: The ``model_image`` subtracted from the observed dataset``s ``image``.
 - The ``normalized_residual_map``: The ``residual_map ``divided by the observed dataset's ``noise_map``.
 - The ``chi_squared_map``: The ``normalized_residual_map`` squared.
 
We can plot all 3 of these on a subplot that also includes the data, signal-to-noise map and model data.

.. code:: python

    fit_plotter.subplot_fit()

In this example, the galaxies used to simulate the data are used to fit it, thus the fit is good and residuals are minimized,
as shown by the subplots below:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/10_subplot_fit.png
  :width: 600
  :alt: Alternative text

The overall quality of the fit is quantified with the ``log_likelihood``.

.. code:: python

    print(fit.log_likelihood)

If you are familiar with statistical analysis, this quick run-through of the fitting tools will make sense and you
will be familiar with concepts like model data, residuals and a likelihood. 

If you are less familiar with these concepts, I recommend you finish this notebook and then go to the fitting API
guide, which explains the concepts in more detail and provides a more thorough overview of the fitting tools.

The take home point is that **PyAutoGalaxy**'s API has extensive tools for fitting models to data and visualizing the
results, which is what makes it a powerful tool for studying the morphologies of galaxies.

Modeling
--------

The fitting tools above are used to fit a model to the data given an input set of galaxies. Above, we used the true
galaxies used to simulate the data to fit the data, but we do not know what this "truth" is in the real world and 
is therefore not something a real scientist can do.

Modeling is the processing of taking a dataset and inferring the model that best fits the data, for example
the galaxy light profile(s) that best fits the light observed in the data or equivalently the combination
of Sersic profile parameters that maximize the likelihood of the fit.

Galaxy modeling uses the probabilistic programming language **PyAutoFit**, an open-source project that allows complex 
model fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you 
are interested in developing your own software to perform advanced model-fitting:

https://github.com/rhayes777/PyAutoFit

We import **PyAutoFit** separately to **PyAutoGalaxy**:

.. code:: python

    import autofit as af

We now compose the galaxy model using ``af.Model`` objects. 

These behave analogously to the ``Galaxy``, ``Galaxies`` and ``LightProfile`` objects above, however when using a ``Model`` 
their parameter values are not specified and are instead determined by a fitting procedure.

We will fit our galaxy data with a model which has one galaxy where:

 - The galaxy's bulge is a ``Sersic`` light profile. 
 - The galaxy's disk is a ``Exponential`` light profile.
 - The redshift of the galaxy is fixed to 0.5.
 
The light profiles below are linear light profiles, input via the ``lp_linear`` module. These solve for the intensity of
the light profiles via linear algebra, making the modeling more efficient and accurate. They are explained in more
detail in other workspace examples, but are a key reason why modeling with **PyAutoGalaxy** performs well and
can scale to complex models.

.. code:: python

    galaxy_model = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.Sersic,  # Note the use of ``lp_linear`` instead of ``lp``.
        disk=ag.lp_linear.Exponential,  # This uses linear light profiles explained in the modeling ``start_here`` example.
    )


By printing the ``Model``'s we see that each parameters has a prior associated with it, which is used by the
model-fitting procedure to fit the model.

.. code:: python

    print(galaxy_model)


We input the galaxy model above into a ``Collection``, which is the model we will fit. 

Note how we could easily extend this object to compose more complex models containing many galaxies.

.. code:: python

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))

The ``info`` attribute shows the model information in a more readable format:

.. code:: python

    print(model.info)


We now choose the 'non-linear search', which is the fitting method used to determine the light profile parameters that 
best-fit the data.

In this example we use [nautilus](https://nautilus-sampler.readthedocs.io/en/stable/), a nested sampling algorithm 
that in our experience has proven very effective at galaxy modeling.

.. code:: python

    search = af.Nautilus(name="start_here")


To perform the model-fit, we create an ``AnalysisImaging`` object which contains the ``log_likelihood_function`` that the
non-linear search calls to fit the galaxy model to the data.

The ``AnalysisImaging`` object is expanded on in the modeling ``start_here`` example, but in brief performs many useful
associated with modeling, including outputting results to hard-disk and visualizing the results of the fit.

.. code:: python

    analysis = ag.AnalysisImaging(dataset=dataset)


To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
Nautilus samples, model parameters, visualization) to your computer's storage device.

Once a model-fit is running, **PyAutoGalaxy** outputs the results of the search to storage device on-the-fly. This
includes galaxy model parameter estimates with errors non-linear samples and the visualization of the best-fit galaxy
model inferred by the search so far.

.. code:: python

    result = search.fit(model=model, analysis=analysis)


The animation below shows a slide-show of the galaxy modeling procedure. Many galaxy models are fitted to the data over
and over, gradually improving the quality of the fit to the data and looking more and more like the observed image.

NOTE, the animation of a non-linear search shown below is for a strong gravitational lens using **PyAutoGalaxy**'s 
child project **PyAutoLens**. Updating the animation to show a galaxy model-fit is on the **PyAutoGalaxy** to-do list!

We can see that initial models give a poor fit to the data but gradually improve (increasing the likelihood) as more
iterations are performed.

.. image:: https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true
  :width: 600

![Lens Modeling Animation](https://github.com/Jammy2211/auto_files/blob/main/lensmodel.gif?raw=true "model")

**Credit: Amy Etherington**

Results
-------


The fit returns a ``Result`` object, which contains the best-fit galaxies and the full posterior information of the 
non-linear search, including all parameter samples, log likelihood values and tools to compute the errors on the 
galaxy model.

Using results is explained in full in the ``guides/results`` section of the workspace, but for a quick illustration
the code below shows how easy it is to plot the fit and posterior of the model.

.. code:: python

    fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

    plotter = aplt.NestPlotter(samples=result.samples)
    plotter.corner_cornerpy()

Here is an example corner plot of the model-fit, which shows the probability density function of every parameter in the
model:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_1/cornerplot.png
  :width: 600
  :alt: Alternative text

Wrap Up
-------

We have now completed the API overview of **PyAutoGalaxy**, including a brief introduction to the core API for
creating galaxies, simulating data, fitting data and performing galaxy modeling.

The next overview describes how a new user should navigate the **PyAutoGalaxy** workspace, which contains many examples
and tutorials, in order to get up and running with the software.