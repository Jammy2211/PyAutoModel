.. _pip:

Installation with pip
=====================

Install
-------

We strongly recommend that you install **PyAutoGalaxy** in a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_, with the link attached
describing what a virtual environment is and how to create one.

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoGalaxy** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autogalaxy

You may get warnings which state something like:

.. code-block:: bash

    ERROR: autoarray 2024.11.6.1 has requirement numpy<=1.22.1, but you'll have numpy 1.22.2 which is incompatible.
    ERROR: numba 0.53.1 has requirement llvmlite<0.37,>=0.36.0rc1, but you'll have llvmlite 0.38.0 which is incompatible.

If you see these messages, they do not mean that the installation has failed and the instructions below will
identify clearly if the installation is a success.

If there are no errors (but only the warnings above) **PyAutoGalaxy** is installed!

If there is an error check out the `troubleshooting section <https://pyautogalaxy.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Numba
-----

Numba (https://numba.pydata.org)  is an optional library which makes **PyAutoGalaxy** run a lot faster, which we
strongly recommend users have installed.

You can install numba via the following command:

.. code-block:: bash

    pip install numba

Some users have experienced difficulties installing numba, which is why it is an optional library. If your
installation is not successful, you can use **PyAutoGalaxy** without it installed for now, to familiarize yourself
with the software and determine if it is the right software for you.

If you decide that **PyAutoGalaxy** is the right software, then I recommend you commit the time to getting a
successful numba install working, with more information provided `at this readthedocs page <https://pyautogalaxy.readthedocs.io/en/latest/installation/numba.html>`_

Workspace
---------

Next, clone the ``autogalaxy workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autogalaxy_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autogalaxy_workspace
   git clone https://github.com/Jammy2211/autogalaxy_workspace --depth 1
   cd autogalaxy_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

It should be clear that **PyAutoGalaxy** runs without issue.

If there is an error check out the `troubleshooting section <https://pyautogalaxy.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Optional
--------

For interferometer analysis there are two optional dependencies that must be installed via the commands:

.. code-block:: bash

    pip install pynufft
    pip install pylops==1.11.1

**PyAutoGalaxy** will run without these libraries and it is recommended that you only install them if you intend to
do interferometer analysis.

If you run interferometer code a message explaining that you need to install these libraries will be printed, therefore
it is safe not to install them initially.