.. targets-start-do-not-remove

.. _`CMake`: https://cmake.org/cmake/help/v3.14/
.. _`Doxygen`: https://www.doxygen.nl/manual/docblocks.html
.. _`LaTeX`: https://www.latex-project.org/help/documentation/
.. _`pipreqs`: https://github.com/bndr/pipreqs
.. _`Anaconda Documentation`: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _`Eigen`: https://eigen.tuxfamily.org/dox/
.. _`BOOST`: https://www.boost.org/doc/libs/1_53_0/
.. _`Sphinx`: https://www.sphinx-doc.org/en/master/
.. _`Sphinx style guide`: https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html
.. _`PEP-8`: https://www.python.org/dev/peps/pep-0008/
.. _`gersemi`: https://github.com/BlankSpruce/gersemi
.. _`clang-tidy`: https://clang.llvm.org/extra/clang-tidy/
.. _`clang-format`: https://clang.llvm.org/docs/ClangFormat.html

.. targets-end-do-not-remove

#######################
tardigrade_vector_tools
#######################

*******************
Project Description
*******************

A collection of tools for C++ that make interfacing with vectors easier and
less prone to error. These tools also allow the user access to the powerful
Eigen library which provides matrix utilities in such a way that Eigen does
not need to be used explicitly in the user's code.

Information
===========

TODO

Developers
==========

* Nathan Miller Nathan.A.Miller@colorado.edu
* Kyle Brindley kbrindley@lanl.gov

************
Dependencies
************

The developer dependencies are found in ``environment.txt``.

.. code-block:: bash

   $ conda create --name tardigrade_vector_tools-dev --file environment.txt

**************************
Building the documentation
**************************

.. warning::

   **API Health Note**: The Sphinx API docs are a work-in-progress. The doxygen
   API is much more useful

.. code-block:: bash

   $ pwd
   /path/to/tardigrade_vector_tools
   $ cmake -S . -B build
   $ cmake --build build --target Doxygen Sphinx

*****************
Build the library
*****************

Vector tools is always header only. There is nothing to build.

****************
Test the library
****************

.. code-block:: bash

   $ pwd
   /path/to/tardigrade_vector_tools
   $ cmake -S . -B build
   $ cmake --build build --target test_tardigrade_vector_tools
   $ ctest --test-dir build

*******************
Install the library
*******************

Build the entire project before performing the installation.

4) Build the entire project

   .. code-block:: bash

      $ pwd
      /path/to/tardigrade_vector_tools
      $ cmake -S . -B build
      $ cmake --build build --target all

5) Install the library

   .. code-block:: bash

      $ pwd
      /path/to/tardigrade_vector_tools
      $ cmake --install build --prefix path/to/root/install

      # Example local user (non-admin) Linux install
      $ cmake --install build --prefix /home/$USER/.local

      # Example install to conda environment
      $ conda activate my_env
      $ cmake --install build --prefix ${CONDA_PREFIX}

***********************
Build the Conda package
***********************

.. code-block:: bash

   $ VERSION=$(python -m setuptools_scm) conda mambabuild recipe --no-anaconda-upload -c conda-forge --output-folder conda-bld

***********************
Contribution Guidelines
***********************

.. contribution-start-do-not-remove

Git Commit Message
==================

Begin Git commit messages with one of the following headings:

* BUG: bug fix
* DOC: documentation
* FEAT: feature
* MAINT: maintenance
* TST: tests
* REL: release
* WIP: work-in-progress

For example:

.. code-block:: bash

   git commit -m "DOC: adds documentation for feature"

Git Branch Names
================

When creating branches use one of the following naming conventions. When in
doubt use ``feature/<description>``.

* ``bugfix/\<description>``
* ``feature/\<description>``
* ``release/\<description>``

reStructured Text
=================

`Sphinx`_ reads in docstrings and other special portions of the code as reStructured text. Developers should follow
styles in this Sphinx style guide`_.

Style Guide
===========

This project uses the `gersemi`_ CMake linter. The CI style guide check runs the following command

.. code-block:
   $ gersemi CMakeLists.txt src/ docs/ --check
and any automatic fixes may be reviewed and then applied by developers with the following commands

.. code-block:
   $ gersemi CMakeLists.txt src/ docs/ --diff
   $ gersemi CMakeLists.txt src/ docs/ --in-place

This project enforces its style using `clang-tidy`_ and `clang-format`_ as configured with the
`.clang-format` and `.clang-tidy` files in the root directory. The formatting of the project can be
checked using `clang-tidy`_ by first configuring the project using

.. code-block:

   $ cmake -S . -B build ... -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

where `...` are the other configuration flags specified. After this clang-tidy can be run on the
full project from the source directory via

.. CAUTION::
    Commit all changes prior to running the clang tidy command. This will edit all source files.

.. code-block:

   $ run-clang-tidy -config-file=.clang-tidy -header-filter=*.h -p build

The formatting can be checked using `clang-format`_ by running

.. code-block:

   $ cmake -S . -B build ...
   $ cmake --build build --target cpp-format-check

which will indicate if the formatting is correct. The c++ files can be re-formatted to match the
style guidance by running

.. CAUTION::
    Commit all changes prior to running the format command. This will edit all source files.

.. code-block

   $ cmake --build build --target cpp-format

If the style is not constrained by the above, it should be inferred by the surrounding code.
Wherever a style can't be inferred from surrounding code this project falls back to `PEP-8`_-like
styles the exceptions to the notional PEP-8 fall back:

1. `Doxygen`_ style docstrings are required for automated, API from source documentation.

.. contribution-end-do-not-remove
