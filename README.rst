========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |coveralls| |codecov|
        | |landscape| |scrutinizer| |codacy| |codeclimate|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/mann2/badge/?style=flat
    :target: https://readthedocs.org/projects/mann2
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/chendaniely/mann2.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/chendaniely/mann2

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/chendaniely/mann2?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/chendaniely/mann2

.. |requires| image:: https://requires.io/github/chendaniely/mann2/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/chendaniely/mann2/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/chendaniely/mann2/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/chendaniely/mann2

.. |codecov| image:: https://codecov.io/github/chendaniely/mann2/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/chendaniely/mann2

.. |landscape| image:: https://landscape.io/github/chendaniely/mann2/master/landscape.svg?style=flat
    :target: https://landscape.io/github/chendaniely/mann2/master
    :alt: Code Quality Status

.. |codacy| image:: https://img.shields.io/codacy/REPLACE_WITH_PROJECT_ID.svg?style=flat
    :target: https://www.codacy.com/app/chendaniely/mann2
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/chendaniely/mann2/badges/gpa.svg
   :target: https://codeclimate.com/github/chendaniely/mann2
   :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/mann2.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/mann2

.. |downloads| image:: https://img.shields.io/pypi/dm/mann2.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/mann2

.. |wheel| image:: https://img.shields.io/pypi/wheel/mann2.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/mann2

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/mann2.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/mann2

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/mann2.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/mann2

.. |scrutinizer| image:: https://img.shields.io/scrutinizer/g/chendaniely/mann2/master.svg?style=flat
    :alt: Scrutinizer Status
    :target: https://scrutinizer-ci.com/g/chendaniely/mann2/


.. end-badges

Multi Agent Neural Network

* Free software: MIT license

Installation
============

::

    # pip install mann2
    pip install -e .

Documentation
=============

https://mann2.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
