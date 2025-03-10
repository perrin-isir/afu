#!/usr/bin/env python
# -*- coding: utf-8 -*-


import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

NAME = "AFU"
DESCRIPTION = "The AFU algorithm within a fork of rljax (https://github.com/ku2482/rljax)."
URL = ""
EMAIL = "perrin@isir.upmc.fr"
AUTHOR = "Nicolas Perrin-Gilbert"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.0.6"

here = os.path.abspath(os.path.dirname(__file__))
REQUIRED = open(os.path.join(here, "requirements.txt")).read().splitlines()
EXTRAS = {}

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))
        
        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[package for package in find_packages() if package.startswith("afu_rljax")],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    cmdclass={
        "upload": UploadCommand,
    },
)
