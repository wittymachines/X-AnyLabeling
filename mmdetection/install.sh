#!/bin/bash
# This script installs the necessary dependencies for the mmdetection library.

echo "Installing tensorboard and setuptools"
pip install -r mmdetect_0.txt

echo "Installing torch"
pip install -r mmdetect_1.txt

echo "Installing mmdetection dependencies"
pip install -r mmdetect_2.txt

echo "Installing patched version of mmdetection"
pip install git+https://github.com/wittymachines/mmdetection.git@fix_version_check

echo "Install deploy dependencies"
pip install -r mmdetect_3.txt


