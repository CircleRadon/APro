#!/usr/bin/env bash

pip install -r requirements/build.txt

pip install -v -e . #Or  python setup develop

cd apro/gp_cuda/ #compile for the global affinity propagation
python setup.py build develop


