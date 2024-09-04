#!/bin/sh

export CMAKE_ARGS=$@

# Install STK first
(cd third_party/stk && python setup.py install)
# Install deform
python -m build