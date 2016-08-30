#!/bin/bash

cd $RECIPE_DIR
# echo "Building !!!!" `pwd`
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
