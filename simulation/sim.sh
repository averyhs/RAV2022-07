#!/bin/bash

python -m cellsium simulate \
    -t Height=60.0 \
    -t Width=60.0 \
    \
    -t SimulationDuration=21.0 \
    -t SimulationOutputInterval=0.5 \
    \
    -t GroundTruthOnlyCompleteCells=False \
    -t GroundTruthOnlyCompleteCellsInImages=False \
    \
    -t NewCellCount=3 \
    -t NewCellRadiusFromCenter=20.0 \
    \
    -t ChipmunkPlacementRadius=0.0001 \
    \
    -t NewCellLength1Mean=3.5 \
    -t NewCellLength1Std=0.2 \
    -t NewCellLength2Mean=2.5 \
    -t NewCellLength2Std=0.15 \
    -t NewCellLengthAbsoluteMax=6 \
    -t NewCellLengthAbsoluteMin=1 \
    \
    -t NewCellWidthMean=0.7 \
    -t NewCellWidthStd=0.1 \
    -t NewCellWidthAbsoluteMax=1.0 \
    -t NewCellWidthAbsoluteMin=0.3 \
    \
    -t NewCellBendLowerLower=-0.007 \
    -t NewCellBendLowerUpper=0.007 \
    -t NewCellBendOverallLower=-0.007 \
    -t NewCellBendOverallUpper=0.007 \
    -t NewCellBendUpperLower=-0.007 \
    -t NewCellBendUpperUpper=0.007 \
    \
    -t LuminanceBackground=0.35 \
    -t LuminanceCell=0.2 \
    \
    -t Seed=6 \
    \
    -o simulate \
    --Output GenericMaskOutput \
    --Output COCOOutput \
    \
    -c mycomodel.py:Cell \
    -p
