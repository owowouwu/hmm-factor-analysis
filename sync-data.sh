#!/bin/bash

# sync results files
rsync -Pcauv --include "*/" --include "*.png" --include "*.csv" --include "*.json" --exclude "*" psnguyen@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim0614/steven/master-thesis/data data
rsync data/ -Pcauv psnguyen@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim0614/steven/master-thesis/data