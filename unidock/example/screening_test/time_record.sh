#!/bin/bash

{
    echo -n "Command: unidock; Date: "
    date
    time ./unidock --receptor ./indata/def.pdbqt --ligand_index def_ligands_30_torsions_3_num_245.txt \
    --center_x -36.01 --center_y 25.63 --center_z 67.49 --size_x 22 --size_y 22 --size_z 22 \
    --dir ./result/def --exhaustiveness 128 --max_step 20 --num_modes 9  --scoring vina \
    --refine_step 3 --seed 5
} 2>> time_results.txt
