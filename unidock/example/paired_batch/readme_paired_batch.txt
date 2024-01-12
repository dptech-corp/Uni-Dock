The below example command has to be run from the Uni-Dock/unidock folder, so that the executable is available at build/unidock.

build/unidock --paired_batch_size 10 --ligand_index example/paired_batch/paired_batch_config.json --size_x 25 --size_y 25 --size_z 25 --dir test/prof_1024 --exhaustiveness 1024 --max_step 60 --seed 5

This example command runs docking proposals on 4 receptor:ligand pairs, as defined in the paired_batch_config.json, within a box of 25, and stores the generated ligand poses in test/prof_1024. 
