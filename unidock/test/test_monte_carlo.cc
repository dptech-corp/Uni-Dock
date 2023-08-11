#include "vina.h"
#include "scoring_function.h"
#include "precalculate.h"
#include "omp.h"
#include "gtest/gtest.h"
// #include "vector.h"
// #include "string.h"

TEST(vina_gpu, monte_carlo){
    Vina v("vina");
    v.set_receptor("receptor/1iep_receptor.pdbqt");
    v.compute_vina_maps(15.19, 53.903, 16.917, 20, 20, 20);
    std::vector<std::string> gpu_batch_ligand_names;
    std::string ligand_name("ligands/1iep_ligand.pdbqt");
    gpu_batch_ligand_names.push_back(ligand_name);
    v.set_ligand_from_file_gpu(gpu_batch_ligand_names);
    v.global_search_gpu(1024, 9, 1, 0, 20, 1);
    EXPECT_LE(v.m_poses_gpu[0][0].e, -11);
}