#include "vina.h"
#include "scoring_function.h"
#include "precalculate.h"
#include "omp.h"
#include "gtest/gtest.h"
// #include "vector.h"
// #include "string.h"

TEST(vina_gpu, precalculate){
    Vina v("vina");
    v.set_receptor("receptor/1iep_receptor.pdbqt");
    v.compute_vina_maps(15.19, 53.903, 16.917, 20, 20, 20);
    std::vector<std::string> gpu_batch_ligand_names;
    std::string ligand_name("ligands/1iep_ligand.pdbqt");
    gpu_batch_ligand_names.push_back(ligand_name);
    v.set_ligand_from_file_gpu(gpu_batch_ligand_names);
    v.global_search_gpu(1,1,1,0,5);

    Vina v_cpu("vina");
    v_cpu.set_receptor("receptor/1iep_receptor.pdbqt");
    v_cpu.compute_vina_maps(15.19, 53.903, 16.917, 20, 20, 20);
    v_cpu.set_ligand_from_file(ligand_name);

    /* the precalculate result on CPU is in m_precalculated_byatom
       the precalculate result on GPU is in m_precalculated_byatom_gpu[0] */

    // // unittest printing, only check the first ligand
    // printf("energies about the first ligand on GPU in vina.cpp:\n");
    // for (int i = 0;i < 10; ++i){
    //     printf("precalculated_byatom.m_data.m_data[%d]: (smooth.first, smooth.second, fast) ", i);
    //     for (int j = 0;j < 20; ++j){
    //         printf("(%f, %f, %f) ", v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first,
    //         v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].second, v.m_precalculated_byatom_gpu[0].m_data.m_data[i].fast[j]);
    //     }
    //     printf("\n");
    // }

    // printf("energies about the first ligand on CPU in vina.cpp:\n");
    // for (int i = 0;i < 10; ++i){
    //     printf("precalculated_byatom.m_data.m_data[%d]: (smooth.first, smooth.second, fast) ", i);
    //     for (int j = 0;j < 20; ++j){
    //         printf("(%f, %f, %f) ", v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].first,
    //         v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].second, v_cpu.m_precalculated_byatom.m_data.m_data[i].fast[j]);
    //     }
    //     printf("\n");
    // }
    for (int i = 0;i < v.m_precalculated_byatom_gpu[0].m_data.m_data.size(); ++i)
        for (int j = 0;j < v.m_precalculated_byatom_gpu[0].m_data.m_data[0].smooth.size();++j){
            if (v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first < 0){
                EXPECT_GE((float)v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first * 0.99999, (float)v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].first);
                EXPECT_LE((float)v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first * 1.00001, (float)v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].first);
            }
            else{
                EXPECT_LE((float)v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first * 0.99999, (float)v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].first);
                EXPECT_GE((float)v.m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first * 1.00001, (float)v_cpu.m_precalculated_byatom.m_data.m_data[i].smooth[j].first);
            }
       }

}