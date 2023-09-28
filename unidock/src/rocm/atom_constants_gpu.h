#ifndef VINA_ATOM_CONSTANTS_H

// based on SY_TYPE_* but includes H
__device__ __constant__ sz EL_TYPE_H = 0;
__device__ __constant__ sz EL_TYPE_C = 1;
__device__ __constant__ sz EL_TYPE_N = 2;
__device__ __constant__ sz EL_TYPE_O = 3;
__device__ __constant__ sz EL_TYPE_S = 4;
__device__ __constant__ sz EL_TYPE_P = 5;
__device__ __constant__ sz EL_TYPE_F = 6;
__device__ __constant__ sz EL_TYPE_Cl = 7;
__device__ __constant__ sz EL_TYPE_Br = 8;
__device__ __constant__ sz EL_TYPE_I = 9;
__device__ __constant__ sz EL_TYPE_Si = 10;  // Silicon
__device__ __constant__ sz EL_TYPE_At = 11;  // Astatine
__device__ __constant__ sz EL_TYPE_Met = 12;
__device__ __constant__ sz EL_TYPE_Dummy = 13;
__device__ __constant__ sz EL_TYPE_SIZE = 14;

// AutoDock4
__device__ __constant__ sz AD_TYPE_C = 0;
__device__ __constant__ sz AD_TYPE_A = 1;
__device__ __constant__ sz AD_TYPE_N = 2;
__device__ __constant__ sz AD_TYPE_O = 3;
__device__ __constant__ sz AD_TYPE_P = 4;
__device__ __constant__ sz AD_TYPE_S = 5;
__device__ __constant__ sz AD_TYPE_H = 6;  // non-polar hydrogen
__device__ __constant__ sz AD_TYPE_F = 7;
__device__ __constant__ sz AD_TYPE_I = 8;
__device__ __constant__ sz AD_TYPE_NA = 9;
__device__ __constant__ sz AD_TYPE_OA = 10;
__device__ __constant__ sz AD_TYPE_SA = 11;
__device__ __constant__ sz AD_TYPE_HD = 12;
__device__ __constant__ sz AD_TYPE_Mg = 13;
__device__ __constant__ sz AD_TYPE_Mn = 14;
__device__ __constant__ sz AD_TYPE_Zn = 15;
__device__ __constant__ sz AD_TYPE_Ca = 16;
__device__ __constant__ sz AD_TYPE_Fe = 17;
__device__ __constant__ sz AD_TYPE_Cl = 18;
__device__ __constant__ sz AD_TYPE_Br = 19;
__device__ __constant__ sz AD_TYPE_Si = 20;  // Silicon
__device__ __constant__ sz AD_TYPE_At = 21;  // Astatine
__device__ __constant__ sz AD_TYPE_G0 = 22;  // closure of cyclic molecules
__device__ __constant__ sz AD_TYPE_G1 = 23;
__device__ __constant__ sz AD_TYPE_G2 = 24;
__device__ __constant__ sz AD_TYPE_G3 = 25;
__device__ __constant__ sz AD_TYPE_CG0 = 26;
__device__ __constant__ sz AD_TYPE_CG1 = 27;
__device__ __constant__ sz AD_TYPE_CG2 = 28;
__device__ __constant__ sz AD_TYPE_CG3 = 29;
__device__ __constant__ sz AD_TYPE_W = 30;  // hydrated ligand
__device__ __constant__ sz AD_TYPE_SIZE = 31;

// X-Score
__device__ __constant__ sz XS_TYPE_C_H = 0;
__device__ __constant__ sz XS_TYPE_C_P = 1;
__device__ __constant__ sz XS_TYPE_N_P = 2;
__device__ __constant__ sz XS_TYPE_N_D = 3;
__device__ __constant__ sz XS_TYPE_N_A = 4;
__device__ __constant__ sz XS_TYPE_N_DA = 5;
__device__ __constant__ sz XS_TYPE_O_P = 6;
__device__ __constant__ sz XS_TYPE_O_D = 7;
__device__ __constant__ sz XS_TYPE_O_A = 8;
__device__ __constant__ sz XS_TYPE_O_DA = 9;
__device__ __constant__ sz XS_TYPE_S_P = 10;
__device__ __constant__ sz XS_TYPE_P_P = 11;
__device__ __constant__ sz XS_TYPE_F_H = 12;
__device__ __constant__ sz XS_TYPE_Cl_H = 13;
__device__ __constant__ sz XS_TYPE_Br_H = 14;
__device__ __constant__ sz XS_TYPE_I_H = 15;
__device__ __constant__ sz XS_TYPE_Si = 16;  // Silicon
__device__ __constant__ sz XS_TYPE_At = 17;  // Astatine
__device__ __constant__ sz XS_TYPE_Met_D = 18;
__device__ __constant__ sz XS_TYPE_C_H_CG0 = 19;  // closure of cyclic molecules
__device__ __constant__ sz XS_TYPE_C_P_CG0 = 20;
__device__ __constant__ sz XS_TYPE_G0 = 21;
__device__ __constant__ sz XS_TYPE_C_H_CG1 = 22;
__device__ __constant__ sz XS_TYPE_C_P_CG1 = 23;
__device__ __constant__ sz XS_TYPE_G1 = 24;
__device__ __constant__ sz XS_TYPE_C_H_CG2 = 25;
__device__ __constant__ sz XS_TYPE_C_P_CG2 = 26;
__device__ __constant__ sz XS_TYPE_G2 = 27;
__device__ __constant__ sz XS_TYPE_C_H_CG3 = 28;
__device__ __constant__ sz XS_TYPE_C_P_CG3 = 29;
__device__ __constant__ sz XS_TYPE_G3 = 30;
__device__ __constant__ sz XS_TYPE_W = 31;  // hydrated ligand
__device__ __constant__ sz XS_TYPE_SIZE = 32;

// DrugScore-CSD
__device__ __constant__ sz SY_TYPE_C_3 = 0;
__device__ __constant__ sz SY_TYPE_C_2 = 1;
__device__ __constant__ sz SY_TYPE_C_ar = 2;
__device__ __constant__ sz SY_TYPE_C_cat = 3;
__device__ __constant__ sz SY_TYPE_N_3 = 4;
__device__ __constant__ sz SY_TYPE_N_ar = 5;
__device__ __constant__ sz SY_TYPE_N_am = 6;
__device__ __constant__ sz SY_TYPE_N_pl3 = 7;
__device__ __constant__ sz SY_TYPE_O_3 = 8;
__device__ __constant__ sz SY_TYPE_O_2 = 9;
__device__ __constant__ sz SY_TYPE_O_co2 = 10;
__device__ __constant__ sz SY_TYPE_S = 11;
__device__ __constant__ sz SY_TYPE_P = 12;
__device__ __constant__ sz SY_TYPE_F = 13;
__device__ __constant__ sz SY_TYPE_Cl = 14;
__device__ __constant__ sz SY_TYPE_Br = 15;
__device__ __constant__ sz SY_TYPE_I = 16;
__device__ __constant__ sz SY_TYPE_Met = 17;
__device__ __constant__ sz SY_TYPE_SIZE = 18;

#endif

struct atom_kind_gpu {
    char name[4];
    fl radius;
    fl depth;
    fl hb_depth;  // pair (i,j) is HB if hb_depth[i]*hb_depth[j] < 0
    fl hb_radius;
    fl solvation;
    fl volume;
    fl covalent_radius;  // from
                         // http://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
};

__device__ bool strcmp(const char* a, const char* b) {
    int i = 0;
    while (true) {
        if (a[i] == 0 && b[i] == 0)
            break;
        else
            return false;
        if (a[i] == b[i])
            ++i;
        else
            return false;
    }
    return true;
}

// generated from edited AD4_parameters.data using a script,
// then covalent radius added from en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
__device__ atom_kind_gpu atom_kind_data_gpu[] = {
    // name, radius, depth, hb_depth, hb_r, solvation, volume, covalent radius
    {"C", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77},    //  0
    {"A", 2.00000, 0.15000, 0.0, 0.0, -0.00052, 33.51030, 0.77},    //  1
    {"N", 1.75000, 0.16000, 0.0, 0.0, -0.00162, 22.44930, 0.75},    //  2
    {"O", 1.60000, 0.20000, 0.0, 0.0, -0.00251, 17.15730, 0.73},    //  3
    {"P", 2.10000, 0.20000, 0.0, 0.0, -0.00110, 38.79240, 1.06},    //  4
    {"S", 2.00000, 0.20000, 0.0, 0.0, -0.00214, 33.51030, 1.02},    //  5
    {"H", 1.00000, 0.02000, 0.0, 0.0, 0.00051, 0.00000, 0.37},      //  6
    {"F", 1.54500, 0.08000, 0.0, 0.0, -0.00110, 15.44800, 0.71},    //  7
    {"I", 2.36000, 0.55000, 0.0, 0.0, -0.00110, 55.05850, 1.33},    //  8
    {"NA", 1.75000, 0.16000, -5.0, 1.9, -0.00162, 22.44930, 0.75},  //  9
    {"OA", 1.60000, 0.20000, -5.0, 1.9, -0.00251, 17.15730, 0.73},  // 10
    {"SA", 2.00000, 0.20000, -1.0, 2.5, -0.00214, 33.51030, 1.02},  // 11
    {"HD", 1.00000, 0.02000, 1.0, 0.0, 0.00051, 0.00000, 0.37},     // 12
    {"Mg", 0.65000, 0.87500, 0.0, 0.0, -0.00110, 1.56000, 1.30},    // 13
    {"Mn", 0.65000, 0.87500, 0.0, 0.0, -0.00110, 2.14000, 1.39},    // 14
    {"Zn", 0.74000, 0.55000, 0.0, 0.0, -0.00110, 1.70000, 1.31},    // 15
    {"Ca", 0.99000, 0.55000, 0.0, 0.0, -0.00110, 2.77000, 1.74},    // 16
    {"Fe", 0.65000, 0.01000, 0.0, 0.0, -0.00110, 1.84000, 1.25},    // 17
    {"Cl", 2.04500, 0.27600, 0.0, 0.0, -0.00110, 35.82350, 0.99},   // 18
    {"Br", 2.16500, 0.38900, 0.0, 0.0, -0.00110, 42.56610, 1.14},   // 19
    {"Si", 2.30000, 0.20000, 0.0, 0.0, -0.00143, 50.96500, 1.11},   // 20
    {"At", 2.40000, 0.55000, 0.0, 0.0, -0.00110, 57.90580, 1.44},   // 21
    {"G0", 0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000, 0.77},     // 22
    {"G1", 0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000, 0.77},     // 23
    {"G2", 0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000, 0.77},     // 24
    {"G3", 0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000, 0.77},     // 25
    {"CG0", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77},  // 26
    {"CG1", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77},  // 27
    {"CG2", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77},  // 28
    {"CG3", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77},  // 29
    {"W", 0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000, 0.00}       // 30
};

__device__ __constant__ fl metal_solvation_parameter_gpu = -0.00110;

__device__ __constant__ fl metal_covalent_radius_gpu
    = 1.75;  // for metals not on the list // FIXME this info should be moved to non_ad_metals

__device__ __constant__ sz atom_kinds_size_gpu = sizeof(atom_kind_data_gpu) / sizeof(atom_kind_gpu);

struct atom_equivalence_gpu {
    char name[4];
    char to[4];
};

__device__ __constant__ atom_equivalence_gpu atom_equivalence_data_gpu[] = {{"Se", "S"}};

__device__ __constant__ sz atom_equivalences_size_gpu
    = sizeof(atom_equivalence_data_gpu) / sizeof(atom_equivalence_gpu);

__device__ __constant__ acceptor_kind acceptor_kind_data_gpu[]
    = {  // ad_type, optimal length, depth
        {AD_TYPE_NA, 1.9, 5.0},
        {AD_TYPE_OA, 1.9, 5.0},
        {AD_TYPE_SA, 2.5, 1.0}};

__device__ __constant__ sz acceptor_kinds_size_gpu
    = sizeof(acceptor_kind_data_gpu) / sizeof(acceptor_kind);

__device__ __forceinline__ bool ad_is_hydrogen_gpu(sz ad) {
    return ad == AD_TYPE_H || ad == AD_TYPE_HD;
}

__device__ __forceinline__ bool ad_is_heteroatom_gpu(
    sz ad) {  // returns false for ad >= AD_TYPE_SIZE
    return ad != AD_TYPE_A && ad != AD_TYPE_C && ad != AD_TYPE_H && ad != AD_TYPE_HD
           && ad < AD_TYPE_SIZE;
}

__device__ __forceinline__ sz ad_type_to_el_type_gpu(sz t) {
    switch (t) {
        case AD_TYPE_C:
            return EL_TYPE_C;
        case AD_TYPE_A:
            return EL_TYPE_C;
        case AD_TYPE_N:
            return EL_TYPE_N;
        case AD_TYPE_O:
            return EL_TYPE_O;
        case AD_TYPE_P:
            return EL_TYPE_P;
        case AD_TYPE_S:
            return EL_TYPE_S;
        case AD_TYPE_H:
            return EL_TYPE_H;
        case AD_TYPE_F:
            return EL_TYPE_F;
        case AD_TYPE_I:
            return EL_TYPE_I;
        case AD_TYPE_NA:
            return EL_TYPE_N;
        case AD_TYPE_OA:
            return EL_TYPE_O;
        case AD_TYPE_SA:
            return EL_TYPE_S;
        case AD_TYPE_HD:
            return EL_TYPE_H;
        case AD_TYPE_Mg:
            return EL_TYPE_Met;
        case AD_TYPE_Mn:
            return EL_TYPE_Met;
        case AD_TYPE_Zn:
            return EL_TYPE_Met;
        case AD_TYPE_Ca:
            return EL_TYPE_Met;
        case AD_TYPE_Fe:
            return EL_TYPE_Met;
        case AD_TYPE_Cl:
            return EL_TYPE_Cl;
        case AD_TYPE_Br:
            return EL_TYPE_Br;
        case AD_TYPE_Si:
            return EL_TYPE_Si;
        case AD_TYPE_At:
            return EL_TYPE_At;
        case AD_TYPE_CG0:
            return EL_TYPE_C;
        case AD_TYPE_CG1:
            return EL_TYPE_C;
        case AD_TYPE_CG2:
            return EL_TYPE_C;
        case AD_TYPE_CG3:
            return EL_TYPE_C;
        case AD_TYPE_G0:
            return EL_TYPE_Dummy;
        case AD_TYPE_G1:
            return EL_TYPE_Dummy;
        case AD_TYPE_G2:
            return EL_TYPE_Dummy;
        case AD_TYPE_G3:
            return EL_TYPE_Dummy;
        case AD_TYPE_W:
            return EL_TYPE_Dummy;
        case AD_TYPE_SIZE:
            return EL_TYPE_SIZE;
        default:
            VINA_CHECK_GPU(false);
    }
    return EL_TYPE_SIZE;  // to placate the compiler in case of warnings - it should never get here
                          // though
}

__device__ __constant__ fl xs_vdw_radii_gpu[] = {
    1.9,  // C_H
    1.9,  // C_P
    1.8,  // N_P
    1.8,  // N_D
    1.8,  // N_A
    1.8,  // N_DA
    1.7,  // O_P
    1.7,  // O_D
    1.7,  // O_A
    1.7,  // O_DA
    2.0,  // S_P
    2.1,  // P_P
    1.5,  // F_H
    1.8,  // Cl_H
    2.0,  // Br_H
    2.2,  // I_H
    2.2,  // Si
    2.3,  // At
    1.2,  // Met_D
    1.9,  // C_H_CG0
    1.9,  // C_P_CG0
    1.9,  // C_H_CG1
    1.9,  // C_P_CG1
    1.9,  // C_H_CG2
    1.9,  // C_P_CG2
    1.9,  // C_H_CG3
    1.9,  // C_P_CG3
    0.0,  // G0
    0.0,  // G1
    0.0,  // G2
    0.0,  // G3
    0.0   // W
};

__device__ __constant__ fl xs_vinardo_vdw_radii_gpu[] = {
    2.0,  // C_H
    2.0,  // C_P
    1.7,  // N_P
    1.7,  // N_D
    1.7,  // N_A
    1.7,  // N_DA
    1.6,  // O_P
    1.6,  // O_D
    1.6,  // O_A
    1.6,  // O_DA
    2.0,  // S_P
    2.1,  // P_P
    1.5,  // F_H
    1.8,  // Cl_H
    2.0,  // Br_H
    2.2,  // I_H
    2.2,  // Si
    2.3,  // At
    1.2,  // Met_D
    2.0,  // C_H_CG0
    2.0,  // C_P_CG0
    2.0,  // C_H_CG1
    2.0,  // C_P_CG1
    2.0,  // C_H_CG2
    2.0,  // C_P_CG2
    2.0,  // C_H_CG3
    2.0,  // C_P_CG3
    0.0,  // G0
    0.0,  // G1
    0.0,  // G2
    0.0,  // G3
    0.0   // W
};

__device__ __forceinline__ fl xs_radius_gpu(sz t) {
    sz n = sizeof(xs_vdw_radii_gpu) / sizeof(fl);
    assert(n == XS_TYPE_SIZE);
    assert(t < n);
    return xs_vdw_radii_gpu[t];
}

__device__ __forceinline__ fl xs_vinardo_radius_gpu(sz t) {
    sz n = sizeof(xs_vdw_radii_gpu) / sizeof(fl);
    assert(n == XS_TYPE_SIZE);
    assert(t < n);
    return xs_vinardo_vdw_radii_gpu[t];
}

__device__ __constant__ char non_ad_metal_names_gpu[][4] = {  // expand as necessary
    "Cu", "Fe", "Na", "K", "Hg", "Co", "U", "Cd", "Ni"};

__device__ __forceinline__ bool is_non_ad_metal_name_gpu(const char name[4]) {
    const sz s = sizeof(non_ad_metal_names_gpu) / (sizeof(char) * 4);
    VINA_FOR(i, s)
    if (strcmp(non_ad_metal_names_gpu[i], name) == 0) return true;
    return false;
}

__device__ __forceinline__ bool xs_is_hydrophobic_gpu(sz xs) {
    return xs == XS_TYPE_C_H || xs == XS_TYPE_F_H || xs == XS_TYPE_Cl_H || xs == XS_TYPE_Br_H
           || xs == XS_TYPE_I_H;
}

__device__ __forceinline__ bool xs_is_acceptor_gpu(sz xs) {
    return xs == XS_TYPE_N_A || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_A || xs == XS_TYPE_O_DA;
}

__device__ __forceinline__ bool xs_is_donor_gpu(sz xs) {
    return xs == XS_TYPE_N_D || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_D || xs == XS_TYPE_O_DA
           || xs == XS_TYPE_Met_D;
}

__device__ __forceinline__ bool xs_donor_acceptor_gpu(sz t1, sz t2) {
    return xs_is_donor_gpu(t1) && xs_is_acceptor_gpu(t2);
}

__device__ __forceinline__ bool xs_h_bond_possible_gpu(sz t1, sz t2) {
    return xs_donor_acceptor_gpu(t1, t2) || xs_donor_acceptor_gpu(t2, t1);
}

__device__ __forceinline__ atom_kind_gpu ad_type_property_gpu(sz i) {
    assert(AD_TYPE_SIZE == atom_kinds_size_gpu);
    assert(i < atom_kinds_size_gpu);
    return atom_kind_data_gpu[i];
}

// Using string literal
__device__ __forceinline__ sz string_to_ad_type_gpu(
    const char* name) {  // returns AD_TYPE_SIZE if not found (no exceptions thrown, because metals
                         // unknown to AD4 are not exceptional)
    VINA_FOR(i, atom_kinds_size_gpu)
    if (strcmp(atom_kind_data_gpu[i].name, name) == 0) return i;
    VINA_FOR(i, atom_equivalences_size_gpu)
    if (strcmp(atom_equivalence_data_gpu[i].name, name) == 0)
        return string_to_ad_type_gpu(atom_equivalence_data_gpu[i].to);
    return AD_TYPE_SIZE;
}

__device__ __forceinline__ fl max_covalent_radius_gpu() {
    fl tmp = 1.74;  // Ca
    return tmp;
}
