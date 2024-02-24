#pragma once

#include <stdexcept>
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
               static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        throw std::runtime_error("CUDA Runtime Error");
    }
}
#define checkCUDA(val) check((val), #val, __FILE__, __LINE__)

// Macros below are shared in both device and host
#define TOLERANCE 1e-16
// kernel1 macros
#define MAX_NUM_OF_EVERY_M_DATA_ELEMENT 512
#define MAX_M_DATA_MI 16
#define MAX_M_DATA_MJ 16
#define MAX_M_DATA_MK 16
#define MAX_NUM_OF_TOTAL_M_DATA \
    MAX_M_DATA_MI *MAX_M_DATA_MJ *MAX_M_DATA_MK *MAX_NUM_OF_EVERY_M_DATA_ELEMENT

// kernel2 macros
#define MAX_NUM_OF_LIG_TORSION 16
#define MAX_NUM_OF_FLEX_TORSION 1
#define MAX_NUM_OF_RIGID 24
#define MAX_NUM_OF_ATOMS 80
#define SIZE_OF_MOLEC_STRUC \
    ((3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) * sizeof(float))
#define SIZE_OF_CHANGE_STRUC \
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) * sizeof(float))
#define MAX_HESSIAN_MATRIX_D_SIZE                           \
    ((6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION) \
     * (6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) / 2)
#define MAX_NUM_OF_LIG_PAIRS 1024
#define MAX_NUM_OF_BFGS_STEPS 64
#define MAX_NUM_OF_RANDOM_MAP 1000  // not too large (stack overflow!)
#define GRIDS_SIZE 37               // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

#define MAX_NUM_OF_GRID_MI 128  // 55
#define MAX_NUM_OF_GRID_MJ 128  // 55
#define MAX_NUM_OF_GRID_MK 128  // 81
#define MAX_NUM_OF_GRID_POINT 512000

#define GRID_MI 65//55
#define GRID_MJ 71//55
#define GRID_MK 61//81
#define MAX_PRECAL_NUM_ATOM 30
#define MAX_P_DATA_M_DATA_SIZE \
    45150  // modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom,
        //    select n=140
#define MAX_NUM_OF_GRID_ATOMS 150
#define FAST_SIZE 2051  // modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
#define SMOOTH_SIZE 2051
#define MAX_CONTAINER_SIZE_EVERY_WI 5

#define MAX_THREAD 41700000  // modified for vina1.2, to calculate random map memory upper bound
#define MAX_LIGAND_NUM \
    10250  // modified for vina1.2, to calculate precalculate_byatom memory upper
        //    bound
struct SizeConfig {
static constexpr size_t MAX_NUM_OF_LIG_TORSION_ = 48;
static constexpr size_t MAX_NUM_OF_FLEX_TORSION_ = 1;
static constexpr size_t MAX_NUM_OF_RIGID_ = 128;
static constexpr size_t MAX_NUM_OF_ATOMS_ = 300;
static constexpr size_t SIZE_OF_MOLEC_STRUC_ =
((3 + 4 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t SIZE_OF_CHANGE_STRUC_ =
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t MAX_HESSIAN_MATRIX_D_SIZE_  =                         
    ((6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_) 
     * (6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) / 2);
static constexpr size_t MAX_NUM_OF_LIG_PAIRS_ =4096;
static constexpr size_t MAX_NUM_OF_BFGS_STEPS_ =64;
static constexpr size_t MAX_NUM_OF_RANDOM_MAP_= 1000  ;// not too large (stack overflow!)
static constexpr size_t GRIDS_SIZE_ =37   ;            // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

static constexpr size_t MAX_NUM_OF_GRID_MI_ =128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MJ_= 128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MK_ =128 ; // 81
static constexpr size_t MAX_NUM_OF_GRID_POINT_ =512000;

//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
static constexpr size_t MAX_PRECAL_NUM_ATOM_ =30;
static constexpr size_t MAX_P_DATA_M_DATA_SIZE_ =MAX_NUM_OF_ATOMS_*(MAX_NUM_OF_ATOMS_+1)/2;
// modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom, select n=140
//#define MAX_NUM_OF_GRID_ATOMS 150
static constexpr size_t FAST_SIZE_ =2051  ;// modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
static constexpr size_t SMOOTH_SIZE_ =2051;
static constexpr size_t MAX_CONTAINER_SIZE_EVERY_WI_ =5;

static constexpr size_t MAX_THREAD_ = 41700000 ; // modified for vina1.2, to calculate random map memory upper bound
static constexpr size_t MAX_LIGAND_NUM_  = 10250;
};
struct SmallConfig {
static constexpr size_t MAX_NUM_OF_LIG_TORSION_ = 10;
static constexpr size_t MAX_NUM_OF_FLEX_TORSION_ = 1;
static constexpr size_t MAX_NUM_OF_RIGID_ = 8;
static constexpr size_t MAX_NUM_OF_ATOMS_ = 50;
static constexpr size_t SIZE_OF_MOLEC_STRUC_ =
((3 + 4 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t SIZE_OF_CHANGE_STRUC_ =
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t MAX_HESSIAN_MATRIX_D_SIZE_  =                         
    ((6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_) 
     * (6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) / 2);
static constexpr size_t MAX_NUM_OF_LIG_PAIRS_ =300;
static constexpr size_t MAX_NUM_OF_BFGS_STEPS_ =64;
static constexpr size_t MAX_NUM_OF_RANDOM_MAP_= 1000  ;// not too large (stack overflow!)
static constexpr size_t GRIDS_SIZE_ =37   ;            // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

static constexpr size_t MAX_NUM_OF_GRID_MI_ =128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MJ_= 128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MK_ =128 ; // 81
static constexpr size_t MAX_NUM_OF_GRID_POINT_ =512000;

//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
static constexpr size_t MAX_PRECAL_NUM_ATOM_ =30;
static constexpr size_t MAX_P_DATA_M_DATA_SIZE_ =MAX_NUM_OF_ATOMS_*(MAX_NUM_OF_ATOMS_+1)/2;
// modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom, select n=140
//#define MAX_NUM_OF_GRID_ATOMS 150
static constexpr size_t FAST_SIZE_ =2051  ;// modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
static constexpr size_t SMOOTH_SIZE_ =2051;
static constexpr size_t MAX_CONTAINER_SIZE_EVERY_WI_ =5;

static constexpr size_t MAX_THREAD_ = 41700000 ; // modified for vina1.2, to calculate random map memory upper bound
static constexpr size_t MAX_LIGAND_NUM_  = 10250;
};
struct MediumConfig {
static constexpr size_t MAX_NUM_OF_LIG_TORSION_ = 12;
static constexpr size_t MAX_NUM_OF_FLEX_TORSION_ = 1;
static constexpr size_t MAX_NUM_OF_RIGID_ = 12;
static constexpr size_t MAX_NUM_OF_ATOMS_ = 50;
static constexpr size_t SIZE_OF_MOLEC_STRUC_ =
((3 + 4 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t SIZE_OF_CHANGE_STRUC_ =
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t MAX_HESSIAN_MATRIX_D_SIZE_  =                         
    ((6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_) 
     * (6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) / 2);
static constexpr size_t MAX_NUM_OF_LIG_PAIRS_ =330;
static constexpr size_t MAX_NUM_OF_BFGS_STEPS_ =64;
static constexpr size_t MAX_NUM_OF_RANDOM_MAP_= 1000  ;// not too large (stack overflow!)
static constexpr size_t GRIDS_SIZE_ =37   ;            // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

static constexpr size_t MAX_NUM_OF_GRID_MI_ =128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MJ_= 128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MK_ =128 ; // 81
static constexpr size_t MAX_NUM_OF_GRID_POINT_ =512000;

//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
static constexpr size_t MAX_PRECAL_NUM_ATOM_ =30;
static constexpr size_t MAX_P_DATA_M_DATA_SIZE_ =MAX_NUM_OF_ATOMS_*(MAX_NUM_OF_ATOMS_+1)/2;
// modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom, select n=140
//#define MAX_NUM_OF_GRID_ATOMS 150
static constexpr size_t FAST_SIZE_ =2051  ;// modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
static constexpr size_t SMOOTH_SIZE_ =2051;
static constexpr size_t MAX_CONTAINER_SIZE_EVERY_WI_ =5;

static constexpr size_t MAX_THREAD_ = 41700000 ; // modified for vina1.2, to calculate random map memory upper bound
static constexpr size_t MAX_LIGAND_NUM_  = 10250;
};
struct LargeConfig {
static constexpr size_t MAX_NUM_OF_LIG_TORSION_ = 14;
static constexpr size_t MAX_NUM_OF_FLEX_TORSION_ = 1;
static constexpr size_t MAX_NUM_OF_RIGID_ = 14;
static constexpr size_t MAX_NUM_OF_ATOMS_ = 50;
static constexpr size_t SIZE_OF_MOLEC_STRUC_ =
((3 + 4 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t SIZE_OF_CHANGE_STRUC_ =
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t MAX_HESSIAN_MATRIX_D_SIZE_  =                         
    ((6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_) 
     * (6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) / 2);
static constexpr size_t MAX_NUM_OF_LIG_PAIRS_ =512;
static constexpr size_t MAX_NUM_OF_BFGS_STEPS_ =64;
static constexpr size_t MAX_NUM_OF_RANDOM_MAP_= 1000  ;// not too large (stack overflow!)
static constexpr size_t GRIDS_SIZE_ =37   ;            // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

static constexpr size_t MAX_NUM_OF_GRID_MI_ =128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MJ_= 128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MK_ =128 ; // 81
static constexpr size_t MAX_NUM_OF_GRID_POINT_ =512000;

//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
static constexpr size_t MAX_PRECAL_NUM_ATOM_ =30;
static constexpr size_t MAX_P_DATA_M_DATA_SIZE_ =MAX_NUM_OF_ATOMS_*(MAX_NUM_OF_ATOMS_+1)/2;
// modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom, select n=140
//#define MAX_NUM_OF_GRID_ATOMS 150
static constexpr size_t FAST_SIZE_ =2051  ;// modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
static constexpr size_t SMOOTH_SIZE_ =2051;
static constexpr size_t MAX_CONTAINER_SIZE_EVERY_WI_ =5;

static constexpr size_t MAX_THREAD_ = 41700000 ; // modified for vina1.2, to calculate random map memory upper bound
static constexpr size_t MAX_LIGAND_NUM_  = 10250;
};
struct ExtraLargeConfig {
static constexpr size_t MAX_NUM_OF_LIG_TORSION_ = 16;
static constexpr size_t MAX_NUM_OF_FLEX_TORSION_ = 1;
static constexpr size_t MAX_NUM_OF_RIGID_ = 24;
static constexpr size_t MAX_NUM_OF_ATOMS_ = 80;
static constexpr size_t SIZE_OF_MOLEC_STRUC_ =
((3 + 4 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t SIZE_OF_CHANGE_STRUC_ =
    ((3 + 3 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) * sizeof(float));
static constexpr size_t MAX_HESSIAN_MATRIX_D_SIZE_  =                         
    ((6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_) 
     * (6 + MAX_NUM_OF_LIG_TORSION_ + MAX_NUM_OF_FLEX_TORSION_ + 1) / 2);
static constexpr size_t MAX_NUM_OF_LIG_PAIRS_ =1024;
static constexpr size_t MAX_NUM_OF_BFGS_STEPS_ =64;
static constexpr size_t MAX_NUM_OF_RANDOM_MAP_= 1000  ;// not too large (stack overflow!)
static constexpr size_t GRIDS_SIZE_ =37   ;            // larger than vina1.1, max(XS_TYPE_SIZE, AD_TYPE_SIZE + 2)

static constexpr size_t MAX_NUM_OF_GRID_MI_ =128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MJ_= 128;  // 55
static constexpr size_t MAX_NUM_OF_GRID_MK_ =128 ; // 81
static constexpr size_t MAX_NUM_OF_GRID_POINT_ =512000;

//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
static constexpr size_t MAX_PRECAL_NUM_ATOM_ =30;
static constexpr size_t MAX_P_DATA_M_DATA_SIZE_ =MAX_NUM_OF_ATOMS_*(MAX_NUM_OF_ATOMS_+1)/2;
// modified for vina1.2, should be larger, n*(n+1)/2, n=num_of_atom, select n=140
//#define MAX_NUM_OF_GRID_ATOMS 150
static constexpr size_t FAST_SIZE_ =2051  ;// modified for vina1.2 m_max_cutoff^2 * factor + 3, ad4=13424
static constexpr size_t SMOOTH_SIZE_ =2051;
static constexpr size_t MAX_CONTAINER_SIZE_EVERY_WI_ =5;

static constexpr size_t MAX_THREAD_ = 41700000 ; // modified for vina1.2, to calculate random map memory upper bound
static constexpr size_t MAX_LIGAND_NUM_  = 10250;
};
typedef struct {
    float data[GRIDS_SIZE];
} affinities_cuda_t;

typedef struct {
    int types[4];
    float coords[3];
} atom_cuda_t;

typedef struct {
    atom_cuda_t atoms[MAX_NUM_OF_ATOMS];
} grid_atoms_cuda_t;

typedef struct {
    float coords[MAX_NUM_OF_ATOMS][3];
} m_coords_cuda_t;

typedef struct {
    float coords[MAX_NUM_OF_ATOMS][3];
} m_minus_forces_t;

typedef struct {  // namely molec_struc
    float position[3];
    float orientation[4];
    float lig_torsion[MAX_NUM_OF_LIG_TORSION];
    float flex_torsion[MAX_NUM_OF_FLEX_TORSION];
    float lig_torsion_size;
    float coords[MAX_NUM_OF_ATOMS][3];
    float e;
} output_type_cuda_t;

typedef struct {  // namely change_struc
    float position[3];
    float orientation[3];
    float lig_torsion[MAX_NUM_OF_LIG_TORSION];
    float flex_torsion[MAX_NUM_OF_FLEX_TORSION];
    float lig_torsion_size;
} change_cuda_t;

typedef struct {  // depth-first order
    int atom_range[MAX_NUM_OF_RIGID][2];
    float origin[MAX_NUM_OF_RIGID][3];
    float orientation_m[MAX_NUM_OF_RIGID][9];  // This matrix is fixed to 3*3
    float orientation_q[MAX_NUM_OF_RIGID][4];

    float axis[MAX_NUM_OF_RIGID][3];             // 1st column is root node, all 0s
    float relative_axis[MAX_NUM_OF_RIGID][3];    // 1st column is root node, all 0s
    float relative_origin[MAX_NUM_OF_RIGID][3];  // 1st column is root node, all 0s

    int parent[MAX_NUM_OF_RIGID];                             // every node has only 1 parent node
    bool children_map[MAX_NUM_OF_RIGID][MAX_NUM_OF_RIGID];    // chidren_map[i][j] = true if node
                                                              // i's child is node j
    bool descendant_map[MAX_NUM_OF_RIGID][MAX_NUM_OF_RIGID];  // descendant_map[i][j] = true if
                                                              // node i is ancestor of node j
    int num_children;
} rigid_cuda_t;

typedef struct {
    int type_pair_index[MAX_NUM_OF_LIG_PAIRS];
    int a[MAX_NUM_OF_LIG_PAIRS];
    int b[MAX_NUM_OF_LIG_PAIRS];
    int num_pairs;
} lig_pairs_cuda_t;

typedef struct {
    lig_pairs_cuda_t pairs;
    rigid_cuda_t rigid;
    int begin;
    int end;
} ligand_cuda_t;

typedef struct {
    int int_map[MAX_NUM_OF_RANDOM_MAP];
    float pi_map[MAX_NUM_OF_RANDOM_MAP];
    float sphere_map[MAX_NUM_OF_RANDOM_MAP][3];
} random_maps_t;

typedef struct {
    atom_cuda_t atoms[MAX_NUM_OF_ATOMS];
    m_coords_cuda_t m_coords;
    m_minus_forces_t minus_forces;
    ligand_cuda_t ligand;
    int m_num_movable_atoms;  // will be -1 if ligand parsing failed
} m_cuda_t;

typedef struct {
    float m_init[3];
    float m_range[3];
    float m_factor[3];
    float m_dim_fl_minus_1[3];
    float m_factor_inv[3];
    int m_i;
    int m_j;
    int m_k;
    float m_data[MAX_NUM_OF_GRID_POINT];
} grid_cuda_t;

typedef struct {
    int atu;
    float slope;
    grid_cuda_t grids[GRIDS_SIZE];
} ig_cuda_t;

typedef struct {
    float ptmp[MAX_NUM_OF_RIGID][3];
    float p[MAX_NUM_OF_RIGID][3];
    float o[MAX_NUM_OF_RIGID][3];
} pot_cuda_t;

typedef struct {
    float fast[FAST_SIZE];
    float smooth[SMOOTH_SIZE][2];
    float factor;
} p_m_data_cuda_t;

typedef struct {
    float m_cutoff_sqr;
    int n;
    float factor;
    int m_data_size;
    p_m_data_cuda_t *m_data;
} p_cuda_t;

typedef struct {
    float m_cutoff_sqr;
    int n;
    float factor;
    int m_data_size;
    p_m_data_cuda_t m_data[MAX_P_DATA_M_DATA_SIZE];
} p_cuda_t_cpu;

typedef struct {
    int max_steps;
    float average_required_improvement;
    int over;
    int ig_grids_m_data_step;
    int p_data_m_data_step;
    int atu;
    int m_num_movable_atoms;
    float slope;
    float epsilon_fl;
    float epsilon_fl2;
    float epsilon_fl3;
    float epsilon_fl4;
    float epsilon_fl5;
} variables_bfgs;

typedef struct {
    output_type_cuda_t container[MAX_CONTAINER_SIZE_EVERY_WI];
    int current_size;
} output_container_cuda_t;

// symmetric matrix_d (only half of it are stored)
typedef struct {
    float data[MAX_HESSIAN_MATRIX_D_SIZE];
    int dim;
} matrix_d;

template <typename Config>
struct matrix_d_ {
    float data[Config::MAX_HESSIAN_MATRIX_D_SIZE_];
    int dim;
} ;

template <typename Config>
struct affinities_cuda_t_ {
    float data[Config::GRIDS_SIZE_];
} ;

template <typename Config>
struct grid_atoms_cuda_t_ {
    atom_cuda_t atoms[Config::MAX_NUM_OF_ATOMS_];
} ;
template <typename Config>
struct m_coords_cuda_t_ {
    float coords[Config::MAX_NUM_OF_ATOMS_][3];
} ;
template <typename Config>
struct m_minus_forces_t_ {
    float coords[Config::MAX_NUM_OF_ATOMS_][3];
} ;
template <typename Config>
struct output_type_cuda_t_ {  // namely molec_struc
    float position[3];
    float orientation[4];
    float lig_torsion[Config::MAX_NUM_OF_LIG_TORSION_];
    float flex_torsion[Config::MAX_NUM_OF_FLEX_TORSION_];
    float lig_torsion_size;
    float coords[Config::MAX_NUM_OF_ATOMS_][3];
    float e;
} ;


template <typename Config>
struct change_cuda_t_{  // namely change_struc
    float position[3];
    float orientation[3];
    float lig_torsion[Config::MAX_NUM_OF_LIG_TORSION_];
    float flex_torsion[Config::MAX_NUM_OF_FLEX_TORSION_];
    float lig_torsion_size;
} ;
template <typename Config>
struct rigid_cuda_t_{  // depth-first order
    int atom_range[Config::MAX_NUM_OF_RIGID_][2];
    float origin[Config::MAX_NUM_OF_RIGID_][3];
    float orientation_m[Config::MAX_NUM_OF_RIGID_][9];  // This matrix is fixed to 3*3
    float orientation_q[Config::MAX_NUM_OF_RIGID_][4];

    float axis[Config::MAX_NUM_OF_RIGID_][3];             // 1st column is root node, all 0s
    float relative_axis[Config::MAX_NUM_OF_RIGID_][3];    // 1st column is root node, all 0s
    float relative_origin[Config::MAX_NUM_OF_RIGID_][3];  // 1st column is root node, all 0s

    int parent[Config::MAX_NUM_OF_RIGID_] ={0};  // every node has only 1 parent node
    bool children_map[Config::MAX_NUM_OF_RIGID_]
                     [Config::MAX_NUM_OF_RIGID_];  // chidren_map[i][j] = true if node i's child is node j
    bool descendant_map[Config::MAX_NUM_OF_RIGID_][Config::MAX_NUM_OF_RIGID_];
    int num_children;

} ;

template <typename Config>
struct lig_pairs_cuda_t_{
    int type_pair_index[Config::MAX_NUM_OF_LIG_PAIRS_];
    int a[Config::MAX_NUM_OF_LIG_PAIRS_];
    int b[Config::MAX_NUM_OF_LIG_PAIRS_];
    int num_pairs;
} ;
template <typename Config>
struct ligand_cuda_t_ {
    lig_pairs_cuda_t_<Config> pairs;
    rigid_cuda_t_<Config> rigid;
    int begin;
    int end;
} ;
template <typename Config>
struct random_maps_t_{
    int int_map[Config::MAX_NUM_OF_RANDOM_MAP_];
    float pi_map[Config::MAX_NUM_OF_RANDOM_MAP_];
    float sphere_map[Config::MAX_NUM_OF_RANDOM_MAP_][3];
} ;

template <typename Config>
struct m_cuda_t_{
    atom_cuda_t atoms[Config::MAX_NUM_OF_ATOMS_];
    m_coords_cuda_t_<Config> m_coords;
    m_minus_forces_t_<Config> minus_forces;
    ligand_cuda_t_<Config> ligand;
    int m_num_movable_atoms;  // will be -1 if ligand parsing failed
} ;
template <typename Config>
struct grid_cuda_t_{
    float m_init[3];
    float m_range[3];
    float m_factor[3];
    float m_dim_fl_minus_1[3];
    float m_factor_inv[3];
    int m_i;
    int m_j;
    int m_k;
    float m_data[Config::MAX_NUM_OF_GRID_POINT_];
} ;
template <typename Config>
struct ig_cuda_t_ {
    int atu;
    float slope;
    grid_cuda_t_<Config> grids[Config::GRIDS_SIZE_];
} ;
template <typename Config>
struct p_m_data_cuda_t_{
    float fast[Config::FAST_SIZE_];
    float smooth[Config::SMOOTH_SIZE_][2];
    float factor;
} ;
template <typename Config>
struct p_cuda_t_ {
    float m_cutoff_sqr;
    int n;
    float factor;
    int m_data_size;
    p_m_data_cuda_t_<Config> *m_data;
} ;
template <typename Config>
struct p_cuda_t_cpu_ {
    float m_cutoff_sqr;
    int n;
    float factor;
    int m_data_size;
    p_m_data_cuda_t_<Config> m_data[Config::MAX_P_DATA_M_DATA_SIZE_];
} ;

template <typename Config>
struct output_container_cuda_t_{
    output_type_cuda_t_<Config> container[Config::MAX_CONTAINER_SIZE_EVERY_WI_];
    int current_size;
} ;
template <typename Config>
struct precalculate_element_cuda_t_ {
    float fast[Config::FAST_SIZE_];
    float smooth[Config::SMOOTH_SIZE_][2];  // smooth
    float factor;
} ;
template <typename Config>
struct pot_cuda_t_{
    float ptmp[Config::MAX_NUM_OF_RIGID_][3];
    float p[Config::MAX_NUM_OF_RIGID_][3];
    float o[Config::MAX_NUM_OF_RIGID_][3];
} ;