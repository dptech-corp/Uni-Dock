#include "math.h"

// Define GPU precalculate structures

/* atom related start */
#include "atom_constants_gpu.cuh"

/* atom related end */

/* potential related start */

__device__ __forceinline__ fl sqr(fl x) { return x * x; }

// Vina common functions
__device__ __forceinline__ fl slope_step_gpu(fl x_bad, fl x_good, fl x) {
    if (x_bad < x_good) {
        if (x <= x_bad) return 0;
        if (x >= x_good) return 1;
    } else {
        if (x >= x_bad) return 0;
        if (x <= x_good) return 1;
    }
    return (x - x_bad) / (x_good - x_bad);
}

__device__ __forceinline__ bool is_glue_type_gpu(sz xs_t) {
    if ((xs_t == XS_TYPE_G0) || (xs_t == XS_TYPE_G1) || (xs_t == XS_TYPE_G2)
        || (xs_t == XS_TYPE_G3))
        return true;
    return false;
}

__device__ __forceinline__ fl optimal_distance_gpu(sz xs_t1, sz xs_t2) {
    if (is_glue_type_gpu(xs_t1) || is_glue_type_gpu(xs_t2)) return 0.0;  // G0, G1, G2 or G3
    return xs_radius_gpu(xs_t1) + xs_radius_gpu(xs_t2);
}

__device__ __forceinline__ fl smooth_div_gpu(fl x, fl y) {
    if (std::abs(x) < epsilon_fl) return 0;
    if (std::abs(y) < epsilon_fl)
        return ((x * y > 0) ? max_fl : -max_fl);  // FIXME I hope -max_fl does not become NaN
    return x / y;
}

// Vinardo common functions
__device__ __forceinline__ fl optimal_distance_vinardo_gpu(sz xs_t1, sz xs_t2) {
    if (is_glue_type_gpu(xs_t1) || is_glue_type_gpu(xs_t2)) return 0.0;  // G0, G1, G2 or G3
    return xs_vinardo_radius_gpu(xs_t1) + xs_vinardo_radius_gpu(xs_t2);
}

// AD42 common functions
__device__ __forceinline__ fl smoothen_gpu(fl r, fl rij, fl smoothing) {
    fl out;
    smoothing *= 0.5;

    if (r > rij + smoothing)
        out = r - smoothing;
    else if (r < rij - smoothing)
        out = r + smoothing;
    else
        out = rij;

    return out;
}

__device__ __forceinline__ fl ad4_hb_eps_gpu(sz a) {
    if (a < AD_TYPE_SIZE) return ad_type_property_gpu(a).hb_depth;
    VINA_CHECK_GPU(false);
    return 0;  // placating the compiler
}

__device__ __forceinline__ fl ad4_hb_radius_gpu(sz t) {
    if (t < AD_TYPE_SIZE) return ad_type_property_gpu(t).hb_radius;
    VINA_CHECK_GPU(false);
    return 0;  // placating the compiler
}

__device__ __forceinline__ fl ad4_vdw_eps_gpu(sz a) {
    if (a < AD_TYPE_SIZE) return ad_type_property_gpu(a).depth;
    VINA_CHECK_GPU(false);
    return 0;  // placating the compiler
}

__device__ __forceinline__ fl ad4_vdw_radius_gpu(sz t) {
    if (t < AD_TYPE_SIZE) return ad_type_property_gpu(t).radius;
    VINA_CHECK_GPU(false);
    return 0;  // placating the compiler
}

// Macrocycle - Vina and AD42
__device__ __forceinline__ bool is_glued_gpu(sz xs_t1, sz xs_t2) {
    return (xs_t1 == XS_TYPE_G0 && xs_t2 == XS_TYPE_C_H_CG0)
           || (xs_t1 == XS_TYPE_G0 && xs_t2 == XS_TYPE_C_P_CG0)
           || (xs_t2 == XS_TYPE_G0 && xs_t1 == XS_TYPE_C_H_CG0)
           || (xs_t2 == XS_TYPE_G0 && xs_t1 == XS_TYPE_C_P_CG0) ||

           (xs_t1 == XS_TYPE_G1 && xs_t2 == XS_TYPE_C_H_CG1)
           || (xs_t1 == XS_TYPE_G1 && xs_t2 == XS_TYPE_C_P_CG1)
           || (xs_t2 == XS_TYPE_G1 && xs_t1 == XS_TYPE_C_H_CG1)
           || (xs_t2 == XS_TYPE_G1 && xs_t1 == XS_TYPE_C_P_CG1) ||

           (xs_t1 == XS_TYPE_G2 && xs_t2 == XS_TYPE_C_H_CG2)
           || (xs_t1 == XS_TYPE_G2 && xs_t2 == XS_TYPE_C_P_CG2)
           || (xs_t2 == XS_TYPE_G2 && xs_t1 == XS_TYPE_C_H_CG2)
           || (xs_t2 == XS_TYPE_G2 && xs_t1 == XS_TYPE_C_P_CG2) ||

           (xs_t1 == XS_TYPE_G3 && xs_t2 == XS_TYPE_C_H_CG3)
           || (xs_t1 == XS_TYPE_G3 && xs_t2 == XS_TYPE_C_P_CG3)
           || (xs_t2 == XS_TYPE_G3 && xs_t1 == XS_TYPE_C_H_CG3)
           || (xs_t2 == XS_TYPE_G3 && xs_t1 == XS_TYPE_C_P_CG3);
}

// Vina

__device__ __forceinline__ fl gauss_gpu(fl x, fl width) { return exp(-sqr(x / width)); };

__device__ __forceinline__ fl vina_gaussian_cuda_eval(sz t1, sz t2, fl r, fl cutoff, fl offset,
                                                      fl width) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    return gauss_gpu(r - (optimal_distance_gpu(t1, t2) + offset),
                     width);  // hard-coded to XS atom type
};

__device__ __forceinline__ fl vina_repulsion_cuda_eval(sz t1, sz t2, fl r, fl cutoff, fl offset) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    fl d = r - (optimal_distance_gpu(t1, t2) + offset);  // hard-coded to XS atom type
    if (d > 0.0) return 0.0;
    return d * d;
};

__device__ __forceinline__ fl vina_hydrophobic_cuda_eval(sz t1, sz t2, fl r, fl good, fl bad,
                                                         fl cutoff) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    if (xs_is_hydrophobic_gpu(t1) && xs_is_hydrophobic_gpu(t2))
        return slope_step_gpu(bad, good, r - optimal_distance_gpu(t1, t2));
    else
        return 0.0;
};

__device__ __forceinline__ fl vina_non_dir_h_bond_cuda_eval(sz t1, sz t2, fl r, fl good, fl bad,
                                                            fl cutoff) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    if (xs_h_bond_possible_gpu(t1, t2)){        
        if  ((t1 >= 32 && t1 <= 35) || (t2 >= 32 && t2 <= 35))
            return 10.0*slope_step_gpu(bad, good, r - optimal_distance_gpu(t1, t2));
        else 
            return slope_step_gpu(bad, good, r - optimal_distance_gpu(t1, t2));}
    return 0.0;
};

// Vinardo
__device__ __forceinline__ fl vinardo_gaussian_eval(sz t1, sz t2, fl r, fl offset, fl width,
                                                    fl cutoff) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    return gauss_gpu(r - (optimal_distance_vinardo_gpu(t1, t2) + offset),
                     width);  // hard-coded to XS atom type
};

__device__ __forceinline__ fl vinardo_repulsion_eval(sz t1, sz t2, fl r, fl cutoff, fl offset) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    fl d = r - (optimal_distance_vinardo_gpu(t1, t2) + offset);  // hard-coded to XS atom type
    if (d > 0.0) return 0.0;
    return d * d;
};

__device__ __forceinline__ fl vinardo_hydrophobic_eval(sz t1, sz t2, fl r, fl good, fl bad,
                                                       fl cutoff) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    if (xs_is_hydrophobic_gpu(t1) && xs_is_hydrophobic_gpu(t2))
        return slope_step_gpu(bad, good, r - optimal_distance_vinardo_gpu(t1, t2));
    else
        return 0.0;
};

__device__ __forceinline__ fl vinardo_non_dir_h_bond_eval(sz t1, sz t2, fl r, fl good, fl bad,
                                                          fl cutoff) {
    if (r >= cutoff) return 0.0;
    if ((t1 >= XS_TYPE_SIZE) || (t2 >= XS_TYPE_SIZE)) return 0.0;
    if (xs_h_bond_possible_gpu(t1, t2))
        return slope_step_gpu(bad, good, r - optimal_distance_vinardo_gpu(t1, t2));
    return 0.0;
};

// AD42
__device__ __forceinline__ fl ad4_electrostatic_eval(const fl a_charge, const fl b_charge, fl r,
                                                     fl cap, fl cutoff) {
    if (r >= cutoff) return 0.0;
    fl q1q2 = a_charge * b_charge * 332.0;
    fl B = 78.4 + 8.5525;
    fl lB = -B * 0.003627;
    fl diel = -8.5525 + (B / (1 + 7.7839 * std::exp(lB * r)));
    if (r < epsilon_fl)
        return q1q2 * cap / diel;
    else {
        return q1q2 * min(cap, 1.0 / (r * diel));
    }
};

__device__ __forceinline__ fl volume_gpu(sz a_ad, sz a_xs) {
    if (a_ad < AD_TYPE_SIZE)
        return ad_type_property_gpu(a_ad).volume;
    else if (a_xs < XS_TYPE_SIZE)
        return 4.0 * pi / 3.0 * pow(xs_radius_gpu(a_xs), 3);
    VINA_CHECK_GPU(false);
    return 0.0;  // placating the compiler
};

__device__ __forceinline__ fl solvation_parameter_gpu(sz a_ad, sz a_xs) {
    if (a_ad < AD_TYPE_SIZE)
        return ad_type_property_gpu(a_ad).solvation;
    else if (a_xs == XS_TYPE_Met_D)
        return metal_solvation_parameter_gpu;
    VINA_CHECK_GPU(false);
    return 0.0;  // placating the compiler
};

__device__ __forceinline__ bool not_max_gpu(fl x) {
    return (x < 0.1 * INFINITY); /* Problem: replace max_fl with INFINITY? */
}

__device__ __forceinline__ fl ad4_solvation_eval_gpu(fl a_ad, fl a_xs, fl a_charge, fl b_ad,
                                                     fl b_xs, fl b_charge, fl desolvation_sigma,
                                                     fl solvation_q, bool charge_dependent,
                                                     fl cutoff, fl r) {
    if (r >= cutoff) return 0.0;
    fl q1 = a_charge;
    fl q2 = b_charge;
    VINA_CHECK_GPU(not_max_gpu(q1));
    VINA_CHECK_GPU(not_max_gpu(q2));
    fl solv1 = solvation_parameter_gpu(a_ad, a_xs);
    fl solv2 = solvation_parameter_gpu(b_ad, b_xs);
    fl volume1 = volume_gpu(a_ad, a_xs);
    fl volume2 = volume_gpu(b_ad, b_xs);
    fl my_solv = charge_dependent ? solvation_q : 0;
    fl tmp
        = ((solv1 + my_solv * std::abs(q1)) * volume2 + (solv2 + my_solv * std::abs(q2)) * volume1)
          * std::exp(-0.5 * sqr(r / desolvation_sigma));
    VINA_CHECK_GPU(not_max_gpu(tmp));
    return tmp;
};

__device__ __forceinline__ fl ad4_vdw_eval(sz a_ad, sz b_ad, fl r, fl smoothing, fl cap,
                                           fl cutoff) {
    if (r >= cutoff) return 0.0;
    sz t1 = a_ad;
    sz t2 = b_ad;
    fl hb_depth = ad4_hb_eps_gpu(t1) * ad4_hb_eps_gpu(t2);
    fl vdw_rij = ad4_vdw_radius_gpu(t1) + ad4_vdw_radius_gpu(t2);
    fl vdw_depth = std::sqrt(ad4_vdw_eps_gpu(t1) * ad4_vdw_eps_gpu(t2));
    if (hb_depth < 0) return 0.0;  // interaction is hb, not vdw.
    r = smoothen_gpu(r, vdw_rij, smoothing);
    fl c_12 = pow(vdw_rij, 12) * vdw_depth;
    fl c_6 = pow(vdw_rij, 6) * vdw_depth * 2.0;
    fl r6 = pow(r, 6);
    fl r12 = pow(r, 12);
    if (r12 > epsilon_fl && r6 > epsilon_fl)
        return min(cap, c_12 / r12 - c_6 / r6);
    else
        return cap;
    VINA_CHECK_GPU(false);
    return 0.0;  // placating the compiler
};

__device__ __forceinline__ fl ad4_hb_eval(sz a_ad, sz b_ad, fl r, fl smoothing, fl cap, fl cutoff) {
    if (r >= cutoff) return 0.0;
    sz t1 = a_ad;
    sz t2 = b_ad;
    fl hb_rij = ad4_hb_radius_gpu(t1) + ad4_hb_radius_gpu(t2);
    fl hb_depth = ad4_hb_eps_gpu(t1) * ad4_hb_eps_gpu(t2);
    fl vdw_rij = ad4_vdw_radius_gpu(t1) + ad4_vdw_radius_gpu(t2);
    if (hb_depth >= 0) return 0.0;  // interaction is vdw, not hb.
    r = smoothen_gpu(r, hb_rij, smoothing);
    fl c_12 = pow(hb_rij, 12) * -hb_depth * 10 / 2.0;
    fl c_10 = pow(hb_rij, 10) * -hb_depth * 12 / 2.0;
    fl r10 = pow(r, 10);
    fl r12 = pow(r, 12);
    if (r12 > epsilon_fl && r10 > epsilon_fl)
        return min(cap, c_12 / r12 - c_10 / r10);
    else
        return cap;
    VINA_CHECK_GPU(false);
    return 0.0;  // placating the compiler
};

// Macrocycle - Vina and AD42
// Cutoff is large (20.0), may be biased if max_cutoff equals 8.0
__device__ __forceinline__ fl linearattraction_eval(sz t1, sz t2, fl r, fl cutoff) {
    if (r >= cutoff) return 0.0;
    if (is_glued_gpu(t1, t2))
        return r;
    else
        return 0.0;
};

/* potential related end */

/* scoring function related start */

typedef struct {
    int m_num_potentials;
    fl m_weights[6];
    int m_sf_choice;  // 0:vina, 1:vinardo, 2:ad4
    // constants used in potential terms
    fl vina_gaussian_offset_1, vina_gaussian_width_1, vina_gaussian_cutoff_1;
    fl vina_gaussian_offset_2, vina_gaussian_width_2, vina_gaussian_cutoff_2;
    fl vina_repulsion_offset, vina_repulsion_cutoff;
    fl vina_hydrophobic_good, vina_hydrophobic_bad, vina_hydrophobic_cutoff;
    fl vina_non_dir_h_bond_good, vina_non_dir_h_bond_bad, vina_non_dir_h_bond_cutoff;
    fl vinardo_gaussian_offset, vinardo_gaussian_width, vinardo_gaussian_cutoff;
    fl vinardo_repulsion_offset, vinardo_repulsion_cutoff;
    fl vinardo_hydrophobic_good, vinardo_hydrophobic_bad, vinardo_hydrophobic_cutoff;
    fl vinardo_non_dir_h_bond_good, vinardo_non_dir_h_bond_bad, vinardo_non_dir_h_bond_cutoff;
    fl ad4_electrostatic_cap, ad4_electrostatic_cutoff;
    fl ad4_solvation_desolvation_sigma, ad4_solvation_solvation_q, ad4_solvation_cutoff;
    bool ad4_solvation_charge_dependent;
    fl ad4_vdw_smoothing, ad4_vdw_cap, ad4_vdw_cutoff;
    fl ad4_hb_smoothing, ad4_hb_cap, ad4_hb_cutoff;
    fl linearattraction_cutoff;  // shared by all scoring functions

} scoring_function_cuda_t;

/* scoring function related end */
