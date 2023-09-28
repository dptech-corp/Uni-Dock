/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>,
           The Olson Lab,
           The Scripps Research Institute

*/

#include "non_cache.h"
#include "curl.h"

non_cache::non_cache(const model& m, const grid_dims& gd_, const precalculate* p_, fl slope_,
                     std::vector<bias_element> bias_list_)
    : sgrid(m, szv_grid_dims(gd_), p_->cutoff_sqr()),
      gd(gd_),
      p(p_),
      slope(slope_),
      bias_list(bias_list_) {}

fl non_cache::eval(const model& m, fl v) const {  // clean up
    fl e = 0;
    const fl cutoff_sqr = p->cutoff_sqr();

    sz n = num_atom_types(atom_type::XS);

    VINA_FOR(i, m.num_movable_atoms()) {
        fl this_e = 0;
        fl out_of_bounds_penalty = 0;
        const atom& a = m.atoms[i];
        sz t1 = a.get(atom_type::XS);
        if (t1 >= n) continue;
        switch (t1) {
            case XS_TYPE_G0:
            case XS_TYPE_G1:
            case XS_TYPE_G2:
            case XS_TYPE_G3:
                continue;
            case XS_TYPE_C_H_CG0:
            case XS_TYPE_C_H_CG1:
            case XS_TYPE_C_H_CG2:
            case XS_TYPE_C_H_CG3:
                t1 = XS_TYPE_C_H;
                break;
            case XS_TYPE_C_P_CG0:
            case XS_TYPE_C_P_CG1:
            case XS_TYPE_C_P_CG2:
            case XS_TYPE_C_P_CG3:
                t1 = XS_TYPE_C_P;
                break;
        }

        const vec& a_coords = m.coords[i];
        vec adjusted_a_coords;
        adjusted_a_coords = a_coords;
        VINA_FOR_IN(j, gd) {
            if (gd[j].n_voxels > 0) {
                if (a_coords[j] < gd[j].begin) {
                    adjusted_a_coords[j] = gd[j].begin;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].begin);
                } else if (a_coords[j] > gd[j].end) {
                    adjusted_a_coords[j] = gd[j].end;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].end);
                }
            }
        }
        out_of_bounds_penalty *= slope;

        const szv& possibilities = sgrid.possibilities(adjusted_a_coords);

        VINA_FOR_IN(possibilities_j, possibilities) {
            const sz j = possibilities[possibilities_j];
            const atom& b = m.grid_atoms[j];
            sz t2 = b.get(atom_type::XS);
            if (t2 >= n) continue;
            vec r_ba;
            r_ba = adjusted_a_coords - b.coords;  // FIXME why b-a and not a-b ?
            fl r2 = sqr(r_ba);
            if (r2 < cutoff_sqr) {
                sz type_pair_index = get_type_pair_index(atom_type::XS, a, b);
                this_e += p->eval_fast(type_pair_index, r2);
            }
        }
        // add bias, used in refining
        if (m.bias_list.size() > 0) {
            // using bias of ligand
            for (auto bias = m.bias_list.begin(); bias != m.bias_list.end(); ++bias) {
                const fl rb2 = vec_distance_sqr(bias->coords, a_coords);
                if (rb2 > cutoff_sqr) continue;
                fl dE = bias->vset * exp(-rb2 / bias->r / bias->r);
                if (dE >= -0.01) continue;
                switch (bias->type) {
                    case bias_element::itype::don: {  // HD
                        break;
                    }
                    case bias_element::itype::acc: {  // OA, NA
                        if (t1 == XS_TYPE_O_A || t1 == XS_TYPE_N_A || t1 == XS_TYPE_O_DA
                            || t1 == XS_TYPE_N_DA)
                            this_e += dE;
                        break;
                    }
                    case bias_element::itype::aro: {  // AC
                        // simply add bias for add carbon in the case of atom type AC
                        if (t1 == XS_TYPE_C_P_CG0 || t1 == XS_TYPE_C_H_CG0 || t1 == XS_TYPE_C_P_CG1
                            || t1 == XS_TYPE_C_H_CG1 || t1 == XS_TYPE_C_P_CG2
                            || t1 == XS_TYPE_C_H_CG2 || t1 == XS_TYPE_C_P_CG3
                            || t1 == XS_TYPE_C_H_CG3) {
                            this_e += dE;
                        }
                        break;
                    }
                    case bias_element::itype::map: {
                        if (bias->atom_list.size() == 0) {  // all
                            this_e += dE;
                        } else {
                            for (int t = 0; t < bias->atom_list.size(); ++t) {
                                if (bias->atom_list[t] == AD_TYPE_SIZE + 1 && t1 == XS_TYPE_Met_D)
                                    this_e += dE;
                                else {
                                    sz ad = bias->atom_list[t];
                                    sz el = ad_type_to_el_type(ad);
                                    switch (el) {
                                        case EL_TYPE_H:
                                            break;
                                        case EL_TYPE_C: {
                                            if (ad == AD_TYPE_CG0
                                                && (t1 == XS_TYPE_C_P_CG0
                                                    || t1 == XS_TYPE_C_H_CG0)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG1
                                                       && (t1 == XS_TYPE_C_P_CG1
                                                           || t1 == XS_TYPE_C_H_CG1)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG2
                                                       && (t1 == XS_TYPE_C_P_CG2
                                                           || t1 == XS_TYPE_C_H_CG2)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG3
                                                       && (t1 == XS_TYPE_C_P_CG3
                                                           || t1 == XS_TYPE_C_H_CG3)) {
                                                this_e += dE;
                                            } else if (t1 == XS_TYPE_C_P || t1 == XS_TYPE_C_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        }
                                        case EL_TYPE_N:
                                            if (t1 == XS_TYPE_N_DA || t1 == XS_TYPE_N_A
                                                || t1 == XS_TYPE_N_D || t1 == XS_TYPE_N_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_O:
                                            if (t1 == XS_TYPE_O_DA || t1 == XS_TYPE_O_A
                                                || t1 == XS_TYPE_O_D || t1 == XS_TYPE_O_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_S:
                                            if (t1 == XS_TYPE_S_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_P:
                                            if (t1 == XS_TYPE_P_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_F:
                                            if (t1 == XS_TYPE_F_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Cl:
                                            if (t1 == XS_TYPE_Cl_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Br:
                                            if (t1 == XS_TYPE_Br_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_I:
                                            if (t1 == XS_TYPE_I_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Si:
                                            if (t1 == XS_TYPE_Si) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_At:
                                            if (t1 == XS_TYPE_At) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Met:
                                            if (t1 == XS_TYPE_Met_D) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Dummy: {
                                            if (ad == AD_TYPE_G0 && t1 == XS_TYPE_G0) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G1 && t1 == XS_TYPE_G1) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G2 && t1 == XS_TYPE_G2) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G3 && t1 == XS_TYPE_G3) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_W && t1 == XS_TYPE_SIZE) {
                                                this_e += dE;
                                            }  // no W atoms in XS types
                                            else
                                                VINA_CHECK(false);
                                            break;
                                        }
                                        case EL_TYPE_SIZE:
                                            break;
                                        default:
                                            VINA_CHECK(false);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            curl(this_e, v);

            e += this_e + out_of_bounds_penalty;
        } else {
            for (auto bias = bias_list.begin(); bias != bias_list.end(); ++bias) {
                const fl rb2 = vec_distance_sqr(bias->coords, a_coords);
                if (rb2 > cutoff_sqr) continue;
                fl dE = bias->vset * exp(-rb2 / bias->r / bias->r);
                if (dE >= -0.01) continue;
                switch (bias->type) {
                    case bias_element::itype::don: {  // HD
                        break;
                    }
                    case bias_element::itype::acc: {  // OA, NA
                        if (t1 == XS_TYPE_O_A || t1 == XS_TYPE_N_A || t1 == XS_TYPE_O_DA
                            || t1 == XS_TYPE_N_DA)
                            this_e += dE;
                        break;
                    }
                    case bias_element::itype::aro: {  // AC
                        // simply add bias for add carbon in the case of atom type AC
                        if (t1 == XS_TYPE_C_P_CG0 || t1 == XS_TYPE_C_H_CG0 || t1 == XS_TYPE_C_P_CG1
                            || t1 == XS_TYPE_C_H_CG1 || t1 == XS_TYPE_C_P_CG2
                            || t1 == XS_TYPE_C_H_CG2 || t1 == XS_TYPE_C_P_CG3
                            || t1 == XS_TYPE_C_H_CG3) {
                            this_e += dE;
                        }
                        break;
                    }
                    case bias_element::itype::map: {
                        if (bias->atom_list.size() == 0) {  // all
                            this_e += dE;
                        } else {
                            for (int t = 0; t < bias->atom_list.size(); ++t) {
                                if (bias->atom_list[t] == AD_TYPE_SIZE + 1 && t1 == XS_TYPE_Met_D)
                                    this_e += dE;
                                else {
                                    sz ad = bias->atom_list[t];
                                    sz el = ad_type_to_el_type(ad);
                                    switch (el) {
                                        case EL_TYPE_H:
                                            break;
                                        case EL_TYPE_C: {
                                            if (ad == AD_TYPE_CG0
                                                && (t1 == XS_TYPE_C_P_CG0
                                                    || t1 == XS_TYPE_C_H_CG0)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG1
                                                       && (t1 == XS_TYPE_C_P_CG1
                                                           || t1 == XS_TYPE_C_H_CG1)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG2
                                                       && (t1 == XS_TYPE_C_P_CG2
                                                           || t1 == XS_TYPE_C_H_CG2)) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_CG3
                                                       && (t1 == XS_TYPE_C_P_CG3
                                                           || t1 == XS_TYPE_C_H_CG3)) {
                                                this_e += dE;
                                            } else if (t1 == XS_TYPE_C_P || t1 == XS_TYPE_C_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        }
                                        case EL_TYPE_N:
                                            if (t1 == XS_TYPE_N_DA || t1 == XS_TYPE_N_A
                                                || t1 == XS_TYPE_N_D || t1 == XS_TYPE_N_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_O:
                                            if (t1 == XS_TYPE_O_DA || t1 == XS_TYPE_O_A
                                                || t1 == XS_TYPE_O_D || t1 == XS_TYPE_O_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_S:
                                            if (t1 == XS_TYPE_S_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_P:
                                            if (t1 == XS_TYPE_P_P) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_F:
                                            if (t1 == XS_TYPE_F_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Cl:
                                            if (t1 == XS_TYPE_Cl_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Br:
                                            if (t1 == XS_TYPE_Br_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_I:
                                            if (t1 == XS_TYPE_I_H) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Si:
                                            if (t1 == XS_TYPE_Si) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_At:
                                            if (t1 == XS_TYPE_At) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Met:
                                            if (t1 == XS_TYPE_Met_D) {
                                                this_e += dE;
                                            }
                                            break;
                                        case EL_TYPE_Dummy: {
                                            if (ad == AD_TYPE_G0 && t1 == XS_TYPE_G0) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G1 && t1 == XS_TYPE_G1) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G2 && t1 == XS_TYPE_G2) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_G3 && t1 == XS_TYPE_G3) {
                                                this_e += dE;
                                            } else if (ad == AD_TYPE_W && t1 == XS_TYPE_SIZE) {
                                                this_e += dE;
                                            }  // no W atoms in XS types
                                            else
                                                VINA_CHECK(false);
                                            break;
                                        }
                                        case EL_TYPE_SIZE:
                                            break;
                                        default:
                                            VINA_CHECK(false);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            curl(this_e, v);

            e += this_e + out_of_bounds_penalty;
        }
    }
    return e;
}

bool non_cache::within(const model& m, fl margin) const {
    VINA_FOR(i, m.num_movable_atoms()) {
        if (m.atoms[i].is_hydrogen()) continue;
        const vec& a_coords = m.coords[i];
        VINA_FOR_IN(j, gd)
        if (gd[j].n_voxels > 0)
            if (a_coords[j] < gd[j].begin - margin || a_coords[j] > gd[j].end + margin)
                return false;
    }
    return true;
}

fl non_cache::eval_deriv(model& m, fl v) const {  // clean up
    fl e = 0;
    const fl cutoff_sqr = p->cutoff_sqr();

    sz n = num_atom_types(atom_type::XS);

    VINA_FOR(i, m.num_movable_atoms()) {
        fl this_e = 0;
        vec deriv(0, 0, 0);
        vec out_of_bounds_deriv(0, 0, 0);
        fl out_of_bounds_penalty = 0;
        const atom& a = m.atoms[i];
        sz t1 = a.get(atom_type::XS);
        if (t1 >= n) {
            m.minus_forces[i].assign(0);
            continue;
        }
        switch (t1) {
            case XS_TYPE_G0:
            case XS_TYPE_G1:
            case XS_TYPE_G2:
            case XS_TYPE_G3:
                continue;
            case XS_TYPE_C_H_CG0:
            case XS_TYPE_C_H_CG1:
            case XS_TYPE_C_H_CG2:
            case XS_TYPE_C_H_CG3:
                t1 = XS_TYPE_C_H;
                break;
            case XS_TYPE_C_P_CG0:
            case XS_TYPE_C_P_CG1:
            case XS_TYPE_C_P_CG2:
            case XS_TYPE_C_P_CG3:
                t1 = XS_TYPE_C_P;
                break;
        }
        const vec& a_coords = m.coords[i];
        vec adjusted_a_coords;
        adjusted_a_coords = a_coords;
        VINA_FOR_IN(j, gd) {
            if (gd[j].n_voxels > 0) {
                if (a_coords[j] < gd[j].begin) {
                    adjusted_a_coords[j] = gd[j].begin;
                    out_of_bounds_deriv[j] = -1;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].begin);
                } else if (a_coords[j] > gd[j].end) {
                    adjusted_a_coords[j] = gd[j].end;
                    out_of_bounds_deriv[j] = 1;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].end);
                }
            }
        }
        out_of_bounds_penalty *= slope;
        out_of_bounds_deriv *= slope;

        const szv& possibilities = sgrid.possibilities(adjusted_a_coords);

        VINA_FOR_IN(possibilities_j, possibilities) {
            const sz j = possibilities[possibilities_j];
            const atom& b = m.grid_atoms[j];
            sz t2 = b.get(atom_type::XS);
            if (t2 >= n) continue;
            vec r_ba;
            r_ba = adjusted_a_coords - b.coords;  // FIXME why b-a and not a-b ?
            fl r2 = sqr(r_ba);
            if (r2 < cutoff_sqr) {
                sz type_pair_index = get_type_pair_index(atom_type::XS, a, b);
                pr e_dor = p->eval_deriv(type_pair_index, r2);
                this_e += e_dor.first;
                deriv += e_dor.second * r_ba;
            }
        }

        // add bias and bias derivs
        if (m.bias_list.size() > 0) {
            // using bias of ligand
            for (auto bias = m.bias_list.begin(); bias != m.bias_list.end(); ++bias) {
                const fl rb2 = vec_distance_sqr(bias->coords, a_coords);
                if (rb2 > cutoff_sqr) continue;
                fl dE = bias->vset * exp(-rb2 / bias->r / bias->r);
                // calculate deriv of bias, we can get higher accuracy without using precalculate
                vec bias_deriv = dE / (-bias->r * bias->r) * (a_coords - bias->coords);

                if (dE >= -0.01) continue;
                switch (bias->type) {
                    case bias_element::itype::don: {  // HD
                        break;
                    }
                    case bias_element::itype::acc: {  // OA, NA
                        if (t1 == XS_TYPE_O_A || t1 == XS_TYPE_N_A || t1 == XS_TYPE_O_DA
                            || t1 == XS_TYPE_N_DA) {
                            this_e += dE;
                            deriv += bias_deriv;
                        }
                        break;
                    }
                    case bias_element::itype::aro: {  // AC
                        // simply add bias for add carbon in the case of atom type AC
                        if (t1 == XS_TYPE_C_P_CG0 || t1 == XS_TYPE_C_H_CG0 || t1 == XS_TYPE_C_P_CG1
                            || t1 == XS_TYPE_C_H_CG1 || t1 == XS_TYPE_C_P_CG2
                            || t1 == XS_TYPE_C_H_CG2 || t1 == XS_TYPE_C_P_CG3
                            || t1 == XS_TYPE_C_H_CG3) {
                            this_e += dE;
                            deriv += bias_deriv;
                        }
                        break;
                    }
                    case bias_element::itype::map: {
                        if (bias->atom_list.size() == 0) {  // all
                            this_e += dE;
                            deriv += bias_deriv;
                        } else {
                            for (int t = 0; t < bias->atom_list.size(); ++t) {
                                if (bias->atom_list[t] == AD_TYPE_SIZE + 1 && t1 == XS_TYPE_Met_D)
                                    this_e += dE;
                                else {
                                    sz ad = bias->atom_list[t];
                                    sz el = ad_type_to_el_type(ad);
                                    switch (el) {
                                        case EL_TYPE_H:
                                            break;
                                        case EL_TYPE_C: {
                                            if (ad == AD_TYPE_CG0
                                                && (t1 == XS_TYPE_C_P_CG0
                                                    || t1 == XS_TYPE_C_H_CG0)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG1
                                                       && (t1 == XS_TYPE_C_P_CG1
                                                           || t1 == XS_TYPE_C_H_CG1)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG2
                                                       && (t1 == XS_TYPE_C_P_CG2
                                                           || t1 == XS_TYPE_C_H_CG2)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG3
                                                       && (t1 == XS_TYPE_C_P_CG3
                                                           || t1 == XS_TYPE_C_H_CG3)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (t1 == XS_TYPE_C_P || t1 == XS_TYPE_C_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        }
                                        case EL_TYPE_N:
                                            if (t1 == XS_TYPE_N_DA || t1 == XS_TYPE_N_A
                                                || t1 == XS_TYPE_N_D || t1 == XS_TYPE_N_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_O:
                                            if (t1 == XS_TYPE_O_DA || t1 == XS_TYPE_O_A
                                                || t1 == XS_TYPE_O_D || t1 == XS_TYPE_O_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_S:
                                            if (t1 == XS_TYPE_S_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_P:
                                            if (t1 == XS_TYPE_P_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_F:
                                            if (t1 == XS_TYPE_F_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Cl:
                                            if (t1 == XS_TYPE_Cl_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Br:
                                            if (t1 == XS_TYPE_Br_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_I:
                                            if (t1 == XS_TYPE_I_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Si:
                                            if (t1 == XS_TYPE_Si) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_At:
                                            if (t1 == XS_TYPE_At) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Met:
                                            if (t1 == XS_TYPE_Met_D) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Dummy: {
                                            if (ad == AD_TYPE_G0 && t1 == XS_TYPE_G0) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G1 && t1 == XS_TYPE_G1) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G2 && t1 == XS_TYPE_G2) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G3 && t1 == XS_TYPE_G3) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_W && t1 == XS_TYPE_SIZE) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }  // no W atoms in XS types
                                            else
                                                VINA_CHECK(false);
                                            break;
                                        }
                                        case EL_TYPE_SIZE:
                                            break;
                                        default:
                                            VINA_CHECK(false);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            curl(this_e, deriv, v);
            m.minus_forces[i] = deriv + out_of_bounds_deriv;

            e += this_e + out_of_bounds_penalty;
        } else {
            for (auto bias = bias_list.begin(); bias != bias_list.end(); ++bias) {
                const fl rb2 = vec_distance_sqr(bias->coords, a_coords);
                if (rb2 > cutoff_sqr) continue;
                fl dE = bias->vset * exp(-rb2 / bias->r / bias->r);
                // calculate deriv of bias, we can get higher accuracy without using precalculate
                vec bias_deriv = dE / (-bias->r * bias->r) * (a_coords - bias->coords);

                if (dE >= -0.01) continue;
                switch (bias->type) {
                    case bias_element::itype::don: {  // HD
                        break;
                    }
                    case bias_element::itype::acc: {  // OA, NA
                        if (t1 == XS_TYPE_O_A || t1 == XS_TYPE_N_A || t1 == XS_TYPE_O_DA
                            || t1 == XS_TYPE_N_DA) {
                            this_e += dE;
                            deriv += bias_deriv;
                        }
                        break;
                    }
                    case bias_element::itype::aro: {  // AC
                        // simply add bias for add carbon in the case of atom type AC
                        if (t1 == XS_TYPE_C_P_CG0 || t1 == XS_TYPE_C_H_CG0 || t1 == XS_TYPE_C_P_CG1
                            || t1 == XS_TYPE_C_H_CG1 || t1 == XS_TYPE_C_P_CG2
                            || t1 == XS_TYPE_C_H_CG2 || t1 == XS_TYPE_C_P_CG3
                            || t1 == XS_TYPE_C_H_CG3) {
                            this_e += dE;
                            deriv += bias_deriv;
                        }
                        break;
                    }
                    case bias_element::itype::map: {
                        if (bias->atom_list.size() == 0) {  // all
                            this_e += dE;
                            deriv += bias_deriv;
                        } else {
                            for (int t = 0; t < bias->atom_list.size(); ++t) {
                                if (bias->atom_list[t] == AD_TYPE_SIZE + 1 && t1 == XS_TYPE_Met_D)
                                    this_e += dE;
                                else {
                                    sz ad = bias->atom_list[t];
                                    sz el = ad_type_to_el_type(ad);
                                    switch (el) {
                                        case EL_TYPE_H:
                                            break;
                                        case EL_TYPE_C: {
                                            if (ad == AD_TYPE_CG0
                                                && (t1 == XS_TYPE_C_P_CG0
                                                    || t1 == XS_TYPE_C_H_CG0)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG1
                                                       && (t1 == XS_TYPE_C_P_CG1
                                                           || t1 == XS_TYPE_C_H_CG1)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG2
                                                       && (t1 == XS_TYPE_C_P_CG2
                                                           || t1 == XS_TYPE_C_H_CG2)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_CG3
                                                       && (t1 == XS_TYPE_C_P_CG3
                                                           || t1 == XS_TYPE_C_H_CG3)) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (t1 == XS_TYPE_C_P || t1 == XS_TYPE_C_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        }
                                        case EL_TYPE_N:
                                            if (t1 == XS_TYPE_N_DA || t1 == XS_TYPE_N_A
                                                || t1 == XS_TYPE_N_D || t1 == XS_TYPE_N_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_O:
                                            if (t1 == XS_TYPE_O_DA || t1 == XS_TYPE_O_A
                                                || t1 == XS_TYPE_O_D || t1 == XS_TYPE_O_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_S:
                                            if (t1 == XS_TYPE_S_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_P:
                                            if (t1 == XS_TYPE_P_P) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_F:
                                            if (t1 == XS_TYPE_F_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Cl:
                                            if (t1 == XS_TYPE_Cl_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Br:
                                            if (t1 == XS_TYPE_Br_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_I:
                                            if (t1 == XS_TYPE_I_H) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Si:
                                            if (t1 == XS_TYPE_Si) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_At:
                                            if (t1 == XS_TYPE_At) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Met:
                                            if (t1 == XS_TYPE_Met_D) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }
                                            break;
                                        case EL_TYPE_Dummy: {
                                            if (ad == AD_TYPE_G0 && t1 == XS_TYPE_G0) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G1 && t1 == XS_TYPE_G1) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G2 && t1 == XS_TYPE_G2) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_G3 && t1 == XS_TYPE_G3) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            } else if (ad == AD_TYPE_W && t1 == XS_TYPE_SIZE) {
                                                this_e += dE;
                                                deriv += bias_deriv;
                                            }  // no W atoms in XS types
                                            else
                                                VINA_CHECK(false);
                                            break;
                                        }
                                        case EL_TYPE_SIZE:
                                            break;
                                        default:
                                            VINA_CHECK(false);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            curl(this_e, deriv, v);
            m.minus_forces[i] = deriv + out_of_bounds_deriv;

            e += this_e + out_of_bounds_penalty;
        }
    }
    return e;
}

fl non_cache::eval_intra(model& m, fl v) const {  // clean up
    fl e = 0;
    const fl cutoff_sqr = p->cutoff_sqr();

    sz n = num_atom_types(atom_type::XS);

    VINA_FOR(i, m.num_movable_atoms()) {
        if (m.is_atom_in_ligand(i)) continue;  // we only want flex-rigid interactions
        fl this_e = 0;
        fl out_of_bounds_penalty = 0;
        const atom& a = m.atoms[i];
        sz t1 = a.get(atom_type::XS);
        if (t1 >= n) continue;
        switch (t1) {
            case XS_TYPE_G0:
            case XS_TYPE_G1:
            case XS_TYPE_G2:
            case XS_TYPE_G3:
                continue;
            case XS_TYPE_C_H_CG0:
            case XS_TYPE_C_H_CG1:
            case XS_TYPE_C_H_CG2:
            case XS_TYPE_C_H_CG3:
                t1 = XS_TYPE_C_H;
                break;
            case XS_TYPE_C_P_CG0:
            case XS_TYPE_C_P_CG1:
            case XS_TYPE_C_P_CG2:
            case XS_TYPE_C_P_CG3:
                t1 = XS_TYPE_C_P;
                break;
        }

        const vec& a_coords = m.coords[i];
        vec adjusted_a_coords;
        adjusted_a_coords = a_coords;
        VINA_FOR_IN(j, gd) {
            if (gd[j].n_voxels > 0) {
                if (a_coords[j] < gd[j].begin) {
                    adjusted_a_coords[j] = gd[j].begin;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].begin);
                } else if (a_coords[j] > gd[j].end) {
                    adjusted_a_coords[j] = gd[j].end;
                    out_of_bounds_penalty += std::abs(a_coords[j] - gd[j].end);
                }
            }
        }
        out_of_bounds_penalty *= slope;

        const szv& possibilities = sgrid.possibilities(adjusted_a_coords);

        VINA_FOR_IN(possibilities_j, possibilities) {
            const sz j = possibilities[possibilities_j];
            const atom& b = m.grid_atoms[j];
            sz t2 = b.get(atom_type::XS);
            if (t2 >= n) continue;
            vec r_ba;
            r_ba = adjusted_a_coords - b.coords;  // FIXME why b-a and not a-b ?
            fl r2 = sqr(r_ba);
            if (r2 < cutoff_sqr) {
                sz type_pair_index = get_type_pair_index(atom_type::XS, a, b);
                this_e += p->eval_fast(type_pair_index, r2);
            }
        }
        curl(this_e, v);
        e += this_e + out_of_bounds_penalty;
    }
    return e;
}

// add to make get_grids() work
std::vector<grid> non_cache::get_grids() const {
    assert(false);  // This function should not be called!
    std::vector<grid> g;
    return g;
};

int non_cache::get_atu() const {
    assert(false);  // This function should not be called!
    return 0;
}

float non_cache::get_slope() const {
    assert(false);  // This function should not be called!
    return 0;
}