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

#ifndef VINA_NON_CACHE_H
#define VINA_NON_CACHE_H

#include "bias.h"
#include "igrid.h"
#include "precalculate.h"
#include "szv_grid.h"

struct non_cache : public igrid {
    non_cache() {}
    non_cache(const model& m, const grid_dims& gd_, const precalculate* p_, fl slope_,
              const std::vector<bias_element> bias_list_);
    virtual fl eval(const model& m, fl v) const;  // needs m.coords // clean up
    virtual fl eval_intra(model& m, fl v) const;
    virtual fl eval_deriv(model& m, fl v) const;  // needs m.coords, sets m.minus_forces // clean up
    std::vector<grid> get_grids() const;
    int get_atu() const;
    float get_slope() const;
    bool within(const model& m, fl margin = 0.0001) const;
    fl slope;
    std::vector<bias_element> bias_list;
    grid_dims get_gd() const { return gd; }

private:
    szv_grid sgrid;
    grid_dims gd;
    const precalculate* p;
};

#endif
