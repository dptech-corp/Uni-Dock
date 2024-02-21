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

#ifndef VINA_IGRID_H
#define VINA_IGRID_H

#include "common.h"
#include "grid.h"

struct model;  // forward declaration

struct igrid {  // grids interface (that cache, etc. conform to)
    virtual fl eval(const model& m, fl v) const = 0;  // needs m.coords // clean up
    virtual fl eval_intra(model& m, fl v) const = 0;  // only flexres-grids
    virtual fl eval_deriv(model& m,
                          fl v) const
        = 0;  // needs m.coords, sets m.minus_forces // clean up
    virtual int get_atu() const = 0;
    virtual float get_slope() const = 0;
    virtual grid_dims get_gd() const = 0;
    virtual std::vector<grid> get_grids() const = 0;
};

#endif
