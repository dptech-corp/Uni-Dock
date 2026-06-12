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

#ifndef VINA_RANDOM_H
#define VINA_RANDOM_H

#include <random>
#ifdef __CUDACC__
// nvcc cannot compile Boost.Math's GPU-enabled headers (Boost >= 1.87), which
// the <boost/random.hpp> umbrella pulls in via its distributions. The CUDA
// translation units only need the mt19937 engine type, so include just that.
// Host translation units keep the full umbrella below (unchanged behavior).
#include <boost/random/mersenne_twister.hpp>
#else
#include <boost/random.hpp>
#endif
#include "common.h"

typedef boost::mt19937 rng;

fl random_fl(fl a, fl b, rng& generator);             // expects a < b, returns rand in [a, b]
fl random_normal(fl mean, fl sigma, rng& generator);  // expects sigma >= 0
int random_int(int a, int b, rng& generator);         // expects a <= b, returns rand in [a, b]
sz random_sz(sz a, sz b, rng& generator);             // expects a <= b, returns rand in [a, b]
vec random_inside_sphere(
    rng& generator);  // returns a random vec inside the sphere centered at 0 with radius 1
vec random_in_box(const vec& corner1, const vec& corner2,
                  rng& generator);  // expects corner1[i] < corner2[i]
int auto_seed();                    // make seed from PID and time

#endif
