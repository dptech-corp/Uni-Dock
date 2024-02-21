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

#ifndef VINA_ATOM_BASE_H
#define VINA_ATOM_BASE_H

#include "atom_type.h"

struct atom_base : public atom_type {
    fl charge;
    atom_base() : charge(0) {}

private:
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive& ar, const unsigned version) {
        ar& boost::serialization::base_object<atom_type>(*this);
        ar & charge;
    }
};

#endif
