#ifndef VINA_BIAS_H
#define VINA_BIAS_H

#include "common.h"
#include "atom_constants.h"
#include <vector>
#include <string>
#include <iostream>

#ifdef DEBUG
#    define DEBUG_PRINTF printf
#else
#    define DEBUG_PRINTF(...)
#endif

struct bias_element {
public:
    vec coords;
    fl vset, r;
    enum itype { don, acc, aro, map, unknown } type;  // depend on interaction type
    szv atom_list;  // affected atom types in AD, used only if type==map
    bias_element(std::istringstream& input) {
        input >> coords[0] >> coords[1] >> coords[2] >> vset >> r;
        std::string stype;
        input >> stype;
        DEBUG_PRINTF("stype=%s\n", stype.c_str());
        if (stype == "don") {
            type = don;
        } else if (stype == "acc") {
            type = acc;
        } else if (stype == "aro") {
            type = aro;
        } else if (stype == "map") {
            type = map;
            std::string atom_type;
            while (input >> atom_type) {  // user can assign affected atom type, default all
                sz type_num;
                // convert atom type to num
                type_num = string_to_ad_type_with_met(atom_type);
                if (type_num < AD_TYPE_SIZE) atom_list.push_back(type_num);
            }
        } else {
            type = unknown;
        }
    }
};

#endif