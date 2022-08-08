#ifndef VINA_BIAS_H
#define VINA_BIAS_H

#include "common.h"
#include <vector>

struct bias_element
{
public:
    fl coords[3];
    fl vset, r;
    enum type{don, acc, aro, map}; // depend on interaction type
    szv atom_list; // effected atom types

};


#endif