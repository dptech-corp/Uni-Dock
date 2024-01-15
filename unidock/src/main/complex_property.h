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
#pragma once

#include <iostream>
#include <string>
#include <vector>  // ligand paths
#include <exception>
#include <boost/program_options.hpp>
#include "vina.h"
#include "utils.h"
#include "scoring_function.h"

#include <thread>
#include <chrono>
#include <iterator>
#include <cstddef>

// Holds properties of each ligand complex

struct complex_property
{
    double center_x = 0;
    double center_y = 0;
    double center_z = 0;
    double box_x = 0;
    double box_y = 0;
    double box_z = 0;    
    std::string protein_name;
    std::string ligand_name;
    complex_property(double x, double y, double z, 
                double box_x, double box_y, double box_z,
                std::string protein_name, std::string ligand_name):
        center_x(x),
        center_y(y),
        center_z(z),
        box_x(box_x),
        box_y(box_y),
        box_z(box_z),
        protein_name(protein_name),
        ligand_name(ligand_name){};
    complex_property(){};
};

// Holds properties of all ligand complexs

struct complex_property_holder
{
    int max_count;
    complex_property* m_properties;
    complex_property_holder(int N):
        max_count(N),
        m_properties(nullptr)
        {
            m_properties = new complex_property[N];
        }
    ~complex_property_holder()
    {
        delete [] m_properties;
        m_properties = nullptr;
    }
    complex_property* get_end()
    {
        return &m_properties[max_count];
    }

    struct complex_property_iterator 
    {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = complex_property;
        using pointer           = complex_property*; 
        using reference         = complex_property&; 

        complex_property_iterator(pointer ptr) : m_ptr(ptr) {}
        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        // Prefix increment
        complex_property_iterator& operator++() { m_ptr++; return *this; }  

        // Postfix increment
        complex_property_iterator operator++(int) { complex_property_iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const complex_property_iterator& a, const complex_property_iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const complex_property_iterator& a, const complex_property_iterator& b) { return a.m_ptr != b.m_ptr; }; 
    private:
        pointer m_ptr;    
    };
    complex_property_iterator begin() { return complex_property_iterator(&m_properties[0]); }
    complex_property_iterator end()   { return complex_property_iterator(get_end()); }
};