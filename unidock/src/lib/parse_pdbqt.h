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

#ifndef VINA_PARSE_PDBQT_H
#define VINA_PARSE_PDBQT_H

#include <string>
#include "model.h"
#include <set>
struct rigid {
    atomv atoms;
};
model parse_receptor_pdbqt(const std::string &rigid = std::string(),
                           const std::string &flex = std::string(),
                           atom_type::t atype = atom_type::XS);  // can throw struct_parse_error
model parse_receptor_pdb(const std::string &rigid = std::string(),
                         const std::string &flex = std::string(),
                         atom_type::t atype = atom_type::XS);  // can throw struct_parse_error

model parse_ligand_pdbqt_from_file(const std::string &name, atom_type::t atype,
                                   bool keep_H = false);  // can throw struct_parse_error
model parse_ligand_pdbqt_from_file_no_failure(const std::string &name, atom_type::t atype,
                                              bool keep_H = false);  // can throw struct_parse_error
model parse_ligand_from_file_no_failure(const std::string &name, atom_type::t atype,
                                        bool keep_H = false);  // can throw struct_parse_error
model parse_ligand_sdf_from_file_no_failure(const std::string &name, atom_type::t atype,
                                            bool keep_H = false);  // can throw struct_parse_error

model parse_ligand_pdbqt_from_string(const std::string &string_name,
                                     atom_type::t atype);  // can exit with code EXIT_FAILURE
model parse_ligand_pdbqt_from_string_no_failure(
    const std::string &string_name, atom_type::t atype);  // can return empty model as failure
void parse_pdbqt_rigid(const path& name, rigid& r);

#endif
