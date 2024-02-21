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

#include <string>
#include <fstream>  // for getline ?
#include <sstream>  // in parse_two_unsigneds
#include <cctype>   // isspace
#include <exception>
#include <boost/utility.hpp>  // for noncopyable
#include <boost/optional.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
// #include <openbabel/mol.h>
// #include <openbabel/obconversion.h>
#include "model.h"
#include "atom_constants.h"
#include "file.h"
#include "convert_substring.h"
#include "utils.h"
#include "parse_pdbqt.h"
#include "parse_error.h"
#include "kernel.h"

struct parsed_atom : public atom {
    unsigned number;
    // sdf: number_sdf >= 1
    parsed_atom(sz ad_, fl charge_, const vec& coords_, unsigned number_, int number_sdf_)
        : number(number_) {
        ad = ad_;
        charge = charge_;
        coords = coords_;
        number_sdf = number_sdf_;
    }
    // pdbqt
    parsed_atom(sz ad_, fl charge_, const vec& coords_, unsigned number_) : number(number_) {
        ad = ad_;
        charge = charge_;
        coords = coords_;
        number_sdf = 0;
    }
};

void print_zero() { std::cout << "zero" << std::endl; }

void add_context(context& c, std::string& str) {
    c.push_back(parsed_line(str, boost::optional<sz>()));
}

std::string omit_whitespace(const std::string& str, sz i, sz j) {
    if (i < 1) i = 1;
    if (j < i - 1) j = i - 1;  // i >= 1
    if (j < str.size()) j = str.size();

    // omit leading whitespace
    while (i <= j && std::isspace(str[i - 1])) ++i;

    // omit trailing whitespace
    while (i <= j && std::isspace(str[j - 1])) --j;

    VINA_CHECK(i - 1 < str.size());
    VINA_CHECK(j - i + 1 < str.size());

    return str.substr(i - 1, j - i + 1);
}

template <typename T>
T checked_convert_substring(const std::string& str, sz i, sz j, const std::string& dest_nature) {
    VINA_CHECK(i >= 1);
    VINA_CHECK(i <= j + 1);

    if (j > str.size()) throw struct_parse_error("This line is too short.", str);

    // omit leading whitespace
    while (i <= j && std::isspace(str[i - 1])) ++i;

    // omit ending whitespace
    while (i <= j && std::isspace(str[j - 1])) --j;

    const std::string substr = str.substr(i - 1, j - i + 1);
    try {
        return boost::lexical_cast<T>(substr);
    } catch (...) {
        throw struct_parse_error(dest_nature + std::string(" \"") + substr + "\" is not valid.",
                                 str);
    }
}

parsed_atom parse_pdbqt_atom_string(const std::string& str) {
    unsigned number = checked_convert_substring<unsigned>(str, 7, 11, "Atom number");
    vec coords(checked_convert_substring<fl>(str, 31, 38, "Coordinate"),
               checked_convert_substring<fl>(str, 39, 46, "Coordinate"),
               checked_convert_substring<fl>(str, 47, 54, "Coordinate"));
    fl charge = 0;
    if (!substring_is_blank(str, 69, 76))
        charge = checked_convert_substring<fl>(str, 69, 76, "Charge");
    std::string name = omit_whitespace(str, 78, 79);
    sz ad = string_to_ad_type(name);
    parsed_atom tmp(ad, charge, coords, number);

    if (is_non_ad_metal_name(name)) tmp.xs = XS_TYPE_Met_D;
    if (tmp.acceptable_type())
        return tmp;
    else
        throw struct_parse_error(
            "Atom type " + name + " is not a valid AutoDock type (atom types are case-sensitive).",
            str);
}

// sdf line parsing
parsed_atom parse_sdf_atom_string(const std::string& str, int number) {
    // unsigned number = checked_convert_substring<unsigned>(str, 0, 10, "Atom number");
    vec coords(checked_convert_substring<fl>(str, 1, 10, "Coordinate"),
               checked_convert_substring<fl>(str, 11, 20, "Coordinate"),
               checked_convert_substring<fl>(str, 21, 30, "Coordinate"));
    std::string name = str.substr(31, 2);
    if (name[1] == ' ') {
        name = name.substr(0, 1);
    }
    sz ad = string_to_ad_type(name);
    // std::cout << "parse_sdf_atom_string, name=" << name << ", ad=" << ad << std::endl;
    fl charge = 0;

    parsed_atom tmp(ad, charge, coords, 0, number);

    return tmp;
}

struct atom_reference {
    sz index;
    bool inflex;
    atom_reference(sz index_, bool inflex_) : index(index_), inflex(inflex_) {}
};

struct movable_atom : public atom {
    vec relative_coords;
    movable_atom(const atom& a, const vec& relative_coords_) : atom(a) {
        relative_coords = relative_coords_;
    }
};

struct rigid {
    atomv atoms;
};

typedef std::vector<movable_atom> mav;

struct non_rigid_parsed {
    vector_mutable<ligand> ligands;
    vector_mutable<residue> flex;

    mav atoms;
    atomv inflex;

    distance_type_matrix atoms_atoms_bonds;
    matrix<distance_type> atoms_inflex_bonds;
    distance_type_matrix inflex_inflex_bonds;

    distance_type_matrix mobility_matrix() const {
        distance_type_matrix tmp(atoms_atoms_bonds);
        tmp.append(atoms_inflex_bonds, inflex_inflex_bonds);
        return tmp;
    }
};

class parsing_struct {
public:
    // start reading after this class
    template <typename T>  // T == parsing_struct
    struct node_t {
        sz context_index;
        parsed_atom a;
        std::vector<T> ps;
        node_t(const parsed_atom& a_, sz context_index_) : context_index(context_index_), a(a_) {}
        // node_t(const parsing_struct::node_t<T> &n): context_index(n.context_index), a(n.a),
        // ps(n.ps) {}

        // inflex atom insertion
        void insert_inflex(non_rigid_parsed& nr) {
            VINA_FOR_IN(i, ps)
            ps[i].axis_begin = atom_reference(nr.inflex.size(), true);
            nr.inflex.push_back(a);
        }
        void insert_immobiles_inflex(non_rigid_parsed& nr) {
            VINA_FOR_IN(i, ps)
            ps[i].insert_immobile_inflex(nr);
        }

        // insertion into non_rigid_parsed
        void insert(non_rigid_parsed& nr, context& c, const vec& frame_origin) {
            VINA_FOR_IN(i, ps)
            ps[i].axis_begin = atom_reference(nr.atoms.size(), false);
            vec relative_coords;
            relative_coords = a.coords - frame_origin;
            c[context_index].second = nr.atoms.size();
            nr.atoms.push_back(movable_atom(a, relative_coords));
        }
        void insert_immobiles(non_rigid_parsed& nr, context& c, const vec& frame_origin) {
            VINA_FOR_IN(i, ps)
            ps[i].insert_immobile(nr, c, frame_origin);
        }
    };

    typedef node_t<parsing_struct> node;
    boost::optional<sz> immobile_atom;           // which of `atoms' is immobile, if any
    boost::optional<atom_reference> axis_begin;  // the index (in non_rigid_parsed::atoms) of the
                                                 // parent bound to immobile atom (if already known)
    boost::optional<atom_reference> axis_end;    // if immobile atom has been pushed into
                                                 // non_rigid_parsed::atoms, this is its index there
    std::vector<node> atoms;

    void add(const parsed_atom& a, const context& c, bool keep_H = true) {
        VINA_CHECK(c.size() > 0);
        if (a.ad == AD_TYPE_H && keep_H == false) return;
        atoms.emplace_back(node(a, c.size() - 1));
    }
    void add(const parsed_atom& a, const sz context_index, bool keep_H = true) {
        VINA_CHECK(context_index > 0);
        if (a.ad == AD_TYPE_H && keep_H == false) return;
        atoms.emplace_back(node(a, context_index));
    }
    const vec& immobile_atom_coords() const {
        VINA_CHECK(immobile_atom);
        VINA_CHECK(immobile_atom.get() < atoms.size());
        return atoms[immobile_atom.get()].a.coords;
    }
    // inflex insertion
    void insert_immobile_inflex(non_rigid_parsed& nr) {
        if (!atoms.empty()) {
            VINA_CHECK(immobile_atom);
            VINA_CHECK(immobile_atom.get() < atoms.size());
            axis_end = atom_reference(nr.inflex.size(), true);
            atoms[immobile_atom.get()].insert_inflex(nr);
        }
    }

    // insertion into non_rigid_parsed
    void insert_immobile(non_rigid_parsed& nr, context& c, const vec& frame_origin) {
        if (!atoms.empty()) {
            VINA_CHECK(immobile_atom);
            VINA_CHECK(immobile_atom.get() < atoms.size());
            axis_end = atom_reference(nr.atoms.size(), false);
            atoms[immobile_atom.get()].insert(nr, c, frame_origin);
        }
    }

    bool essentially_empty()
        const {  // no sub-branches besides immobile atom, including sub-sub-branches, etc
        VINA_FOR_IN(i, atoms) {
            if (immobile_atom && immobile_atom.get() != i) return false;
            const node& nd = atoms[i];
            if (!nd.ps.empty()) return false;  // FIXME : iffy
        }
        return true;
    }
};

unsigned parse_one_unsigned(const std::string& str, const std::string& start) {
    std::istringstream in_str(str.substr(start.size()));
    int tmp;
    in_str >> tmp;

    if (!in_str || tmp < 0) throw struct_parse_error("Syntax error.", str);

    return unsigned(tmp);
}

void parse_two_unsigneds(const std::string& str, const std::string& start, unsigned& first,
                         unsigned& second) {
    std::istringstream in_str(str.substr(start.size()));
    int tmp1, tmp2;
    in_str >> tmp1;
    in_str >> tmp2;

    if (!in_str || tmp1 < 0 || tmp2 < 0) throw struct_parse_error("Syntax error.", str);

    first = unsigned(tmp1);
    second = unsigned(tmp2);
}

void parse_pdbqt_rigid(const path& name, rigid& r) {
    ifile in(name);
    std::string str;

    while (std::getline(in, str)) {
        if (str.empty()) {
        }  // ignore ""
        else if (starts_with(str, "TER")) {
        }  // ignore
        else if (starts_with(str, "END")) {
        }  // ignore
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "ATOM  ") || starts_with(str, "HETATM"))
            r.atoms.push_back(parse_pdbqt_atom_string(str));
        else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in rigid receptor. "
                "Only one model can be used for the rigid receptor.");
        else
            throw struct_parse_error("Unknown or inappropriate tag found in rigid receptor.", str);
    }
}

void parse_pdbqt_root_aux(std::istream& in, parsing_struct& p, context& c, bool keep_H = true) {
    std::string str;

    while (std::getline(in, str)) {
        add_context(c, str);

        if (str.empty()) {
        }  // ignore ""
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "ATOM  ") || starts_with(str, "HETATM"))
            p.add(parse_pdbqt_atom_string(str), c, keep_H);
        else if (starts_with(str, "ENDROOT"))
            return;
        else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in flex residue or ligand PDBQT file. "
                "Use \"vina_split\" to split flex residues or ligands in multiple PDBQT files.");
        else
            throw struct_parse_error(
                "Unknown or inappropriate tag found in flex residue or ligand.", str);
    }
}

void parse_pdbqt_root(std::istream& in, parsing_struct& p, context& c, bool keep_H = true) {
    std::string str;

    while (std::getline(in, str)) {
        add_context(c, str);

        if (str.empty()) {
        }  // ignore
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "ROOT")) {
            parse_pdbqt_root_aux(in, p, c, keep_H);
            break;
        } else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in flex residue or ligand PDBQT file. "
                "Use \"vina_split\" to split flex residues or ligands in multiple PDBQT files.");
        else
            throw struct_parse_error(
                "Unknown or inappropriate tag found in flex residue or ligand.", str);
    }
}

void parse_pdbqt_branch(std::istream& in, parsing_struct& p, context& c, unsigned from, unsigned to,
                        bool keep_H = true);  // forward declaration
void parse_sdf_branch(std::vector<std::vector<int> >& frags,
                      std::vector<std::vector<int> >& torsions, int frag_id, parsing_struct& new_p,
                      parsing_struct& p, context& c, unsigned& number, unsigned from, unsigned to,
                      std::set<int>& been_frags, bool keep_H = true);

void parse_pdbqt_branch_aux(std::istream& in, const std::string& str, parsing_struct& p, context& c,
                            bool keep_H = true) {
    unsigned first, second;
    parse_two_unsigneds(str, "BRANCH", first, second);
    sz i = 0;

    for (; i < p.atoms.size(); ++i)
        if (p.atoms[i].a.number == first) {
            parsing_struct p0;
            p.atoms[i].ps.push_back(p0);
            parse_pdbqt_branch(in, p.atoms[i].ps.back(), c, first, second, keep_H);
            break;
        }

    if (i == p.atoms.size())
        throw struct_parse_error(
            "Atom number " + std::to_string(first) + " is missing in this branch.", str);
}

void parse_sdf_branch_aux(std::vector<std::vector<int> >& frags,
                          std::vector<std::vector<int> >& torsions, int frag_id,
                          parsing_struct& new_p, parsing_struct& p, context& c, unsigned& number,
                          int from, int to, std::set<int>& been_frags, bool keep_H = true) {
    sz i = 0;
    // std::cout << "entering parse_sdf_branch_aux " << from << ' '  << to << std::endl;

    for (; i < new_p.atoms.size(); ++i) {
        // printf("new_p.atoms[i].a.number_sdf=%d\n",new_p.atoms[i].a.number_sdf);
        if (new_p.atoms[i].a.number_sdf == from && been_frags.find(frag_id) == been_frags.end()) {
            // std::cout << "pushing atom in parse_sdf_branch_aux i=" << i << ' '  <<
            // new_p.atoms.size() << std::endl;
            parsing_struct p0;
            new_p.atoms[i].ps.push_back(p0);
            been_frags.insert(frag_id);
            // std::cout << "current frag id=" << frag_id << ", from=" << from << std::endl;
            parse_sdf_branch(frags, torsions, frag_id, new_p.atoms[i].ps.back(), p, c, number, from,
                             to, been_frags, keep_H);
            break;
        }
    }
}

void parse_pdbqt_aux(std::istream& in, parsing_struct& p, context& c,
                     boost::optional<unsigned>& torsdof, bool residue, bool keep_H = true) {
    parse_pdbqt_root(in, p, c, keep_H);

    std::string str;

    while (std::getline(in, str)) {
        add_context(c, str);

        if (str.empty()) {
        }  // ignore ""
        if (str[0] == '\0') {
        }  // ignore a different kind of emptiness (potential issues on Windows)
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "BRANCH"))
            parse_pdbqt_branch_aux(in, str, p, c, keep_H);
        else if (!residue && starts_with(str, "TORSDOF")) {
            if (torsdof) throw struct_parse_error("TORSDOF keyword can be defined only once.");
            torsdof = parse_one_unsigned(str, "TORSDOF");
        } else if (residue && starts_with(str, "END_RES"))
            return;
        else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in flex residue or ligand PDBQT file. "
                "Use \"vina_split\" to split flex residues or ligands in multiple PDBQT files.");
        else
            throw struct_parse_error(
                "Unknown or inappropriate tag found in flex residue or ligand.", str);
    }
}

void add_bonds(non_rigid_parsed& nr, boost::optional<atom_reference> atm, const atom_range& r) {
    if (atm) VINA_RANGE(i, r.begin, r.end) {
            atom_reference& ar = atm.get();
            if (ar.inflex)
                nr.atoms_inflex_bonds(i, ar.index)
                    = DISTANCE_FIXED;  //(max_unsigned); // first index - atoms, second index -
                                       // inflex
            else
                nr.atoms_atoms_bonds(ar.index, i) = DISTANCE_FIXED;  // (max_unsigned);
        }
}

void set_rotor(non_rigid_parsed& nr, boost::optional<atom_reference> axis_begin,
               boost::optional<atom_reference> axis_end) {
    if (axis_begin && axis_end) {
        atom_reference& r1 = axis_begin.get();
        atom_reference& r2 = axis_end.get();
        if (r2.inflex) {
            VINA_CHECK(r1.inflex);  // no atom-inflex rotors
            nr.inflex_inflex_bonds(r1.index, r2.index) = DISTANCE_ROTOR;
        } else if (r1.inflex)
            nr.atoms_inflex_bonds(r2.index, r1.index) = DISTANCE_ROTOR;  // (atoms, inflex)
        else
            nr.atoms_atoms_bonds(r1.index, r2.index) = DISTANCE_ROTOR;
    }
}

typedef std::pair<sz, sz> axis_numbers;
typedef boost::optional<axis_numbers> axis_numbers_option;

void nr_update_matrixes(non_rigid_parsed& nr) {
    // atoms with indexes p.axis_begin and p.axis_end can not move relative to [b.node.begin,
    // b.node.end)

    nr.atoms_atoms_bonds.resize(nr.atoms.size(), DISTANCE_VARIABLE);
    nr.atoms_inflex_bonds.resize(nr.atoms.size(), nr.inflex.size(),
                                 DISTANCE_VARIABLE);  // first index - inflex, second index - atoms
    nr.inflex_inflex_bonds.resize(nr.inflex.size(), DISTANCE_FIXED);  // FIXME?
}

template <typename B>  // B == branch or main_branch or flexible_body
void postprocess_branch(non_rigid_parsed& nr, parsing_struct& p, context& c, B& b) {
    b.node.begin = nr.atoms.size();
    VINA_FOR_IN(i, p.atoms) {  // postprocess atoms into 'b.node'
        parsing_struct::node& p_node = p.atoms[i];
        if (p.immobile_atom && i == p.immobile_atom.get()) {
        }  // skip immobile_atom - it's already inserted in "THERE"
        else
            p_node.insert(nr, c, b.node.get_origin());
        p_node.insert_immobiles(nr, c, b.node.get_origin());
    }
    b.node.end = nr.atoms.size();

    nr_update_matrixes(nr);
    add_bonds(nr, p.axis_begin, b.node);  // b.node is used as atom_range
    add_bonds(nr, p.axis_end, b.node);    // b.node is used as atom_range
    set_rotor(nr, p.axis_begin, p.axis_end);

    VINA_RANGE(i, b.node.begin, b.node.end)
    VINA_RANGE(j, i + 1, b.node.end)
    nr.atoms_atoms_bonds(i, j) = DISTANCE_FIXED;  // FIXME

    VINA_FOR_IN(i, p.atoms) {  // postprocess children
        parsing_struct::node& p_node = p.atoms[i];
        VINA_FOR_IN(j, p_node.ps) {
            parsing_struct& ps = p_node.ps[j];
            if (!ps.essentially_empty()) {  // immobile already inserted // FIXME ?!
                b.children.push_back(
                    segment(ps.immobile_atom_coords(), 0, 0, p_node.a.coords,
                            b.node));  // postprocess_branch will assign begin and end
                postprocess_branch(nr, ps, c, b.children.back());
            }
        }
    }
    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
    VINA_CHECK(nr.atoms_inflex_bonds.dim_1() == nr.atoms.size());
    VINA_CHECK(nr.atoms_inflex_bonds.dim_2() == nr.inflex.size());
}

void postprocess_ligand(non_rigid_parsed& nr, parsing_struct& p, context& c, unsigned torsdof) {
    VINA_CHECK(!p.atoms.empty());
    nr.ligands.push_back(ligand(flexible_body(rigid_body(p.atoms[0].a.coords, 0, 0)),
                                torsdof));  // postprocess_branch will assign begin and end
    postprocess_branch(nr, p, c, nr.ligands.back());
    nr_update_matrixes(nr);  // FIXME ?
}

void postprocess_residue(non_rigid_parsed& nr, parsing_struct& p, context& c) {
    VINA_FOR_IN(i, p.atoms) {  // iterate over "root" of a "residue"
        parsing_struct::node& p_node = p.atoms[i];
        p_node.insert_inflex(nr);
        p_node.insert_immobiles_inflex(nr);
    }
    VINA_FOR_IN(i, p.atoms) {  // iterate over "root" of a "residue"
        parsing_struct::node& p_node = p.atoms[i];
        VINA_FOR_IN(j, p_node.ps) {
            parsing_struct& ps = p_node.ps[j];
            if (!ps.essentially_empty()) {  // immobile atom already inserted // FIXME ?!
                nr.flex.push_back(main_branch(first_segment(
                    ps.immobile_atom_coords(), 0, 0,
                    p_node.a.coords)));  // postprocess_branch will assign begin and end
                postprocess_branch(nr, ps, c, nr.flex.back());
            }
        }
    }
    nr_update_matrixes(nr);  // FIXME ?
    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
    VINA_CHECK(nr.atoms_inflex_bonds.dim_1() == nr.atoms.size());
    VINA_CHECK(nr.atoms_inflex_bonds.dim_2() == nr.inflex.size());
}

// dkoes, stream version
void parse_pdbqt_ligand(std::istream& in, non_rigid_parsed& nr, context& c) {
    parsing_struct p;
    boost::optional<unsigned> torsdof;

    parse_pdbqt_aux(in, p, c, torsdof, false);

    if (p.atoms.empty()) throw struct_parse_error("No atoms in this ligand.");
    if (!torsdof) throw struct_parse_error("Missing TORSDOF keyword.");

    try {
        postprocess_ligand(
            nr, p, c, unsigned(torsdof.get()));  // bizarre size_t -> unsigned compiler complaint
    } catch (int e) {
        if (e == 1) {
            throw struct_parse_error("Ligand with zero coords.");
        }
    }

    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
}

void parse_pdbqt_ligand(const path& name, non_rigid_parsed& nr, context& c, bool keep_H = true) {
    ifile in(name);
    parsing_struct p;
    boost::optional<unsigned> torsdof;

    parse_pdbqt_aux(in, p, c, torsdof, false, keep_H);

    if (p.atoms.empty()) throw struct_parse_error("No atoms in this ligand.");
    if (!torsdof) throw struct_parse_error("Missing TORSDOF keyword in this ligand.");

    try {
        postprocess_ligand(
            nr, p, c, unsigned(torsdof.get()));  // bizarre size_t -> unsigned compiler complaint
    } catch (int e) {
        if (e == 1) {
            throw struct_parse_error("Ligand with zero coords.");
        }
    }

    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
}

void parse_sdf_aux(std::istream& in, parsing_struct& new_p, parsing_struct& p, context& c,
                   unsigned& torsdof, bool residue, bool keep_H = true) {
    std::string str;
    // sdf header has three lines
    for (int i = 0; i < 3; ++i) {
        std::getline(in, str);
        // std::cout << "read sdf line:" << str << std::endl;
        add_context(c, str);
    }

    // parse counts line

    std::getline(in, str);
    // std::cout << "read sdf line:" << str << std::endl;
    add_context(c, str);
    int atom_num = checked_convert_substring<fl>(str, 1, 3, "Atom num");
    int bond_num = checked_convert_substring<fl>(str, 4, 6, "Bond num");
    // int property_num = checked_convert_substring<fl>(str,  34, 36, "Property num");

    for (int i = 0; i < atom_num; ++i) {
        std::getline(in, str);
        add_context(c, str);
        // std::cout << "read sdf line:" << str << std::endl;
        parsed_atom a = parse_sdf_atom_string(str, i + 1);
        p.add(a, c, true);
    }

    for (int i = 0; i < bond_num; ++i) {
        std::getline(in, str);
        add_context(c, str);
        // std::cout << "read sdf bond line:" << str << std::endl;
    }

    // read property
    while (std::getline(in, str)) {
        add_context(c, str);
        // std::cout << "read sdf property line:" << str << ' ' << str.find("M  END") << std::endl;

        if (str.find("M  END") < str.length()) {
            break;
        }
    }

    // use property given by ligprep to construct tree
    std::vector<std::vector<int> > frags;
    std::vector<std::vector<int> > torsions;

    while (std::getline(in, str)) {
        if (str.find("$$$$") < str.length()) continue;
        add_context(c, str);
        // std::cout << "read sdf line:" << str << std::endl;

        if (str[0] == '>') {
            std::string data_type = str.substr(6, str.length() - 7);
            if (str.find("atomInfo") < str.length() || str.find("atom_info") < str.length()) {
                // update p.atoms[num].a.charge and type
                while (std::getline(in, str)) {
                    add_context(c, str);
                    // std::cout << "read info sdf line:" << str << std::endl;
                    if (str.empty()) {
                        break;
                    }
                    std::string ad_name = omit_whitespace(str, 14, 14);
                    int atomid = checked_convert_substring<int>(
                        str, 1, std::min(unsigned(str.find(' ')), 3U), "AtomId");
                    // std::cout << "atomid=" << atomid << ",  ad_name=" << ad_name << std::endl;
                    fl charge = checked_convert_substring<fl>(str, 4, 13, "Partial Charge");
                    sz ad = string_to_ad_type(ad_name);
                    p.atoms[atomid - 1].a.charge = charge;
                    p.atoms[atomid - 1].a.ad = ad;
                }
            } else if (str.find("torsion") < str.length()) {
                // update p.atoms[num].a.charge
                // std::cout << "start torsion" << std::endl;
                while (std::getline(in, str)) {
                    add_context(c, str);
                    // std::cout << "read torsion sdf line:" << str << std::endl;
                    if (str.empty()) {
                        break;
                    }

                    std::vector<int> torsion;
                    int num = int(str[0]) - int('0');

                    for (int i = 1; i < str.length(); ++i) {
                        if (str[i] == ' ') {
                            torsion.push_back(num);
                            num = 0;
                        } else {
                            num = num * 10 + int(str[i]) - int('0');
                        }
                    }
                    torsion.push_back(num);
                    torsions.push_back(torsion);
                }
            } else if (str.find("frag") < str.length()) {
                while (std::getline(in, str)) {
                    add_context(c, str);
                    // std::cout << "read frag sdf line:" << str << std::endl;
                    if (str.empty()) {
                        break;
                    }
                    std::vector<int> frag;
                    int num = int(str[0]) - int('0');

                    for (int i = 1; i < str.length(); ++i) {
                        if (str[i] == ' ') {
                            frag.push_back(num);
                            num = 0;
                        } else {
                            num = num * 10 + int(str[i]) - int('0');
                        }
                    }
                    frag.push_back(num);
                    frags.push_back(frag);
                }
            } else {
                while (std::getline(in, str)) {
                    add_context(c, str);
                    // std::cout << "read und sdf line:" << str << std::endl;
                    if (str.empty()) {
                        break;
                    }
                }
            }
        }
    }

    torsdof = unsigned(torsions.size());

    // print_zero();
    // similar to parse_pdbqt_root

    if (!keep_H) {
        for (int i = 0; i < frags.size(); ++i) {
            std::vector<int> new_frag_nonH;
            for (int j = 0; j < frags[i].size(); ++j) {
                if (p.atoms[frags[i][j] - 1].a.ad != AD_TYPE_H) {
                    new_frag_nonH.push_back(frags[i][j]);
                    // std::cout << "atom num=" << frags[i][j] << " , AD type = " <<
                    // p.atoms[frags[i][j]-1].a.ad << std::endl;
                } else {
                    // std::cout << "atom num=" << frags[i][j] << " is H, omitted" << std::endl;
                }
            }
            frags[i] = new_frag_nonH;
        }
    } else {
        for (int i = 0; i < frags.size(); ++i) {
            std::vector<int> new_frag_keep_H;
            for (int j = 0; j < frags[i].size(); ++j) {
                new_frag_keep_H.push_back(frags[i][j]);
                // std::cout << "atom num=" << frags[i][j] << " , AD type = " <<
                // p.atoms[frags[i][j]-1].a.ad << std::endl;
            }
            frags[i] = new_frag_keep_H;
        }
    }
    if (frags.size() == 0) {
        std::cerr << "No fragment info, using rigid docking" << std::endl;
        torsdof = 0;
        new_p = p;
        return;  // do not use new p
    }
    // print_zero();
    // print_zero();
    int max_torsion_frag_id, max_atom_frag_id;
    int center_atom_id, center_atom_frag_id = 0;
    int max_atom_frag = -1;
    float center_distance2 = 1000;

    for (int i = 0; i < frags.size(); ++i) {
        if (frags[i].size() > max_atom_frag) {
            max_atom_frag_id = i;
            max_atom_frag = frags[i].size();
        }
    }

    vec center = {0, 0, 0};
    for (int i = 0; i < p.atoms.size(); ++i) {
        center = center + p.atoms[i].a.coords;
    }
    center = center / float(p.atoms.size());
    for (int i = 0; i < p.atoms.size(); ++i) {
        vec dvec = center - p.atoms[i].a.coords;
        float dist2 = dvec.norm_sqr();
        if (dist2 < center_distance2 && p.atoms[i].a.ad != AD_TYPE_H) {
            center_distance2 = dist2;
            center_atom_id = i;
        }
    }
    // std::cout << center[0] << ' ' << center[1] << ' ' << center[2] << "atom:" << center_atom_id
    // << std::endl;
    for (int i = 0; i < frags.size(); ++i) {
        for (int j = 0; j < frags[i].size(); ++j) {
            if (frags[i][j] - 1 == center_atom_id) {
                center_atom_frag_id = i;
                break;
            }
        }
    }
    // for (int i = 1;i <= atom_num;++i){
    //     int cnt_torsion = 0;
    //     for (int j = 0;j < torsions.size();++j){
    //         if (torsions[j][0] == i || torsions[j][1] == i){
    //             ++cnt_torsion;
    //         }
    //     }
    //     if (cnt_torsion > max_torsion){
    //         max_torsion = cnt_torsion;
    //         max_torsion_frag_id = i;
    //     }
    // }

    // use fragment #0 as root fragment and root is atom #0
    max_atom_frag_id = 0;

    // use center frag as root frag
    // max_atom_frag_id = center_atom_frag_id;
    // std::cout << "start with " << max_atom_frag_id << std::endl;

    unsigned number = 0;

    for (int i = 0; i < frags[max_atom_frag_id].size(); ++i) {
        p.atoms[frags[max_atom_frag_id][i] - 1].a.number
            = number;  // assign new number of tree structure
        ++number;
        // std::cout << "pushing atom in parse_sdf_aux i=" << i << ' '  <<
        // frags[max_atom_frag_id].size() << std::endl;

        new_p.add(p.atoms[frags[max_atom_frag_id][i] - 1].a,
                  p.atoms[frags[max_atom_frag_id][i] - 1].context_index,
                  true);  // keep_H is set to true, controled by frag
    }
    // similar to parse_pdbqt_branch_aux
    // if(starts_with(str, "BRANCH")) parse_pdbqt_branch_aux(in, str, p, c);

    std::set<int> been_frags = {max_atom_frag_id};  // prevent dead loop caused by fraginfo errors
    for (int i = 0; i < frags[max_atom_frag_id].size(); ++i) {
        for (int j = 0; j < torsions.size(); ++j) {
            // std::cout << "j=" << j << std::endl;
            if (torsions[j][0] == frags[max_atom_frag_id][i]) {
                int frag_id = torsions[j][3];
                parse_sdf_branch_aux(frags, torsions, frag_id, new_p, p, c, number, torsions[j][0],
                                     torsions[j][1], been_frags, keep_H);
            } else if (torsions[j][1] == frags[max_atom_frag_id][i]) {
                int frag_id = torsions[j][2];
                parse_sdf_branch_aux(frags, torsions, frag_id, new_p, p, c, number, torsions[j][1],
                                     torsions[j][0], been_frags, keep_H);
            }
        }
    }
}

void parse_sdf_ligand(const path& name, non_rigid_parsed& nr, context& c, bool keep_H = true) {
    ifile in(name);
    parsing_struct* p = new parsing_struct();
    parsing_struct* new_p = new parsing_struct();
    unsigned int torsdof;

    // transfer_parsing_struct
    parse_sdf_aux(in, *new_p, *p, c, torsdof, false, keep_H);
    // free(p);

    // print_zero();
    if (new_p->atoms.empty()) throw struct_parse_error("No atoms in this ligand.");

    try {
        postprocess_ligand(nr, *new_p, c,
                           unsigned(torsdof));  // bizarre size_t -> unsigned compiler complaint
    } catch (int e) {
        if (e == 1) {
            throw struct_parse_error("Ligand with zero coords.");
        }
    }

    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
}

void parse_pdbqt_residue(std::istream& in, parsing_struct& p, context& c) {
    boost::optional<unsigned> dummy;
    parse_pdbqt_aux(in, p, c, dummy, true);
}

void parse_pdbqt_flex(const path& name, non_rigid_parsed& nr, context& c) {
    ifile in(name);
    std::string str;

    while (std::getline(in, str)) {
        add_context(c, str);

        if (str.empty()) {
        }  // ignore ""
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "BEGIN_RES")) {
            parsing_struct p;
            parse_pdbqt_residue(in, p, c);
            postprocess_residue(nr, p, c);
        } else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in flex residue PDBQT file. "
                "Use \"vina_split\" to split flex residues in multiple PDBQT files.");
        else
            throw struct_parse_error("Unknown or inappropriate tag found in flex residue.", str);
    }

    VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());
}

void parse_pdbqt_branch(std::istream& in, parsing_struct& p, context& c, unsigned from, unsigned to,
                        bool keep_H) {
    std::string str;

    while (std::getline(in, str)) {
        add_context(c, str);

        if (str.empty()) {
        }  // ignore ""
        else if (starts_with(str, "WARNING")) {
        }  // ignore - AutoDockTools bug workaround
        else if (starts_with(str, "REMARK")) {
        }  // ignore
        else if (starts_with(str, "BRANCH"))
            parse_pdbqt_branch_aux(in, str, p, c, keep_H);
        else if (starts_with(str, "ENDBRANCH")) {
            unsigned first, second;
            parse_two_unsigneds(str, "ENDBRANCH", first, second);
            if (first != from || second != to)
                throw struct_parse_error("Inconsistent branch numbers.");
            if (!p.immobile_atom)
                throw struct_parse_error("Atom " + boost::lexical_cast<std::string>(to)
                                         + " has not been found in this branch.");
            return;
        } else if (starts_with(str, "ATOM  ") || starts_with(str, "HETATM")) {
            parsed_atom a = parse_pdbqt_atom_string(str);
            if (a.number == to) p.immobile_atom = p.atoms.size();
            p.add(a, c, keep_H);
        } else if (starts_with(str, "MODEL"))
            throw struct_parse_error(
                "Unexpected multi-MODEL tag found in flex residue or ligand PDBQT file. "
                "Use \"vina_split\" to split flex residues or ligands in multiple PDBQT files.");
        else
            throw struct_parse_error(
                "Unknown or inappropriate tag found in flex residue or ligand.", str);
    }
}

void parse_sdf_branch(std::vector<std::vector<int> >& frags,
                      std::vector<std::vector<int> >& torsions, int frag_id, parsing_struct& new_p,
                      parsing_struct& p, context& c, unsigned& number, unsigned from, unsigned to,
                      std::set<int>& been_frags, bool keep_H) {
    // std::cout << "entering parse_sdf_branch frag= "<< frag_id << ' ' << from << ' '  << to <<
    // std::endl;

    // push new fragment atoms into new_p
    // new_p.atoms.reserve(frags[frag_id].size());
    for (int i = 0; i < frags[frag_id].size(); ++i) {
        if (p.atoms[frags[frag_id][i] - 1].a.number_sdf == to) {
            new_p.immobile_atom = new_p.atoms.size();
        }
        p.atoms[frags[frag_id][i] - 1].a.number = number;
        ++number;
        // debug
        //  std::cout << "pushing atom in parse_sdf_branch i=" << i << "frag id=" << frag_id << ' '
        //  << frags[frag_id].size() << std::endl;
        new_p.add(p.atoms[frags[frag_id][i] - 1].a, p.atoms[frags[frag_id][i] - 1].context_index,
                  keep_H);
        // new_p.atoms.push_back(p.atoms[frags[frag_id][i]-1]); // equal to p.add()
    }
    for (int i = 0; i < frags[frag_id].size(); ++i) {
        for (int j = 0; j < torsions.size(); ++j) {
            if (torsions[j][0] == frags[frag_id][i]) {
                int next_frag_id = torsions[j][3];
                parse_sdf_branch_aux(frags, torsions, next_frag_id, new_p, p, c, number,
                                     torsions[j][0], torsions[j][1], been_frags, keep_H);
            } else if (torsions[j][1] == frags[frag_id][i]) {
                int next_frag_id = torsions[j][2];
                parse_sdf_branch_aux(frags, torsions, next_frag_id, new_p, p, c, number,
                                     torsions[j][1], torsions[j][0], been_frags, keep_H);
            }
        }
    }

    if (!new_p.immobile_atom)
        throw struct_parse_error("Atom " + boost::lexical_cast<std::string>(to)
                                 + " has not been found in this branch.");

    return;
}

//////////// new stuff //////////////////
struct pdbqt_initializer {
    atom_type::t atom_typing_used;
    model m;
    // pdbqt_initializer(): atom_typing_used(atom_type::XS), m(atom_type::XS) {}
    pdbqt_initializer(atom_type::t atype) : atom_typing_used(atype), m(atype) {}
    void initialize_from_rigid(const rigid& r) {  // static really
        VINA_CHECK(m.grid_atoms.empty());
        m.grid_atoms = r.atoms;
    }
    void initialize_from_nrp(const non_rigid_parsed& nrp, const context& c,
                             bool is_ligand) {  // static really
        VINA_CHECK(m.ligands.empty());
        VINA_CHECK(m.flex.empty());

        m.ligands = nrp.ligands;
        m.flex = nrp.flex;

        VINA_CHECK(m.atoms.empty());

        sz n = nrp.atoms.size() + nrp.inflex.size();
        m.atoms.reserve(n);
        m.coords.reserve(n);

        VINA_FOR_IN(i, nrp.atoms) {
            const movable_atom& a = nrp.atoms[i];
            atom b = static_cast<atom>(a);
            b.coords = a.relative_coords;
            m.atoms.push_back(b);
            m.coords.push_back(a.coords);
        }
        VINA_FOR_IN(i, nrp.inflex) {
            const atom& a = nrp.inflex[i];
            atom b = a;
            b.coords
                = zero_vec;  // to avoid any confusion; presumably these will never be looked at
            m.atoms.push_back(b);
            m.coords.push_back(a.coords);
        }
        VINA_CHECK(m.coords.size() == n);

        m.minus_forces = m.coords;
        m.m_num_movable_atoms = nrp.atoms.size();

        if (is_ligand) {
            VINA_CHECK(m.ligands.size() == 1);
            m.ligands.front().cont = c;
        } else
            m.flex_context = c;
    }
    void initialize(const distance_type_matrix& mobility) { m.initialize(mobility); }
};

model parse_ligand_pdbqt_from_file(const std::string& name,
                                   atom_type::t atype) {  // can throw parse_error
    non_rigid_parsed nrp;
    context c;

    try {
        parse_pdbqt_ligand(make_path(name), nrp, c);
    } catch (struct_parse_error& e) {
        std::cerr << e.what();
        exit(EXIT_FAILURE);
    }

    pdbqt_initializer tmp(atype);
    tmp.initialize_from_nrp(nrp, c, true);
    tmp.initialize(nrp.mobility_matrix());
    return tmp.m;
}

model parse_ligand_from_file_no_failure(const std::string& name, atom_type::t atype,
                                        bool keep_H) {  // can throw parse_error
    DEBUG_PRINTF("ligand name: %s\n", name.c_str());    // debug
    // std::cout << name.substr(name.length()-5,5) << std::endl;
    if (strcmp("pdbqt", name.substr(name.length() - 5, 5).c_str()) == 0) {
        return parse_ligand_pdbqt_from_file_no_failure(name, atype, keep_H);
    } else if (strcmp("sdf", name.substr(name.length() - 3, 3).c_str()) == 0) {
        return parse_ligand_sdf_from_file_no_failure(name, atype, keep_H);
    }
    model m(atype);
    return m;
}

model parse_ligand_pdbqt_from_file_no_failure(const std::string& name, atom_type::t atype,
                                              bool keep_H) {  // can throw parse_error
    non_rigid_parsed nrp;
    context c;

    try {
        parse_pdbqt_ligand(make_path(name), nrp, c, keep_H);
    } catch (struct_parse_error& e) {
        std::cerr << e.what() << "Ligand name:" << name << "\n\n";
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;  // return empty model as failure, ligand.size = 0
    }

    pdbqt_initializer tmp(atype);
    tmp.initialize_from_nrp(nrp, c, true);
    tmp.initialize(nrp.mobility_matrix());
    assert(tmp.m.ligands.count_torsions().size() == 1);
    if (tmp.m.ligands.count_torsions()[0] > MAX_NUM_OF_LIG_TORSION) {
        std::cerr << "Ligand " << name << " exceed max torsion counts. "
                  << tmp.m.ligands.count_torsions()[0] << std::endl;
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;
    }
    if (tmp.m.atoms.size() > MAX_NUM_OF_ATOMS) {
        std::cerr << "Ligand " << name << " exceed max atom counts. " << tmp.m.atoms.size()
                  << std::endl;
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;
    }
    return tmp.m;
}

model parse_ligand_sdf_from_file_no_failure(const std::string& name, atom_type::t atype,
                                            bool keep_H) {  // can throw parse_error
    non_rigid_parsed nrp;
    context c;

    try {
        parse_sdf_ligand(make_path(name), nrp, c, keep_H);
    } catch (struct_parse_error& e) {
        std::cerr << e.what() << "Ligand name:" << name << "\n\n";
        model m_(atype);
        assert(m_.num_ligands() == 0);
        return m_;  // return empty model as failure, ligand.size = 0
    }

    // the rest is the same
    pdbqt_initializer tmp(atype);
    tmp.initialize_from_nrp(nrp, c, true);
    tmp.initialize(nrp.mobility_matrix());
    assert(tmp.m.ligands.count_torsions().size() == 1);
    // // debug
    // for (int i = 0;i < tmp.m.atoms.size(); ++i){
    //     printf("atom type of model ad=%lu, xs=%lu, numsdf=%d\n", tmp.m.atoms[i].ad,
    //     tmp.m.atoms[i].xs, tmp.m.atoms[i].number_sdf);
    // }
    if (tmp.m.ligands.count_torsions()[0] > MAX_NUM_OF_LIG_TORSION) {
        std::cerr << "Ligand " << name << " exceed max torsion counts. "
                  << tmp.m.ligands.count_torsions()[0] << std::endl;
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;
    }
    if (tmp.m.atoms.size() > MAX_NUM_OF_ATOMS) {
        std::cerr << "Ligand " << name << " exceed max atom counts. " << tmp.m.atoms.size()
                  << std::endl;
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;
    }
    return tmp.m;

    // assert(m.ligands.count_torsions().size()==1);
    // if(m.ligands.count_torsions()[0] > MAX_NUM_OF_LIG_TORSION)
    // {
    //     std::cerr << "Ligand " << name << " exceed max torsion counts. " <<
    //     m.ligands.count_torsions()[0] << std::endl; model m_(atype); assert(m_.num_ligands() ==
    //     0); return m_;
    // }
    // return m;
}

model parse_ligand_pdbqt_from_string(const std::string& string_name,
                                     atom_type::t atype) {  // can throw parse_error
    non_rigid_parsed nrp;
    context c;

    try {
        std::stringstream molstream(string_name);
        parse_pdbqt_ligand(molstream, nrp, c);
    } catch (struct_parse_error& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    pdbqt_initializer tmp(atype);
    tmp.initialize_from_nrp(nrp, c, true);
    tmp.initialize(nrp.mobility_matrix());
    return tmp.m;
}

model parse_ligand_pdbqt_from_string_no_failure(const std::string& string_name,
                                                atom_type::t atype) {  // can throw parse_error
    non_rigid_parsed nrp;
    context c;

    try {
        std::stringstream molstream(string_name);
        parse_pdbqt_ligand(molstream, nrp, c);
    } catch (struct_parse_error& e) {
        std::cerr << e.what() << '\n';
        model m(atype);
        assert(m.num_ligands() == 0);
        return m;  // return empty model as failure, ligand.size = 0
    }

    pdbqt_initializer tmp(atype);
    tmp.initialize_from_nrp(nrp, c, true);
    tmp.initialize(nrp.mobility_matrix());
    return tmp.m;
}

model parse_receptor_pdbqt(const std::string& rigid_name, const std::string& flex_name,
                           atom_type::t atype) {
    // Parse PDBQT receptor with flex residues
    // if (rigid_name.empty() && flex_name.empty()) {
    //    // CONDITION 1
    //    std::cerr << "ERROR: No (rigid) receptor or flexible residues were specified.\n";
    //    exit(EXIT_FAILURE);
    //}

    rigid r;
    non_rigid_parsed nrp;
    context c;
    pdbqt_initializer tmp(atype);

    if (!rigid_name.empty()) {
        try {
            parse_pdbqt_rigid(make_path(rigid_name), r);
        } catch (struct_parse_error& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if (!flex_name.empty()) {
        try {
            parse_pdbqt_flex(make_path(flex_name), nrp, c);
        } catch (struct_parse_error& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if (!rigid_name.empty()) {
        tmp.initialize_from_rigid(r);
        if (flex_name.empty()) {
            distance_type_matrix mobility_matrix;
            tmp.initialize(mobility_matrix);
        }
    }

    if (!flex_name.empty()) {
        tmp.initialize_from_nrp(nrp, c, false);
        tmp.initialize(nrp.mobility_matrix());
    }

    return tmp.m;
}

model parse_receptor_pdb(const std::string& rigid_name, const std::string& flex_name,
                         atom_type::t atype) {
    // Parse PDBQT receptor with flex residues
    // if (rigid_name.empty() && flex_name.empty()) {
    //    // CONDITION 1
    //    std::cerr << "ERROR: No (rigid) receptor or flexible residues were specified.\n";
    //    exit(EXIT_FAILURE);
    //}

    rigid r;
    non_rigid_parsed nrp;
    context c;
    pdbqt_initializer tmp(atype);

    if (!rigid_name.empty()) {
        try {
            parse_pdbqt_rigid(make_path(rigid_name), r);
        } catch (struct_parse_error& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if (!flex_name.empty()) {
        try {
            parse_pdbqt_flex(make_path(flex_name), nrp, c);
        } catch (struct_parse_error& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if (!rigid_name.empty()) {
        tmp.initialize_from_rigid(r);
        if (flex_name.empty()) {
            distance_type_matrix mobility_matrix;
            tmp.initialize(mobility_matrix);
        }
    }

    if (!flex_name.empty()) {
        tmp.initialize_from_nrp(nrp, c, false);
        tmp.initialize(nrp.mobility_matrix());
    }

    return tmp.m;
}