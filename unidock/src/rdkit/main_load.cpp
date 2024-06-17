#include <iostream>
#include <fstream>
#include <vector>
#include <RDGeneral/Invariant.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/AtomIterators.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <Geometry/Transform3D.h>
#include <Geometry/point.h>
#include <Eigen/Dense>
#include <GraphMol/RDKitBase.h>

using namespace RDKit;

// Helper function to calculate dihedral angle
double calcDihedralAngle(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3) {
    Eigen::Vector3d b1 = p1 - p0;
    Eigen::Vector3d b2 = p2 - p1;
    Eigen::Vector3d b3 = p3 - p2;

    Eigen::Vector3d n1 = b1.cross(b2).normalized();
    Eigen::Vector3d n2 = b2.cross(b3).normalized();

    double angle = std::atan2(b2.normalized().dot(n1.cross(n2)), n1.dot(n2));
    return angle;
}

// Helper function to create rotation matrix
Eigen::Matrix3d get_rotation_matrix(double angle, const Eigen::Vector3d& axis) {
    Eigen::AngleAxisd rotation(angle, axis);
    return rotation.toRotationMatrix();
}

void print_positions(const Conformer& conformer) {
    std::cout << "Atom positions:" << std::endl;
    for (unsigned int i = 0; i < conformer.getNumAtoms(); ++i) {
        const RDGeom::Point3D& pos = conformer.getAtomPos(i);
        std::cout << "Atom " << i << ": (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
    }
}

void rotate_torsion_angle(ROMol& mol, const std::vector<int>& torsion_atom_idx_list, int root_atom_idx, double torsion_angle) {
    if (torsion_atom_idx_list.size() != 4) {
        std::cerr << "Error: torsion_atom_idx_list must contain exactly 4 indices." << std::endl;
        return;
    }

    int torsion_atom_idx_0 = torsion_atom_idx_list[0];
    int torsion_atom_idx_1 = torsion_atom_idx_list[1];
    int torsion_atom_idx_2 = torsion_atom_idx_list[2];
    int torsion_atom_idx_3 = torsion_atom_idx_list[3];

    std::cout << "Torsion atom indices: " << torsion_atom_idx_0 << ", " << torsion_atom_idx_1 << ", "
              << torsion_atom_idx_2 << ", " << torsion_atom_idx_3 << std::endl;

    std::pair<int, int> target_rotatable_bond_info = {torsion_atom_idx_1, torsion_atom_idx_2};

    int bond_idx = -1;
    for (const auto& bond : mol.bonds()) {
        std::pair<int, int> bond_info = {bond->getBeginAtomIdx(), bond->getEndAtomIdx()};
        std::pair<int, int> bond_info_reversed = {bond->getEndAtomIdx(), bond->getBeginAtomIdx()};
        std::cout << "Checking bond: " << bond_info.first << "-" << bond_info.second << std::endl;
        if (bond_info == target_rotatable_bond_info || bond_info_reversed == target_rotatable_bond_info) {
            bond_idx = bond->getIdx();
            break;
        }
    }

    if (bond_idx == -1) {
        std::cerr << "Error: Rotatable bond not found." << std::endl;
        return;
    }

    std::cout << "Rotatable bond index: " << bond_idx << std::endl;

    // Fragment molecule on the bond
    RWMol* molCopy = new RWMol(mol);
    molCopy->removeBond(torsion_atom_idx_1, torsion_atom_idx_2);
    std::vector<boost::shared_ptr<ROMol>> fragments = MolOps::getMolFrags(*molCopy, true, nullptr, nullptr, false);

    if (fragments.size() < 2) {
        std::cerr << "Error: Molecule fragmentation failed." << std::endl;
        delete molCopy;
        return;
    }

    std::cout << "Fragmentation successful. Number of fragments: " << fragments.size() << std::endl;

    ROMol* static_fragment = nullptr;
    ROMol* mobile_fragment = nullptr;

    for (const auto& fragment : fragments) {
        for (const auto& atom : fragment->atoms()) {
            if (atom->getIdx() == root_atom_idx) {
                static_fragment = fragment.get();
                break;
            }
        }
        if (static_fragment) {
            break;
        }
    }

    if (!static_fragment) {
        std::cerr << "Error: Root atom not found in any fragment." << std::endl;
        static_fragment = fragments[1].get();
        mobile_fragment = fragments[0].get();
    } else {
        for (const auto& fragment : fragments) {
            if (fragment.get() != static_fragment) {
                mobile_fragment = fragment.get();
                break;
            }
        }
    }

    if (!static_fragment || !mobile_fragment) {
        std::cerr << "Error: Failed to identify static and mobile fragments." << std::endl;
        delete molCopy;
        return;
    }

    std::cout << "Static fragment atom count: " << static_fragment->getNumAtoms() << std::endl;
    std::cout << "Mobile fragment atom count: " << mobile_fragment->getNumAtoms() << std::endl;

    // Reconstruct the molecule from fragments to avoid aromaticity issues
    RWMol reconstructed_mol(*static_fragment);
    MolOps::addHs(reconstructed_mol);
    MolOps::sanitizeMol(reconstructed_mol);

    // Get atom indices of the fragments
    std::vector<int> static_atom_idx_list(static_fragment->getNumAtoms());
    for (const auto& atom : static_fragment->atoms()) {
        static_atom_idx_list[atom->getIdx()] = atom->getIdx();
    }

    std::vector<int> mobile_atom_idx_list(mobile_fragment->getNumAtoms());
    for (const auto& atom : mobile_fragment->atoms()) {
        mobile_atom_idx_list[atom->getIdx()] = atom->getIdx();
    }

    Conformer& conformer = mol.getConformer();
    print_positions(conformer);

    std::vector<RDGeom::Point3D> positions(mol.getNumAtoms());
    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
        positions[i] = conformer.getAtomPos(i);
    }

    Eigen::Vector3d torsion_atom_position_0(positions[torsion_atom_idx_0].x, positions[torsion_atom_idx_0].y, positions[torsion_atom_idx_0].z);
    Eigen::Vector3d torsion_atom_position_1(positions[torsion_atom_idx_1].x, positions[torsion_atom_idx_1].y, positions[torsion_atom_idx_1].z);
    Eigen::Vector3d torsion_atom_position_2(positions[torsion_atom_idx_2].x, positions[torsion_atom_idx_2].y, positions[torsion_atom_idx_2].z);
    Eigen::Vector3d torsion_atom_position_3(positions[torsion_atom_idx_3].x, positions[torsion_atom_idx_3].y, positions[torsion_atom_idx_3].z);

    double target_torsion_value = calcDihedralAngle(torsion_atom_position_0, torsion_atom_position_1, torsion_atom_position_2, torsion_atom_position_3);

    std::cout << "Target torsion value (degrees): " << target_torsion_value * 180.0 / M_PI << std::endl;

    std::vector<Eigen::Vector3d> static_positions(static_atom_idx_list.size());
    for (size_t i = 0; i < static_atom_idx_list.size(); ++i) {
        static_positions[i] = Eigen::Vector3d(positions[static_atom_idx_list[i]].x, positions[static_atom_idx_list[i]].y, positions[static_atom_idx_list[i]].z);
    }

    std::vector<Eigen::Vector3d> mobile_positions(mobile_atom_idx_list.size());
    for (size_t i = 0; i < mobile_atom_idx_list.size(); ++i) {
        mobile_positions[i] = Eigen::Vector3d(positions[mobile_atom_idx_list[i]].x, positions[mobile_atom_idx_list[i]].y, positions[mobile_atom_idx_list[i]].z);
    }

    Eigen::Vector3d torsion_bond_position_0, torsion_bond_position_1;
    if (std::find(static_atom_idx_list.begin(), static_atom_idx_list.end(), target_rotatable_bond_info.first) != static_atom_idx_list.end()) {
        torsion_bond_position_0 = Eigen::Vector3d(positions[target_rotatable_bond_info.first].x, positions[target_rotatable_bond_info.first].y, positions[target_rotatable_bond_info.first].z);
        torsion_bond_position_1 = Eigen::Vector3d(positions[target_rotatable_bond_info.second].x, positions[target_rotatable_bond_info.second].y, positions[target_rotatable_bond_info.second].z);
    } else {
        torsion_bond_position_0 = Eigen::Vector3d(positions[target_rotatable_bond_info.second].x, positions[target_rotatable_bond_info.second].y, positions[target_rotatable_bond_info.second].z);
        torsion_bond_position_1 = Eigen::Vector3d(positions[target_rotatable_bond_info.first].x, positions[target_rotatable_bond_info.first].y, positions[target_rotatable_bond_info.first].z);
    }

    Eigen::Vector3d dihedral_rotate_axis = torsion_bond_position_1 - torsion_bond_position_0;
    Eigen::Vector3d unit_dihedral_rotate_axis = dihedral_rotate_axis.normalized();

    double delta_torsion_angle = torsion_angle - target_torsion_value;
    delta_torsion_angle = M_PI / 180.0 * delta_torsion_angle;

    Eigen::Matrix3d rotation_matrix = get_rotation_matrix(delta_torsion_angle, unit_dihedral_rotate_axis);
    Eigen::Vector3d translation = torsion_bond_position_0 - rotation_matrix * torsion_bond_position_0;

    for (auto& pos : mobile_positions) {
        pos = rotation_matrix * pos + translation;
    }

    for (size_t i = 0; i < mobile_atom_idx_list.size(); ++i) {
        positions[mobile_atom_idx_list[i]] = RDGeom::Point3D(mobile_positions[i].x(), mobile_positions[i].y(), mobile_positions[i].z());
    }

    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
        conformer.setAtomPos(i, positions[i]);
    }

    print_positions(conformer);

    delete molCopy; // Ensure the temporary molecule is deleted
}

int main() {
    // Example usage
    std::string input_sdf = "input_molecule.sdf";
    std::string output_original_sdf = "original_molecule.sdf";
    std::string output_modified_sdf = "modified_molecule.sdf";

    // Read the molecule from the input SDF file
    SDMolSupplier supplier(input_sdf);
    ROMol* mol = supplier.next();
    if (!mol) {
        std::cerr << "Error: Failed to read molecule from SDF file." << std::endl;
        return 1;
    }

    // Ensure the molecule has a conformer
    if (!mol->getNumConformers()) {
        Conformer* conformer = new Conformer(mol->getNumAtoms());
        for (unsigned int i = 0; i < mol->getNumAtoms(); ++i) {
            // Set arbitrary initial positions for atoms
            conformer->setAtomPos(i, RDGeom::Point3D(i * 1.5, 0.0, 0.0));
        }
        mol->addConformer(conformer, true);
    }

    // Write the original molecule to an SDF file
    SDWriter original_writer(output_original_sdf);
    original_writer.write(*mol);
    original_writer.close();

    // Example torsion indices from <torsionInfo>
    std::vector<int> torsion_atom_idx_list = {6, 7, 8, 9};  // Replace with desired torsion
    int root_atom_idx = 6;
    double torsion_angle = 90.0;

    rotate_torsion_angle(*mol, torsion_atom_idx_list, root_atom_idx, torsion_angle);

    std::string new_smiles = MolToSmiles(*mol);
    std::cout << "New SMILES: " << new_smiles << std::endl;

    // Write the modified molecule to an SDF file
    SDWriter modified_writer(output_modified_sdf);
    modified_writer.write(*mol);
    modified_writer.close();

    delete mol;
    return 0;
}
