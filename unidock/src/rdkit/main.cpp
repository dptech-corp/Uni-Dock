#include <iostream>
#include <fstream>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <Eigen/Dense>
#include <vector>

// 读取SDF文件中的分子
RDKit::ROMol* readMoleculeFromSDF(const std::string& filename) {
    RDKit::SDMolSupplier supplier(filename, true, false);
    if (supplier.atEnd()) {
        std::cerr << "Error reading SDF file." << std::endl;
        return nullptr;
    }
    return supplier.next();
}

// 获取指定原子的坐标
Eigen::Vector3d getAtomCoordinates(const RDKit::ROMol& mol, int atomIdx) {
    const auto& conf = mol.getConformer();
    const auto& pos = conf.getAtomPos(atomIdx);
    return Eigen::Vector3d(pos.x, pos.y, pos.z);
}

// 计算绕指定轴旋转的旋转矩阵
Eigen::Matrix3d computeRotationMatrix(double angle, const Eigen::Vector3d& axis) {
    Eigen::AngleAxisd rotation(angle, axis.normalized());
    return rotation.toRotationMatrix();
}

// 获取需要旋转的原子
std::vector<int> getAtomsToRotate(const RDKit::ROMol& mol, int fixedAtomIdx, int rotateAtomIdx) {
    std::vector<int> atomsToRotate;
    std::vector<bool> visited(mol.getNumAtoms(), false);
    std::vector<int> stack = { rotateAtomIdx };

    while (!stack.empty()) {
        int current = stack.back();
        stack.pop_back();

        if (!visited[current]) {
            visited[current] = true;
            atomsToRotate.push_back(current);

            for (const auto& bond : mol.atomBonds(mol.getAtomWithIdx(current))) {
                int neighbor = bond->getOtherAtomIdx(current);
                if (!visited[neighbor] && neighbor != fixedAtomIdx) {
                    stack.push_back(neighbor);
                }
            }
        }
    }

    return atomsToRotate;
}

// 应用旋转矩阵到指定原子的坐标
void rotateAtoms(RDKit::ROMol& mol, const Eigen::Matrix3d& rotationMatrix, int fixedAtomIdx, int rotateAtomIdx) {
    auto& conf = mol.getConformer();
    Eigen::Vector3d origin = getAtomCoordinates(mol, fixedAtomIdx);
    std::vector<int> atomsToRotate = getAtomsToRotate(mol, fixedAtomIdx, rotateAtomIdx);

    for (int atomIdx : atomsToRotate) {
        Eigen::Vector3d coord = getAtomCoordinates(mol, atomIdx) - origin;
        Eigen::Vector3d rotatedCoord = rotationMatrix * coord + origin;
        conf.setAtomPos(atomIdx, RDGeom::Point3D(rotatedCoord[0], rotatedCoord[1], rotatedCoord[2]));
    }
}

// 保存分子到SDF文件
bool saveMoleculeToSDF(const RDKit::ROMol& mol, const std::string& filename) {
    RDKit::SDWriter writer(filename);
    // if (!writer.good()) {
    //     std::cerr << "Error opening SDF file for writing: " << filename << std::endl;
    //     return false;
    // }
    writer.write(mol);
    writer.flush();
    writer.close();
    return true;
}

// 通用旋转函数
bool rotateMolecule(const std::string& inputSDFFile, const std::string& outputSDFFile, int fixedAtomIdx, int rotateAtomIdx, double angle) {
    RDKit::ROMol* mol = readMoleculeFromSDF(inputSDFFile);
    if (!mol) {
        return false;
    }

    if (fixedAtomIdx >= mol->getNumAtoms() || rotateAtomIdx >= mol->getNumAtoms() || fixedAtomIdx < 0 || rotateAtomIdx < 0) {
        std::cerr << "Invalid atom index" << std::endl;
        delete mol;
        return false;
    }

    Eigen::Vector3d axis = getAtomCoordinates(*mol, rotateAtomIdx) - getAtomCoordinates(*mol, fixedAtomIdx);
    Eigen::Matrix3d rotationMatrix = computeRotationMatrix(angle, axis);

    rotateAtoms(*mol, rotationMatrix, fixedAtomIdx, rotateAtomIdx);

    bool success = saveMoleculeToSDF(*mol, outputSDFFile);

    delete mol;
    return success;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_sdf> <output_sdf> <fixed_atom_idx> <rotate_atom_idx> <angle_in_degrees>" << std::endl;
        return 1;
    }

    std::string inputSDFFile = argv[1];
    std::string outputSDFFile = argv[2];
    int fixedAtomIdx = std::stoi(argv[3]);
    int rotateAtomIdx = std::stoi(argv[4]);
    double angle = std::stod(argv[5]) * M_PI / 180.0; // 转换角度为弧度

    if (rotateMolecule(inputSDFFile, outputSDFFile, fixedAtomIdx, rotateAtomIdx, angle)) {
        std::cout << "Molecule rotated and saved to " << outputSDFFile << std::endl;
    } else {
        std::cerr << "Failed to rotate molecule" << std::endl;
        return 1;
    }

    return 0;
}
