#include "../catch_amalgamated.hpp"
#include "parse_pdbqt.h"
#include "atom.h"
// #include "parse_pdbqt.cpp"

TEST_CASE("parse pdbqt rigid", "[parse_pdbqt_rigid]") {
    // 假设你有一个有效的PDBQT文件路径
    std::string valid_pdbqt_receptor_path = "../unitest/ParseTests/def.pdbqt";
    rigid r;
    REQUIRE_NOTHROW(parse_pdbqt_rigid(valid_pdbqt_receptor_path, r));
    REQUIRE(r.atoms.size() == 1613); // 假设你的atom解析函数会添加到atoms向量
    // // 测试正常情况
    // // REQUIRE_NOTHROW(parse_receptor_pdbqt(valid_pdbqt_receptor_path));
    // // atom_type::t atype;
    // // pdbqt_initializer tmp(atype);
    // model m = parse_receptor_pdbqt(valid_pdbqt_receptor_path);
    // std::cout <<"aaaaaaaaaa"<<m.num_atoms() <<std::endl; 
}
