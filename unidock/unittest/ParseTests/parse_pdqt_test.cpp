#include <catch2/catch_test_macros.hpp>
#include "parse_pdbqt.h"
#include "atom.h"


TEST_CASE("parse pdbqt rigid", "[parse_pdbqt_rigid]") {
    // Test pdbqt atom number
    std::string valid_pdbqt_receptor_path = "./test_data/def.pdbqt";
    rigid r;
    REQUIRE_NOTHROW(parse_pdbqt_rigid(valid_pdbqt_receptor_path, r));
    REQUIRE(r.atoms.size() == 1613);

}