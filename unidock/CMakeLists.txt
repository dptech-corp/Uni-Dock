cmake_minimum_required(VERSION 3.16)

project(
  Uni-Dock
  VERSION 1.1.1
  DESCRIPTION "GPU-Accelerated Docking with 1000-fold speedup"
  HOMEPAGE_URL "https://github.com/dptech-corp/Uni-Dock"
  LANGUAGES CXX CUDA)
set(VINA_BIN_NAME unidock)

# Make sure the default is a release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# >>>>>> Language Configuration Begin
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
option(Boost_USE_STATIC_LIBS "Whether to use static libs for boost" OFF)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  set(CMAKE_CUDA_ARCHITECTURES
    70 # V100
    # 75 # RTX 20, T4
    80 # A100, A800
    # 86 # RTX 30 89 # RTX 40, L40
    90 # H100
  )
endif()

# Add fast math for release build
option(BUILD_FAST_MATH "Build in fast math mode" ON)

if(BUILD_FAST_MATH)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
  message(
    "CMAKE_CUDA_FLAGS_RELEASE updated for fast_math: ${CMAKE_CUDA_FLAGS_RELEASE}"
  )
endif()

# Add preprocessor definitions
add_compile_definitions(ENABLE_CUDA) # ENABLE_CUDA is required
add_compile_definitions(VERSION="v${PROJECT_VERSION}")

# >>>>>> Language Configuration End

# >>>>>> Dependencies Begin
include(FetchContent)

# OpenMP only required in main.cpp

# >>> OpenMP
find_package(OpenMP REQUIRED)

# >>> CMake
option(FETCH_BOOST "Whether to use FetchContent to include boost" OFF)
set(REQUIRED_BOOST_COMPONENTS
  system thread serialization
  filesystem program_options timer
)

Set(FETCHCONTENT_QUIET FALSE) # Needed to print downloading progress

if(FETCH_BOOST)
  set(BOOST_ENABLE_CMAKE ON)
  set(BOOST_INCLUDE_LIBRARIES ${REQUIRED_BOOST_COMPONENTS})

  # downloading a compressed release speeds up the download
  FetchContent_Declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
  )
  FetchContent_MakeAvailable(Boost)
else()
  find_package(Boost 1.72 REQUIRED COMPONENTS ${REQUIRED_BOOST_COMPONENTS})
  include_directories(${Boost_INCLUDE_DIRS})
endif()

# >>> Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# >>>>>> Dependencies End

# >>>>>> Libraries Definitions Begin
# Find RDKit
find_package(RDKit REQUIRED)
# Find Eigen3
find_package(Eigen3 REQUIRED)

# Group boost dependencies
add_library(boost_deps INTERFACE)
target_link_libraries(
  boost_deps INTERFACE Boost::system Boost::thread Boost::serialization
  Boost::filesystem Boost::program_options Boost::timer)

add_library(
  lib OBJECT
  src/lib/ad4cache.cpp
  src/lib/cache.cpp
  src/lib/non_cache.cpp
  src/lib/conf_independent.cpp
  src/lib/coords.cpp
  src/lib/grid.cpp
  src/lib/szv_grid.cpp
  src/lib/model.cpp
  src/lib/mutate.cpp
  src/lib/parallel_mc.cpp
  src/lib/parse_pdbqt.cpp
  src/lib/quasi_newton.cpp
  src/lib/quaternion.cpp
  src/lib/random.cpp
  src/lib/utils.cpp
  src/lib/vina.cpp
)
target_include_directories(lib PUBLIC src/lib src/cuda)

add_library(cuda OBJECT src/cuda/monte_carlo.cu src/cuda/precalculate.cu)
target_include_directories(cuda PUBLIC src/lib src/cuda)

# >>>>>> Libraries Definitions End

# >>>>>> Binaries Definitions Begin
# >>> Main binary (obb test)
add_executable(obb src/cuda/obb.cu)
# >>> Main binary (obb test)
add_executable(bvh src/bvh/bvh.cpp src/bvh/bvh.hpp)
# >>> Main binary (rdkit rotation)
add_executable(unidock2 src/rdkit/main.cpp )
target_link_libraries(unidock2 PRIVATE
  RDKit::SmilesParse RDKit::MolHash
  Eigen3::Eigen
)
add_executable(bfgs src/rdkit/bfgs.cpp )
target_link_libraries(bfgs PRIVATE
  RDKit::SmilesParse RDKit::MolHash
  Eigen3::Eigen
)
# >>> Main binary (unidock)
add_executable(${VINA_BIN_NAME} src/main/main.cpp)
target_link_libraries(${VINA_BIN_NAME} PRIVATE
  boost_deps cuda lib
  OpenMP::OpenMP_CXX
  RDKit::SmilesParse RDKit::MolHash
)
install(TARGETS ${VINA_BIN_NAME})

# For detecting CUDA memory size
target_include_directories(${VINA_BIN_NAME}
  PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# >>> split
add_executable(split src/split/split.cpp)
target_link_libraries(split PRIVATE boost_deps)
target_include_directories(split PRIVATE src/lib)

# >>>>>> Binaries Definitions End

# >>>>>> Tests Definitions Begin
file(GLOB_RECURSE TEST_SOURCES ${CMAKE_SOURCE_DIR}/unittest/*.cpp)
add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE
  boost_deps lib cuda
  OpenMP::OpenMP_CXX
  Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy test data to build directory
file(
  COPY ${CMAKE_SOURCE_DIR}/unittest/test_data
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# >>>>>> Tests Definitions End

# >>>>>> Clang Format Begin

# Configure clang-format
file(
  GLOB_RECURSE
  sources
  CONFIGURE_DEPENDS
  src/*.cpp
  src/*.h
  src/*.cu
  src/*.cuh)
set(CLANG_FORMAT clang-format)
add_custom_target(
  clang-format
  COMMAND ${CLANG_FORMAT} -i ${sources}
  COMMENT "Running clang-format"
  VERBATIM)

# >>>>>> Clang Format End