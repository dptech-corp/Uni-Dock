cmake_minimum_required(VERSION 3.16)
project(
	Uni-Dock
	VERSION 0.1.0
	DESCRIPTION "GPU-Accelerated Docking with 1000-fold speedup"
	HOMEPAGE_URL "https://github.com/dptech-corp/Uni-Dock"
	LANGUAGES CXX CUDA
)
set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
Option(Boost_USE_STATIC_LIBS "Whether to use static libs for boost" OFF)

include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
	set(CMAKE_CUDA_ARCHITECTURES
		60 # P100
		70 # V100
		# 75 # T4
		80 # A100
		# 89 # RTX 40
		90 # H100
	)
endif()

# Add fast math
option(BUILD_FAST_MATH "Build in fast math mode" ON)
if(BUILD_FAST_MATH)
	set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
	message("CMAKE_CUDA_FLAGS_RELEASE updated for fast_math: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(VINA_BIN_NAME unidock)
add_compile_definitions(ENABLE_CUDA)
add_compile_definitions(VERSION="v${PROJECT_VERSION}")

find_package(OpenMP REQUIRED) # OpenMP only required in main.cpp
find_package(Boost 1.72 REQUIRED
	COMPONENTS system thread serialization filesystem program_options timer)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(src/lib src/cuda)
add_executable(${VINA_BIN_NAME} src/main/main.cpp)
add_executable(split src/split/split.cpp)

target_link_libraries(${VINA_BIN_NAME} Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options Boost::timer)
target_link_libraries(split Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options Boost::timer)
target_link_libraries(${VINA_BIN_NAME} OpenMP::OpenMP_CXX)

# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/lib)
add_library(lib OBJECT
	src/lib/ad4cache.cpp src/lib/cache.cpp src/lib/non_cache.cpp src/lib/conf_independent.cpp src/lib/coords.cpp src/lib/grid.cpp src/lib/szv_grid.cpp src/lib/model.cpp src/lib/mutate.cpp src/lib/parallel_mc.cpp src/lib/parse_pdbqt.cpp src/lib/quasi_newton.cpp src/lib/quaternion.cpp src/lib/random.cpp src/lib/utils.cpp src/lib/vina.cpp src/lib/precalculate.h)
	# src/lib/monte_carlo
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/cuda)
add_library(cuda OBJECT src/cuda/monte_carlo.cu src/cuda/precalculate.cu)
target_link_libraries(${VINA_BIN_NAME} cuda lib)
target_include_directories(${VINA_BIN_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # For detecting CUDA memory size
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${VINA_BIN_NAME} TYPE BIN)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.cu)
set(CLANG_FORMAT clang-format)


add_custom_target(
  clang-format
  COMMAND ${CLANG_FORMAT} -i ${sources}
  COMMENT "Running clang-format"
  VERBATIM
)
