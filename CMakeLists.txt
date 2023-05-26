cmake_minimum_required(VERSION 3.16)
project(
	Vina-GPU
	VERSION 0.1.0
	DESCRIPTION "AutoDock Vina 1.2 with GPU acceleration up to 400x."
	HOMEPAGE_URL "https://github.com/dptech-corp/Vina1.2-GPU"
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
		75 # T4
		80 # A100
		# 89 # RTX 40
		90 # H100
	)
endif()

set(VINA_BIN_NAME vina)
add_compile_definitions(ENABLE_CUDA)
add_compile_definitions(VERSION="v${PROJECT_VERSION}")

find_package(OpenMP REQUIRED) # OpenMP only required in main.cpp
find_package(Boost REQUIRED
	COMPONENTS system thread serialization filesystem program_options)

include_directories(src/lib src/cuda)
add_executable(${VINA_BIN_NAME} src/main/main.cpp)
add_executable(split src/split/split.cpp)

target_link_libraries(${VINA_BIN_NAME} Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options)
target_link_libraries(split Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options)
target_link_libraries(${VINA_BIN_NAME} OpenMP::OpenMP_CXX)

# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/lib)
add_library(lib OBJECT
	src/lib/ad4cache.cpp src/lib/cache.cpp src/lib/non_cache.cpp src/lib/conf_independent.cpp src/lib/coords.cpp src/lib/grid.cpp src/lib/szv_grid.cpp src/lib/model.cpp src/lib/mutate.cpp src/lib/parallel_mc.cpp src/lib/parse_pdbqt.cpp src/lib/quasi_newton.cpp src/lib/quaternion.cpp src/lib/random.cpp src/lib/utils.cpp src/lib/vina.cpp src/lib/precalculate.h)
	# src/lib/monte_carlo
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/cuda)
add_library(cuda OBJECT src/cuda/monte_carlo.cu src/cuda/precalculate.cu)
target_link_libraries(${VINA_BIN_NAME} cuda lib)
target_include_directories(${VINA_BIN_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # For detecting CUDA memory size
