cmake_minimum_required(VERSION 3.16)
project(
	Vina-GPU
	VERSION 0.1.0
	DESCRIPTION "AutoDock Vina 1.2 with GPU acceleration up to 400x."
	HOMEPAGE_URL "https://github.com/dptech-corp/Vina1.2-GPU"
	LANGUAGES CXX CUDA
)
set(CMAKE_CXX_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(VINA_BIN_NAME vina)

add_compile_definitions(ENABLE_CUDA)
add_compile_options(-O2 -g)
add_compile_definitions(VERSION="v${PROJECT_VERSION}")


include_directories(src/lib src/cuda)
add_executable(${VINA_BIN_NAME} src/main/main)
add_executable(split src/split/split)

find_package(Boost REQUIRED
	COMPONENTS system thread serialization filesystem program_options)
target_link_libraries(${VINA_BIN_NAME} Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options)
target_link_libraries(split Boost::system Boost::thread Boost::serialization Boost::filesystem Boost::program_options)
find_package(OpenMP REQUIRED)
target_link_libraries(${VINA_BIN_NAME} OpenMP::OpenMP_CXX)

# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/lib)
add_library(lib OBJECT
	src/lib/ad4cache src/lib/cache src/lib/non_cache src/lib/conf_independent src/lib/coords src/lib/grid src/lib/szv_grid src/lib/model src/lib/mutate src/lib/parallel_mc src/lib/parse_pdbqt src/lib/quasi_newton src/lib/quaternion src/lib/random src/lib/utils src/lib/vina src/lib/precalculate)
	# src/lib/monte_carlo
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/cuda)
add_library(cuda OBJECT src/cuda/monte_carlo.cu src/cuda/precalculate.cu)
target_link_libraries(${VINA_BIN_NAME} cuda lib)
target_include_directories(${VINA_BIN_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # For detecting CUDA memory size
# set_property(TARGET ${VINA_BIN_NAME} PROPERTY CUDA_ARCHITECTURES
# 	# 60 # P100
# 	70 # V100
# 	# 75 # T4
# 	# 80 # A100
# )