BASE=/public/software/apps/boost/intel/1.67.0
BOOST_VERSION=
BOOST_INCLUDE = $(BASE)/include
# C_PLATFORM=-static -pthread
# GPP=/usr/bin/g++
GPP=hipcc
C_OPTIONS= -O3  -std=c++11 -g  -Xcompiler -fopenmp # -DNDEBUG # -DDEBUG
BOOST_LIB_VERSION=

include ../../makefile_common_rocm
