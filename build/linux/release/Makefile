BASE=/opt/lib/packages/boost1_77
BOOST_VERSION=
BOOST_INCLUDE = $(BASE)/include
# C_PLATFORM=-static -pthread
# GPP=/usr/bin/g++
GPP=nvcc
C_OPTIONS= -O3 -std=c++11 -g -lineinfo -Xcompiler -fopenmp #--compiler-options -fsanitize=address  # -DNDEBUG # -DDEBUG
BOOST_LIB_VERSION=

include ../../makefile_common
