LIBS = ../build/dcu/release/ad4cache.o ../build/dcu/release/cache.o ../build/dcu/release/non_cache.o ../build/dcu/release/conf_independent.o ../build/dcu/release/coords.o ../build/dcu/release/grid.o ../build/dcu/release/szv_grid.o ../build/dcu/release/model.o ../build/dcu/release/monte_carlo.o ../build/dcu/release/mutate.o ../build/dcu/release/parallel_mc.o ../build/dcu/release/parse_pdbqt.o ../build/dcu/release/quasi_newton.o ../build/dcu/release/quaternion.o ../build/dcu/release/random.o ../build/dcu/release/utils.o ../build/dcu/release/vina.o ../build/dcu/release/precalculate.o
LIB_FLAG = -l boost_system -l boost_thread -l boost_serialization -l boost_filesystem -l boost_program_options -lgtest -lgtest_main
C_INCLUDE_FLAG = -I /usr/local/include -L/usr/local/lib -I../src/lib -I../src/rocm -I /public/software/apps/boost/intel/1.67.0/include  -L.
C_FLAG = -O3  -std=c++11 -g -lineinfo -Xcompiler -fopenmp   -DVERSION=\"ef540d3-mod\"
CC = hipcc

test: test_precalculate test_monte_carlo

test_precalculate: test_precalculate.cc 
	$(CC) $(LIBOBJ) $(C_FLAG) $(C_INCLUDE_FLAG) -o $@ $< $(LIBS) $(LIB_FLAG)

test_monte_carlo: test_monte_carlo.cc 
	$(CC) $(LIBOBJ) $(C_FLAG) $(C_INCLUDE_FLAG) -o $@ $< $(LIBS) $(LIB_FLAG)

clean:
	rm -f test_precalculate test_monte_carlo

dependency:
	cd ../build/dcu/release; make -j
