LIBS = ../build/linux/release/ad4cache.o ../build/linux/release/cache.o ../build/linux/release/non_cache.o ../build/linux/release/conf_independent.o ../build/linux/release/coords.o ../build/linux/release/grid.o ../build/linux/release/szv_grid.o ../build/linux/release/model.o ../build/linux/release/monte_carlo.o ../build/linux/release/mutate.o ../build/linux/release/parallel_mc.o ../build/linux/release/parse_pdbqt.o ../build/linux/release/quasi_newton.o ../build/linux/release/quaternion.o ../build/linux/release/random.o ../build/linux/release/utils.o ../build/linux/release/vina.o ../build/linux/release/precalculate.o
LIB_FLAG = -l boost_system -l boost_thread -l boost_serialization -l boost_filesystem -l boost_program_options -lgtest -lgtest_main
C_INCLUDE_FLAG = -I /usr/local/include -L/usr/local/lib -I../src/lib -I../src/cuda -L.
C_FLAG = -O3  -std=c++11 -g -lineinfo -Xcompiler -fopenmp   -DVERSION=\"ef540d3-mod\"
CC = nvcc

test: test_precalculate test_monte_carlo

test_precalculate: test_precalculate.cc 
	$(CC) $(LIBOBJ) $(C_FLAG) $(C_INCLUDE_FLAG) -o $@ $< $(LIBS) $(LIB_FLAG)

test_monte_carlo: test_monte_carlo.cc 
	$(CC) $(LIBOBJ) $(C_FLAG) $(C_INCLUDE_FLAG) -o $@ $< $(LIBS) $(LIB_FLAG)

clean:
	rm -f test_precalculate test_monte_carlo

dependency:
	cd ../build/linux/release; make -j
