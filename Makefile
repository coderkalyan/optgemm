MKLROOT = /opt/intel/oneapi/mkl/latest
CPPFLAGS = -std=c++17 -O3 -march=native -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
LDFLAGS = -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_gnu_thread.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

mkl: mkl.cpp
	clang++ $(CPPFLAGS) mkl.cpp -o mkl $(LDFLAGS)

zen2: zen2.cpp
	clang++ $(CPPFLAGS) zen2.cpp -o zen2 $(LDFLAGS)

clean:
	rm -f mkl zen2
