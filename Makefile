OUTPUT_DIR = ./
CFLAGS= -g -Wall -O3 -lm -L.

all: ${OUTPUT_DIR}pagerank_mpi

PageRankMPI.o: PageRankMPI.cpp
	mpic++ $(CFLAGS) -c PageRankMPI.cpp

${OUTPUT_DIR}pagerank_mpi: PageRankMPI.o
	mpic++ $(CFLAGS) PageRankMPI.o -o ${OUTPUT_DIR}"pagerank_mpi" -lrt

clean:
	rm *.o
	rm ${OUTPUT_DIR}pagerank_mpi