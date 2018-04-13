# This makefile was write to help run the experiments but it does not "works"
# like an traditional makefile, that is, it does not list the inputs for the
# targets. So, when an input changes, the target must be manually erased and
# then regenerated with the appropriated make command.

BASE := ../target/release
COPY_VS_MUT := $(shell seq 1000 1000 65000)

all: \
	results/time-copy-vs-mut.pdf \
	results/time-copy-vs-mut-2.pdf


#######################
# Time Copy vs Mutation

results/time-copy-vs-mut.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-copy-vs-mut-pred \
		"pred change-pred|change-any 100 ${COPY_VS_MUT}" # times sizes
	tectonic plots/time-copy-vs-mut.tex -o results/

results/time-copy-vs-mut-2.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-copy-vs-mut-pred2 \
		"pred2 change-pred|change-any 100 ${COPY_VS_MUT}" # times sizes
	tectonic plots/time-copy-vs-mut-2.tex -o results/
