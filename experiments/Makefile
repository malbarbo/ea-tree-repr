# This makefile was write to help run the experiments but it does not "works"
# like an traditional makefile, that is, it does not list the inputs for the
# targets. So, when an input changes, the target must be manually erased and
# then regenerated with the appropriated make command.

BASE := ../target/release
COPY_VS_MUT := $(shell seq 1000 1000 65000)
SUBTREE_FIRST := $(shell seq 10 10 1000)

all: \
	results/time-copy-vs-mut.pdf \
	results/time-copy-vs-mut-2.pdf \
	results/nddr-subtree-len-1.pdf \
	results/nddr-subtree-len-2.pdf \
	results/nddr-subtree-len-first-1.pdf


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


#######################
# Time NDDR subtree

results/nddr-subtree-len-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-1 \
		"adj|balanced change-any 100 150 10000" # n calls times
	tectonic plots/nddr-subtree-len-1.tex -o results/

results/nddr-subtree-len-2.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-2 \
		"adj|balanced change-any 1000 500 10000" # n calls times
	tectonic plots/nddr-subtree-len-2.tex -o results/

results/nddr-subtree-len-first-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len-first \
		results/nddr-subtree-len-first-1 \
		"adj|balanced change-any 10000 ${SUBTREE_FIRST}" # times sizes
	tectonic plots/nddr-subtree-len-first-1.tex -o results/
