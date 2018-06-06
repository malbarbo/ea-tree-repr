# This makefile was write to help run the experiments but it does not "works"
# like a traditional makefile, that is, it does not list the inputs for the
# targets. So, when an input changes, the target must be manually erased and
# then regenerated with the appropriated make command.

BASE := ../target/release
COPY_VS_MUT := $(shell seq 1000 1000 75000)
SUBTREE_FIRST := $(shell seq 10 10 1000)
CHANGE_PRED := $(shell seq 500 500 50000)
CHANGE_PRED_FOREST := ${CHANGE_PRED}
CHANGE_ANY := ${CHANGE_PRED}
CHANGE_ANY_FOREST := ${CHANGE_PRED}

all: \
	results/time-copy-vs-mut.pdf \
	results/time-copy-vs-mut-2.pdf \
	results/nddr-subtree-len-1.pdf \
	results/nddr-subtree-len-2.pdf \
	results/nddr-subtree-len-first-1.pdf \
	results/time-change-any.pdf \
	results/time-change-any-forest.pdf \
	results/time-change-pred.pdf \
	results/time-change-pred-forest.pdf \
	results/time-diameter-change-any.pdf \
	results/time-diameter-change-pred.pdf


cases/fhcp:
	curl -O "http://www.flinders.edu.au/science_engineering/fms/School-CSEM/csem_image_files/FMSL/HCP%20Project%20Website/FHCPCS.7z"
	mkdir -p cases/fhcp && cd cases/fhcp && 7z x ../../FHCPCS.7z
	rm FHCPCS.7z

#######################
# Copy time vs Mutation time

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
# NDDR subtree length

results/nddr-subtree-len-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-1 \
		"adj|balanced|edge change-any 100 150 100000" # n calls times
	tectonic plots/nddr-subtree-len-1.tex -o results/

results/nddr-subtree-len-2.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-2 \
		"adj|balanced|edge change-any 1000 150 100000" # n calls times
	tectonic plots/nddr-subtree-len-2.tex -o results/

results/nddr-subtree-len-first-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len-first \
		results/nddr-subtree-len-first-1 \
		"adj|balanced|edge change-any 10000 ${SUBTREE_FIRST}" # times sizes
	tectonic plots/nddr-subtree-len-first-1.tex -o results/


#######################
# Mutation time vs tree size with random diameter

results/time-change-pred.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-pred \
		"euler-tour|nddr-balanced|pred|pred2 change-pred 100 ${CHANGE_PRED}" # times sizes
	tectonic plots/time-change-pred.tex -o results/

results/time-change-pred-forest.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-pred \
		"--forest euler-tour|nddr-adj|nddr-edge|pred|pred2 change-pred 100 ${CHANGE_PRED_FOREST}" # times sizes
	tectonic plots/time-change-pred-forest.tex -o results/

results/time-change-any.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-any \
		"euler-tour|nddr-balanced|pred change-any 100 ${CHANGE_ANY}" # times sizes
	tectonic plots/time-change-any.tex -o results/

results/time-change-any-forest.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-any \
		"--forest euler-tour|nddr-adj|nddr-edge|pred change-any 100 ${CHANGE_ANY_FOREST}" # times sizes
	tectonic plots/time-change-any-forest.tex -o results/


#######################
# Mutation time vs tree diameter

results/time-diameter-change-pred.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time-diameter \
		results/time-diameter-change-pred \
		"euler-tour|nddr-balanced|pred|pred2 change-pred 5000 100 100" # n samples times
	tectonic plots/time-diameter-change-pred.tex -o results/

results/time-diameter-change-any.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time-diameter \
		results/time-diameter-change-any \
		"euler-tour|nddr-balanced|pred change-any 5000 100 100" # n samples times
	tectonic plots/time-diameter-change-any.tex -o results/
