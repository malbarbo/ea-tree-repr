# This makefile was write to help run the experiments but it does not "works"
# like a traditional makefile, that is, it does not list the inputs for the
# targets. So, when an input changes, the target must be manually erased and
# then regenerated with the appropriated make command.

BASE := ../target/release
COPY_VS_MUT := $(shell seq 1000 1000 75000)
SUBTREE_FIRST := $(shell seq 20 20 1000)
CHANGE_PRED := $(shell seq 1000 1000 50000)
CHANGE_PRED_FOREST := ${CHANGE_PRED}
CHANGE_PRED_LARGE := $(shell seq 6000 6000 300000)
CHANGE_ANY := ${CHANGE_PRED}
CHANGE_ANY_FOREST := ${CHANGE_PRED}
CHANGE_ANY_LARGE := ${CHANGE_PRED_LARGE}
CHANGE_TIMES := 30
CHANGE_PARAMS := -k 1 --iters 100000

all: \
	results/nddr-subtree-len-1.pdf \
	results/nddr-subtree-len-2.pdf \
	results/nddr-subtree-len-first-1.pdf \
	results/time-change-any.pdf \
	results/time-change-any-forest.pdf \
	results/time-change-pred.pdf \
	results/time-change-pred-forest.pdf \
	results/time-change-any-large.pdf \
	results/time-change-pred-large.pdf \
	results/time-diameter-change-any.pdf \
	results/time-diameter-change-pred.pdf \
	results/time-copy-vs-mut-pred.pdf \
	results/time-copy-vs-mut-pred2.pdf


#######################
# Cases

cases: cases/beasley cases/carrabs-mbv cases/fhcp cases/leighton cases/tsplib-hcp

cases/beasley:
	mkdir -p cases/beasley && cd cases/beasley; for x in 11 12 13 14 15; do curl -O "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/steind$$x.txt"; done
	cd cases/beasley && for f in *.txt; do ../../mbv2tsp $$f > $$(basename $$f .txt).tsp; done

cases/carrabs-mbv:
	curl -O "http://www.dipmat2.unisa.it/people/carrabs/www/DataSet/MBV_Instances.zip"
	mkdir -p cases/carrabs-mbv && cd cases/carrabs-mbv && unzip -q ../../MBV_Instances.zip MBV_Instances/Spd_Inst_Rid_Final2/* MBV_Instances/Spd_Inst_Rid_Final2_500-1000/* && mv MBV_Instances/Spd_Inst_Rid_Final2/* MBV_Instances/Spd_Inst_Rid_Final2_500-1000/* . && rm -rf MBV_Instances
	cd cases/carrabs-mbv && for f in *.txt; do ../../mbv2tsp $$f > $$(basename $$f .txt).tsp; done
	rm MBV_Instances.zip

cases/fhcp:
	curl -O "http://www.flinders.edu.au/science_engineering/fms/School-CSEM/csem_image_files/FMSL/HCP%20Project%20Website/FHCPCS.7z"
	mkdir -p cases/fhcp && cd cases/fhcp && 7z x ../../FHCPCS.7z
	rm FHCPCS.7z

cases/leighton:
	mkdir -p cases/leighton && cd cases/leighton; for x in 5a 5b 5c 5d 15a 15b 15c 15d 25a 25b 25c 25d; do curl -O "http://archive.dimacs.rutgers.edu/pub/challenge/graph/benchmarks/color/le450_$$x.col"; done
	cd cases/leighton && for f in *.col; do ../../dimacs2tsp $$f > $$(basename $$f .col).tsp; done

cases/tsplib-hcp:
	curl -O "https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/hcp/ALL_hcp.tar"
	mkdir cases/tsplib-hcp && cd cases/tsplib-hcp && tar xf ../../ALL_hcp.tar && for f in *.hcp.gz; do gunzip $$f; done && rm *.gz
	rm ALL_hcp.tar


#######################
# Copy time vs Mutation time

results/time-copy-vs-mut-pred.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-copy-vs-mut-pred \
		"pred change-pred|change-any 100 ${COPY_VS_MUT}" # times sizes
	tectonic plots/time-copy-vs-mut-pred.tex -o results/

results/time-copy-vs-mut-pred2.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-copy-vs-mut-pred2 \
		"pred2 change-pred|change-any 100 ${COPY_VS_MUT}" # times sizes
	tectonic plots/time-copy-vs-mut-pred2.tex -o results/


#######################
# NDDR subtree length

results/nddr-subtree-len-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-1 \
		"adj|balanced change-any 100 150 100000" # n calls times
	tectonic plots/nddr-subtree-len-1.tex -o results/

results/nddr-subtree-len-2.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len \
		results/nddr-subtree-len-2 \
		"adj|balanced change-any 1000 150 100000" # n calls times
	tectonic plots/nddr-subtree-len-2.tex -o results/

results/nddr-subtree-len-first-1.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/nddr-subtree-len-first \
		results/nddr-subtree-len-first-1 \
		"adj|balanced change-any 10000 ${SUBTREE_FIRST}" # times sizes
	tectonic plots/nddr-subtree-len-first-1.tex -o results/


#######################
# Mutation time vs tree size with random diameter

# pred

results/time-change-pred.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-pred \
		"euler-tour|nddr-balanced|pred|pred2 change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_PRED}" # times sizes
	tectonic plots/time-change-pred.tex -o results/

results/time-change-pred-forest.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-pred-forest \
		"--forest euler-tour|nddr-adj|pred|pred2 change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_PRED_FOREST}" # times sizes
	tectonic plots/time-change-pred-forest.tex -o results/

results/time-change-pred-large.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-pred-large \
		"euler-tour|nddr-balanced|pred|pred2 change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_PRED_LARGE}" # times sizes
	tectonic plots/time-change-pred-large.tex -o results/

# any

results/time-change-any.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-any \
		"euler-tour|nddr-balanced|pred change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_ANY}" # times sizes
	tectonic plots/time-change-any.tex -o results/

results/time-change-any-forest.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-any-forest \
		"--forest euler-tour|nddr-adj|pred change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_ANY_FOREST}" # times sizes
	tectonic plots/time-change-any-forest.tex -o results/

results/time-change-any-large.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time \
		results/time-change-any-large \
		"euler-tour|nddr-balanced|pred change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_ANY_LARGE}" # times sizes
	tectonic plots/time-change-any-large.tex -o results/


#######################
# Mutation time vs tree diameter

results/time-diameter-change-pred.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time-diameter \
		results/time-diameter-change-pred \
		"euler-tour|nddr-balanced|pred|pred2 change-pred 5000 50 100" # n samples times
	tectonic plots/time-diameter-change-pred.tex -o results/

results/time-diameter-change-any.pdf:
	mkdir -p results
	./run-combinations \
		${BASE}/time-diameter \
		results/time-diameter-change-any \
		"euler-tour|nddr-balanced|pred change-any 5000 50 100" # n samples times
	tectonic plots/time-diameter-change-any.tex -o results/
