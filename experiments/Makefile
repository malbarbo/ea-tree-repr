# This makefile was write to help run the experiments but it does not "works"
# like a traditional makefile, that is, it does not list the inputs for the
# targets. So, when an input changes, the target must be manually erased and
# then regenerated with the appropriated make command.

RUNNER := /usr/bin/parallel --jobs 1 --linebuffer -t
TIME := /usr/bin/time -f '\nTotal time %E'
BASE := ../target/release
COPY_VS_MUT := $(shell seq 1000 1000 75000)
SUBTREE_FIRST := $(shell seq 20 20 1000)
CHANGE_SIZES := $(shell seq 1000 1000 50000)
CHANGE_FOREST_SIZES := ${CHANGE_PRED}
CHANGE_LARGE_SIZES := $(shell seq 6000 6000 300000)
CHANGE_TIMES := 30
CHANGE_PARAMS := -k 1 --iters 100000

plots: \
	results/nddr-subtree-len-1.pdf \
	results/nddr-subtree-len-2.pdf \
	results/nddr-subtree-len-first-1.pdf \
	results/time.pdf \
	results/time-forest.pdf \
	results/time-large.pdf \
	results/time-diameter.pdf \
	results/omt.pdf \
	results/omt-change-any.pdf \
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
	mkdir -p results/time-copy-vs-mut-pred
	# ds op times sizes
	${RUNNER} "${TIME} ${BASE}/time pred {} 100 ${COPY_VS_MUT} > results/time-copy-vs-mut-pred/{}" \
		::: change-pred change-any
	tectonic plots/time-copy-vs-mut-pred.tex -o results/

results/time-copy-vs-mut-pred2.pdf:
	mkdir -p results/time-copy-vs-mut-pred2
	# ds op times sizes
	${RUNNER} "${TIME} ${BASE}/time pred2 {} 100 ${COPY_VS_MUT} > results/time-copy-vs-mut-pred2/{}" \
		::: change-pred change-any
	tectonic plots/time-copy-vs-mut-pred2.tex -o results/


#######################
# NDDR subtree length

results/nddr-subtree-len-1.pdf:
	mkdir -p results/nddr-subtree-len-1
	# ds op n calls times
	${RUNNER} "${TIME} ${BASE}/nddr-subtree-len {} change-any 100 150 100000 > results/nddr-subtree-len-1/{}" \
		::: adj free
	tectonic plots/nddr-subtree-len-1.tex -o results/

results/nddr-subtree-len-2.pdf:
	mkdir -p results/nddr-subtree-len-2
	# ds op n calls times
	${RUNNER} "${TIME} ${BASE}/nddr-subtree-len {} change-any 1000 150 100000 > results/nddr-subtree-len-2/{}" \
		::: adj free
	tectonic plots/nddr-subtree-len-2.tex -o results/

results/nddr-subtree-len-first-1.pdf:
	mkdir -p results/nddr-subtree-len-first-1
	# ds op times sizes
	${RUNNER} "${TIME} ${BASE}/nddr-subtree-len-first {} change-any 10000 ${SUBTREE_FIRST} > results/nddr-subtree-len-first-1/{}" \
		::: adj free
	tectonic plots/nddr-subtree-len-first-1.tex -o results/


#######################
# Mutation time vs tree size with random diameter

# time

results/time.pdf: results/time-change-pred results/time-change-any
	tectonic plots/time.tex -o results/

results/time-change-pred:
	mkdir -p results/time-change-pred
	${RUNNER} "${TIME} ${BASE}/time {} change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_PRED} > results/time-change-pred/{}" \
		::: euler-tour nddr-free pred pred2

results/time-change-any:
	mkdir -p results/time-change-any
	${RUNNER} "${TIME} ${BASE}/time {} change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_ANY} > results/time-change-any/{}" \
		::: euler-tour nddr-free pred


# time forest
results/time-forest.pdf: results/time-change-pred-forest results/time-change-any-forest
	tectonic plots/time-forest.tex -o results/

results/time-change-pred-forest:
	mkdir -p results/time-change-pred-forest
	${RUNNER} "${TIME} ${BASE}/time --forest {} change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_FOREST_SIZES} > results/time-change-pred-forest/{}" \
		::: euler-tour nddr-adj pred pred2

results/time-change-any-forest:
	mkdir -p results/time-change-any-forest
	${RUNNER} "${TIME} ${BASE}/time --forest {} change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_FOREST_SIZES} > results/time-change-any-forest/{}" \
		::: euler-tour nddr-adj pred


# time large

results/time-large.pdf: results/time-change-pred-large results/time-change-any-large
	tectonic plots/time-large.tex -o results/

results/time-change-pred-large:
	mkdir -p results/time-change-pred-large
	${RUNNER} "${TIME} ${BASE}/time {} change-pred ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_LARGE_SIZES} > results/time-change-pred-large/{}" \
		::: euler-tour nddr-free pred pred2

results/time-change-any-large:
	mkdir -p results/time-change-any-large
	${RUNNER} "${TIME} ${BASE}/time {} change-any ${CHANGE_PARAMS} ${CHANGE_TIMES} ${CHANGE_LARGE_SIZES} > results/time-change-any-large/{}" \
		::: euler-tour nddr-free pred


#######################
# Mutation time vs tree diameter

results/time-diameter.pdf: results/time-diameter-change-pred results/time-diameter-change-any
	tectonic plots/time-diameter.tex -o results/

results/time-diameter-change-pred:
	mkdir -p results/time-diameter-change-pred
	# ds op n samples times
	${RUNNER} "${TIME} ${BASE}/time-diameter {} change-pred 5000 50 100 > results/time-diameter-change-pred/{}" \
		::: euler-tour nddr-free pred pred2

results/time-diameter-change-any:
	mkdir -p results/time-diameter-change-any
	# ds op n samples times
	${RUNNER} "${TIME} ${BASE}/time-diameter {} change-any 5000 50 100 > results/time-diameter-change-any/{}" \
		::: euler-tour nddr-free pred


#######################
# One Tree Max problem best solution evolution

results/omt.pdf: results/omt-change-pred results/omt-change-any
	tectonic plots/omt.tex -o results/

results/omt-change-pred:
	mkdir -p results/omt-change-pred
	# ds op times n
	${RUNNER} "${TIME} ${BASE}/omt --quiet --max_num_iters 1000000 {} change-pred 1000 5000 > results/omt-change-pred/{}" \
		::: euler-tour nddr-adj pred pred2

results/omt-change-any:
	mkdir -p results/omt-change-any
	# ds op times n
	${RUNNER} "${TIME} ${BASE}/omt --quiet --max_num_iters 1000000 {} change-any 1000 5000 > results/omt-change-any/{}" \
		::: euler-tour nddr-adj pred pred2
	tectonic plots/omt-change-any.tex -o results/
