.PHONY: core-audit regression reproduction-dag
PYTHON ?= python3

# NOTE: Only `core-audit` and `regression` are operational in the anonymous artifact.
# The `reproduction-dag` target references scripts from sibling revision directories
# (../revision_20260728_round3, ../revision_20260728_round4) that are not included
# because they depend on intermediate data files generated during the audit that
# exceed the artifact size limit. The saved hypotheses in data/cells/ and the
# sufficient statistics in results/ allow full recomputation of all reported
# numbers via `core-audit`. The DAG scripts will be published with the camera-ready
# version at the public repository.

core-audit: regression
	$(PYTHON) scripts/reviewer_round2_diagnostics.py --cells data/cells --output results/bleu_diagnostics

regression:
	$(PYTHON) -c "import tests.test_checkpoint_gt_alignment as t; t.test_every_transfer_cell_uses_its_evaluators_local_gt(); print('checkpoint-GT alignment PASS')"

# Reproduction DAG: stages A-N. See README.md section 2 for full table.
# Each stage is idempotent (skips if output JSON exists with matching SHA-256).
reproduction-dag:
	@echo "[dag] Stage A: build exposure-symmetric systems"
	$(PYTHON) ../revision_20260728_round3/scripts/e2_1_build_exposure_symmetric.py
	@echo "[dag] Stage B: evaluate exposure-symmetric cells (GPU 0)"
	$(PYTHON) ../revision_20260728_round3/scripts/e2_2_eval_exposure.py --gpu 0
	@echo "[dag] Stage C: build SEEN-PURE-MATCHED + RAND640"
	$(PYTHON) ../revision_20260728_round4/scripts/r4_seen_pure_matched.py
	$(PYTHON) ../revision_20260728_round4/scripts/r4_eval_matched.py --gpu 0
	@echo "[dag] Stage D-E: common-support eval + decomposition"
	$(PYTHON) ../revision_20260728_round4/scripts/r5_common_support_eval.py --gpu 0
	$(PYTHON) ../revision_20260728_round4/scripts/r5_common_decomposition_fast.py
	@echo "[dag] Stage F-G: cluster bootstrap + pool randomization"
	$(PYTHON) ../revision_20260728_round4/scripts/r5_cluster_bootstrap.py
	$(PYTHON) ../revision_20260728_round4/scripts/r5_pool_level_randomization.py --gpu 0 --n-pools 20
	@echo "[dag] Stage H-I: cross-fit fold-local + scorer equivalence"
	$(PYTHON) ../revision_20260728_round4/scripts/r5_crossfit_fold_local.py --gpu 0
	$(PYTHON) ../revision_20260728_round4/scripts/r5_official_scorer.py
	@echo "[dag] Stage J-L: BERTScore + BLEURT + BLEURT consistency"
	$(PYTHON) ../revision_20260728_round4/scripts/r4_bertscore.py
	$(PYTHON) ../revision_20260728_round4/scripts/r4_bleurt_gpu.py --gpu 0
	@echo "[dag] Stage M-N: figures + decoding sensitivity"
	$(PYTHON) ../revision_20260728_round4/scripts/r4_plots.py
	$(PYTHON) ../revision_20260728_round4/scripts/r4_decoding_sensitivity.py --gpu 0
	$(PYTHON) ../revision_20260728_round4/scripts/r4_decoding_plot.py
	@echo "[dag] reproduction-dag COMPLETE"

paper:
	$(PYTHON) scripts/build_paper_numbers.py
	@echo "Paper numbers written to results/paper_numbers.json"
	@echo "Compare with main_lre.tex values for consistency check"
