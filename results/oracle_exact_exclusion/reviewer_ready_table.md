# Task 9 v3 reviewer table

White-box winner's-curse/search-budget audit only; not deployable SLP, causal evidence, or general optimization.

## Confirmatory original-selector full-pool family

| evaluator | LL margin [95% CI] | corpus BLEU / local GT | BLEU margin [95% CI] | LL raw/Holm p | BLEU raw/Holm p | success |
|---|---:|---:|---:|---:|---:|:---:|
| matched_101 | -0.0383555 [-0.0595723,-0.0171387] | 0.0812461/0.0857067 | -0.0520448 [-0.138314,0.0352361] | 1.99998e-05/9.9999e-05 | 0.273473/1 | False |
| matched_202 | -0.0327101 [-0.0559206,-0.00949965] | 0.0828347/0.0881348 | -0.0601368 [-0.155033,0.0566648] | 0.000369996/0.00147999 | 0.30187/1 | False |
| matched_303 | -0.043581 [-0.0621882,-0.0249738] | 0.0915642/0.0966986 | -0.0530965 [-0.121174,0.0205861] | 9.9999e-06/6.99993e-05 | 0.164884/0.824418 | False |
| matched_404 | -0.0248843 [-0.0428405,-0.00692817] | 0.0651855/0.0719282 | -0.0937428 [-0.246582,0.0836597] | 0.00187998/0.00563994 | 0.278872/1 | False |
| matched_505 | -0.0684894 [-0.102217,-0.0347613] | 0.0805021/0.090472 | -0.110198 [-0.184524,-0.0369149] | 9.9999e-06/6.99993e-05 | 0.0049995/0.0349965 | False |
| matched_606 | -0.0146572 [-0.0240383,-0.00527611] | 0.0825691/0.080899 | 0.0206448 [-0.0586372,0.0999724] | 0.00237998/0.00563994 | 0.594641/1 | False |
| original | 0.00858309 [-0.110345,0.127511] | 0.138574/0.127771 | 0.0845427 [-0.00434536,0.185797] | 0.890401/0.890401 | 0.0611939/0.367163 | False |

Frozen success rule resolves historical 2/7 to **0/7**; Holm is reported separately for interpretation.

## Random-subset and oracle full-sweep curves (original selector/evaluator)

| budget | random LL margin [CI] | random corpus BLEU margin [CI] | oracle LL margin [CI] | oracle corpus BLEU margin [CI] |
|---:|---:|---:|---:|---:|
| 8 | -6.9726 [-12.245,-1.6999] | -0.82766 [-0.84988,-0.80545] | -8.6815 [-14.884,-2.479] | -0.94346 [-0.96425,-0.92017] |
| 32 | -4.972 [-8.7319,-1.212] | -0.67868 [-0.70803,-0.64934] | -5.2851 [-9.274,-1.2963] | -0.72199 [-0.78206,-0.65935] |
| 128 | -3.0732 [-5.4581,-0.68823] | -0.47792 [-0.51919,-0.43665] | -3.3567 [-5.9285,-0.78485] | -0.46907 [-0.54047,-0.39355] |
| 512 | -1.512 [-2.6788,-0.34526] | -0.28387 [-0.33645,-0.2313] | -2.5393 [-4.729,-0.34959] | -0.321 [-0.40254,-0.23772] |
| 2048 | -0.54868 [-1.0349,-0.062455] | -0.084045 [-0.15119,-0.016896] | -0.85286 [-1.7459,0.040199] | -0.13692 [-0.2226,-0.044579] |
| 3530 | -0.29651 [-0.62325,0.030234] | -0.0058617 [-0.081194,0.06947] | -0.53865 [-1.1715,0.094176] | -0.0090274 [-0.10174,0.090253] |
| 7060 | 0.0085831 [-0.10969,0.12685] | 0.084543 [-0.010534,0.17962] | 0.0085831 [-0.10795,0.12512] | 0.084543 [-0.0043454,0.1858] |

## A/B disjoint candidate-pool replication

A/B intervals are paired over shared targets; they are not independent target/evaluator replications. Full 7x7 transfer matrices and the separately frozen diagonal family are descriptive outside their named confirmatory families.

See `partition_AB_paired.csv`, `random_transfer_curves.csv`, and `oracle_pool_curves.csv`.
