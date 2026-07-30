# Task 9 v3 reviewer table

White-box winner's-curse/search-budget audit only; not deployable SLP, causal evidence, or general optimization.

## Confirmatory original-selector full-pool family

| evaluator | LL margin [95% CI] | corpus BLEU / local GT | BLEU margin [95% CI] | LL raw/Holm p | BLEU raw/Holm p | success |
|---|---:|---:|---:|---:|---:|:---:|
| matched_101 | -0.050332 [-0.0667287,-0.0339354] | 0.0705997/0.0857067 | -0.176264 [-0.266256,-0.0722155] | 9.9999e-06/6.99993e-05 | 0.00129987/0.00259974 | False |
| matched_202 | -0.0742564 [-0.0962628,-0.0522501] | 0.0610012/0.0881348 | -0.307865 [-0.415825,-0.186088] | 9.9999e-06/6.99993e-05 | 9.999e-05/0.00069993 | False |
| matched_303 | -0.0808691 [-0.100716,-0.061022] | 0.0737316/0.0966986 | -0.237511 [-0.319406,-0.146663] | 9.9999e-06/6.99993e-05 | 9.999e-05/0.00069993 | False |
| matched_404 | -0.0769313 [-0.102444,-0.0514185] | 0.0537159/0.0719282 | -0.253201 [-0.380576,-0.108783] | 9.9999e-06/6.99993e-05 | 0.00079992/0.00239976 | False |
| matched_505 | -0.0611026 [-0.0753951,-0.0468101] | 0.0694389/0.090472 | -0.232482 [-0.323214,-0.145733] | 9.9999e-06/6.99993e-05 | 9.999e-05/0.00069993 | False |
| matched_606 | -0.0676956 [-0.0912801,-0.0441112] | 0.0677833/0.080899 | -0.162125 [-0.272039,-0.0493099] | 9.9999e-06/6.99993e-05 | 0.00539946/0.00539946 | False |
| original | -0.461469 [-0.931823,0.00888545] | 0.0830401/0.127771 | -0.350089 [-0.441029,-0.246126] | 0.0261797/0.0261797 | 9.999e-05/0.00069993 | False |

Frozen success rule resolves historical 2/7 to **0/7**; Holm is reported separately for interpretation.

## Random-subset and oracle full-sweep curves (original selector/evaluator)

| budget | random LL margin [CI] | random corpus BLEU margin [CI] | oracle LL margin [CI] | oracle corpus BLEU margin [CI] |
|---:|---:|---:|---:|---:|
| 8 | -7.0144 [-12.288,-1.7404] | -0.91186 [-0.92598,-0.89774] | -8.6822 [-14.885,-2.4797] | -0.94487 [-0.96541,-0.92165] |
| 32 | -5.0795 [-8.8818,-1.2773] | -0.83832 [-0.86195,-0.8147] | -5.3065 [-9.2953,-1.3177] | -0.87819 [-0.93857,-0.82355] |
| 128 | -3.2571 [-5.7242,-0.79003] | -0.74988 [-0.78376,-0.716] | -3.4564 [-6.0298,-0.88298] | -0.71137 [-0.7717,-0.64821] |
| 512 | -1.7531 [-3.0354,-0.4707] | -0.62834 [-0.67579,-0.5809] | -2.6383 [-4.8276,-0.44909] | -0.62925 [-0.70184,-0.55219] |
| 2048 | -0.87764 [-1.538,-0.21729] | -0.49077 [-0.55785,-0.42369] | -0.98166 [-1.884,-0.079315] | -0.51465 [-0.59477,-0.41829] |
| 3530 | -0.68357 [-1.2395,-0.12767] | -0.42915 [-0.50716,-0.35114] | -0.67393 [-1.3143,-0.03354] | -0.44851 [-0.52748,-0.35478] |
| 7060 | -0.46147 [-0.92333,0.00039062] | -0.35009 [-0.45023,-0.24994] | -0.46147 [-0.91683,-0.0061066] | -0.35009 [-0.44103,-0.24613] |

## A/B disjoint candidate-pool replication

A/B intervals are paired over shared targets; they are not independent target/evaluator replications. Full 7x7 transfer matrices and the separately frozen diagonal family are descriptive outside their named confirmatory families.

See `partition_AB_paired.csv`, `random_transfer_curves.csv`, and `oracle_pool_curves.csv`.
