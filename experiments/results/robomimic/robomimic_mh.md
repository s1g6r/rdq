# RoboMimic MH natural-noise benchmark

Keep 50% of demonstrations with each method; report the mean operator tier kept (better = 3, okay = 2, worse = 1).

## can (300 demos; median length by tier: better 136, okay 176, worse 270)

| Method | Kept-set mean tier | AUROC worse vs better | Spearman with tier | Spearman with length |
|---|---|---|---|---|
| oracle | 2.67 | nan | nan | nan |
| keep shortest (length) | 2.53 | 0.96 | 0.66 | 1.00 |
| PSD (Sojib et al., replicated) | 2.53 | 0.95 | 0.65 | 1.00 |
| rdq ldlj | 2.50 | 0.94 | 0.65 | 0.96 |
| rdq deminf-lite | 2.19 | 0.68 | 0.35 | 0.08 |
| rdq composite | 2.16 | 0.69 | 0.33 | 0.05 |
| rdq consistency | 2.15 | 0.66 | 0.24 | 0.03 |
| rdq dither | 2.00 | 0.50 | -0.30 | -0.36 |
| random (mean of 1000 draws) | 2.00 | nan | nan | nan |
| rdq hfr | 1.92 | 0.39 | -0.17 | -0.01 |

Published on Can-MH at 50% (Sojib et al. 2026, Table I): random 1.99, DemInf (Hejna et al.) 2.37, CUPID (Agia et al.) 2.32, PSD (Sojib et al.) 2.53, oracle 2.67

rdq rules: 299/300 demos pass (kept-set tier 2.0); scoring took 17.88 s.

## square (300 demos; median length by tier: better 173, okay 259, worse 323)

| Method | Kept-set mean tier | AUROC worse vs better | Spearman with tier | Spearman with length |
|---|---|---|---|---|
| oracle | 2.67 | nan | nan | nan |
| rdq ldlj | 2.49 | 0.94 | 0.62 | 0.93 |
| keep shortest (length) | 2.47 | 0.94 | 0.64 | 1.00 |
| PSD (Sojib et al., replicated) | 2.47 | 0.94 | 0.64 | 1.00 |
| rdq composite | 2.15 | 0.63 | 0.20 | 0.11 |
| rdq deminf-lite | 2.13 | 0.65 | 0.24 | 0.14 |
| rdq consistency | 2.05 | 0.56 | 0.09 | 0.04 |
| rdq dither | 2.01 | 0.52 | 0.08 | -0.02 |
| random (mean of 1000 draws) | 2.00 | nan | nan | nan |
| rdq hfr | 1.88 | 0.35 | -0.21 | -0.34 |

rdq rules: 300/300 demos pass (kept-set tier 2.0); scoring took 18.29 s.
