# Curation benchmark results

## Scenario: mixed

200 episodes, 5 seeds; subset methods keep 130 episodes (the number of good ones). rdq's rules kept 170.2 on average. Scoring took 3.92 s per dataset.

| Method | Success | 95% CI (over seeds) | Bad episodes kept |
|---|---|---|---|
| oracle | 0.995 | 0.985–1.000 | 0% |
| composite | 0.930 | 0.869–0.991 | 16% |
| rules (rdq check) | 0.922 | 0.904–0.941 | 24% |
| rules + composite | 0.904 | 0.882–0.927 | 12% |
| deminf-lite | 0.868 | 0.819–0.917 | 18% |
| consistency | 0.768 | 0.623–0.913 | 20% |
| loss-based | 0.757 | 0.603–0.910 | 26% |
| random | 0.367 | 0.264–0.470 | 34% |
| all data | 0.340 | 0.283–0.396 | 35% |
| random, rules size | 0.327 | 0.253–0.401 | 34% |
| hfr | 0.320 | 0.206–0.433 | 25% |
| psd (Sojib et al.) | 0.271 | 0.126–0.415 | 28% |
| keep shortest | 0.235 | 0.113–0.358 | 29% |

Detection AUROC (bad vs expert):

| Score | All | desync | detour | failed | frozen_log | glitch | hesitant | shaky |
|---|---|---|---|---|---|---|---|---|
| length (longest first) | 0.64 | 0.45 | 0.98 | 0.61 | 0.42 | 0.50 | 0.99 | 0.49 |
| psd (Sojib et al.) | 0.65 | 0.43 | 0.94 | 0.58 | 0.48 | 0.69 | 0.91 | 0.49 |
| deminf-lite | 0.81 | 0.96 | 0.84 | 0.59 | 0.91 | 0.56 | 0.83 | 0.94 |
| consistency | 0.77 | 0.97 | 0.75 | 0.72 | 0.86 | 0.52 | 0.69 | 0.85 |
| hfr | 0.62 | 0.52 | 0.70 | 0.19 | 1.00 | 1.00 | 0.04 | 0.89 |
| dither | 0.53 | 0.50 | 0.68 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| final-state outlier | 0.46 | 0.46 | 0.46 | 0.39 | 0.49 | 0.47 | 0.50 | 0.44 |
| composite | 0.84 | 0.99 | 0.87 | 0.71 | 0.94 | 0.56 | 0.84 | 0.95 |
| loss-based | 0.68 | 1.00 | 0.88 | 0.44 | 0.36 | 0.60 | 0.50 | 0.95 |
| rules (rdq check) | 0.71 | 0.99 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 0.50 |

## Scenario: skill

200 episodes, 5 seeds; subset methods keep 120 episodes (the number of good ones). rdq's rules kept 200.0 on average. Scoring took 5.02 s per dataset.

| Method | Success | 95% CI (over seeds) | Bad episodes kept |
|---|---|---|---|
| oracle | 0.994 | 0.984–1.000 | 0% |
| keep shortest | 0.889 | 0.838–0.940 | 24% |
| psd (Sojib et al.) | 0.884 | 0.835–0.932 | 26% |
| all data | 0.880 | 0.841–0.919 | 40% |
| rules (rdq check) | 0.880 | 0.841–0.919 | 40% |
| random, rules size | 0.880 | 0.841–0.919 | 40% |
| composite | 0.828 | 0.752–0.904 | 20% |
| rules + composite | 0.828 | 0.752–0.904 | 20% |
| consistency | 0.822 | 0.750–0.894 | 24% |
| loss-based | 0.776 | 0.615–0.937 | 23% |
| random | 0.735 | 0.624–0.846 | 41% |
| deminf-lite | 0.731 | 0.640–0.823 | 22% |
| hfr | 0.628 | 0.490–0.766 | 38% |

Detection AUROC (bad vs expert):

| Score | All | detour | failed | hesitant | shaky |
|---|---|---|---|---|---|
| length (longest first) | 0.79 | 0.99 | 0.67 | 0.99 | 0.52 |
| psd (Sojib et al.) | 0.74 | 0.96 | 0.56 | 0.94 | 0.51 |
| deminf-lite | 0.80 | 0.81 | 0.59 | 0.84 | 0.94 |
| consistency | 0.71 | 0.75 | 0.64 | 0.68 | 0.79 |
| hfr | 0.43 | 0.65 | 0.14 | 0.04 | 0.91 |
| dither | 0.54 | 0.66 | 0.50 | 0.50 | 0.50 |
| final-state outlier | 0.50 | 0.49 | 0.42 | 0.56 | 0.52 |
| composite | 0.83 | 0.87 | 0.65 | 0.84 | 0.96 |
| loss-based | 0.82 | 1.00 | 0.50 | 0.77 | 1.00 |
| rules (rdq check) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

## Scenario: structural

200 episodes, 5 seeds; subset methods keep 140 episodes (the number of good ones). rdq's rules kept 140.0 on average. Scoring took 4.66 s per dataset.

| Method | Success | 95% CI (over seeds) | Bad episodes kept |
|---|---|---|---|
| oracle | 0.997 | 0.992–1.000 | 0% |
| rules (rdq check) | 0.997 | 0.992–1.000 | 0% |
| rules + composite | 0.997 | 0.992–1.000 | 0% |
| composite | 0.920 | 0.807–1.000 | 15% |
| deminf-lite | 0.887 | 0.753–1.000 | 16% |
| loss-based | 0.881 | 0.782–0.981 | 26% |
| consistency | 0.764 | 0.574–0.954 | 17% |
| random, rules size | 0.221 | 0.162–0.280 | 29% |
| psd (Sojib et al.) | 0.213 | 0.119–0.308 | 26% |
| keep shortest | 0.211 | 0.107–0.315 | 30% |
| random | 0.185 | 0.113–0.258 | 29% |
| all data | 0.157 | 0.095–0.218 | 30% |
| hfr | 0.129 | 0.070–0.187 | 12% |

Detection AUROC (bad vs expert):

| Score | All | desync | frozen_log | glitch |
|---|---|---|---|---|
| length (longest first) | 0.49 | 0.51 | 0.49 | 0.47 |
| psd (Sojib et al.) | 0.60 | 0.53 | 0.59 | 0.68 |
| deminf-lite | 0.80 | 0.95 | 0.84 | 0.61 |
| consistency | 0.74 | 0.94 | 0.79 | 0.49 |
| hfr | 0.82 | 0.48 | 1.00 | 1.00 |
| dither | 0.50 | 0.50 | 0.51 | 0.50 |
| final-state outlier | 0.51 | 0.48 | 0.52 | 0.53 |
| composite | 0.80 | 0.98 | 0.85 | 0.57 |
| loss-based | 0.60 | 1.00 | 0.28 | 0.52 |
| rules (rdq check) | 1.00 | 1.00 | 1.00 | 1.00 |
