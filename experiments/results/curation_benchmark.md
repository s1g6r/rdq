# Curation benchmark results

## Scenario: mixed

200 episodes, 5 seeds; subset methods keep 130 episodes (the number of good ones). rdq's rules kept 170.2 on average. Scoring took 10.7 s per dataset.

| Method | Success | 95% CI | Bad episodes kept |
|---|---|---|---|
| oracle | 0.995 | 0.990–0.997 | 0% |
| composite | 0.932 | 0.920–0.942 | 16% |
| rules (rdq check) | 0.922 | 0.910–0.933 | 24% |
| rules + composite | 0.922 | 0.910–0.933 | 12% |
| deminf-lite | 0.877 | 0.862–0.891 | 18% |
| consistency | 0.768 | 0.749–0.786 | 20% |
| loss-based | 0.757 | 0.737–0.775 | 26% |
| random | 0.367 | 0.346–0.388 | 34% |
| all data | 0.340 | 0.319–0.361 | 35% |
| random, rules size | 0.327 | 0.307–0.348 | 34% |
| hfr | 0.320 | 0.299–0.340 | 25% |
| psd (Sojib et al.) | 0.271 | 0.251–0.290 | 28% |
| keep shortest | 0.235 | 0.217–0.255 | 29% |

Detection AUROC (bad vs expert):

| Score | All | desync | detour | failed | frozen_log | glitch | hesitant | shaky |
|---|---|---|---|---|---|---|---|---|
| length (longest first) | 0.64 | 0.45 | 0.98 | 0.61 | 0.42 | 0.50 | 0.99 | 0.49 |
| psd (Sojib et al.) | 0.65 | 0.43 | 0.94 | 0.58 | 0.48 | 0.69 | 0.91 | 0.49 |
| deminf-lite | 0.81 | 0.97 | 0.85 | 0.60 | 0.91 | 0.56 | 0.83 | 0.95 |
| consistency | 0.77 | 0.97 | 0.75 | 0.72 | 0.86 | 0.52 | 0.69 | 0.85 |
| hfr | 0.62 | 0.52 | 0.70 | 0.19 | 1.00 | 1.00 | 0.04 | 0.89 |
| dither | 0.53 | 0.50 | 0.68 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| final-state outlier | 0.46 | 0.46 | 0.46 | 0.39 | 0.49 | 0.47 | 0.50 | 0.44 |
| composite | 0.84 | 0.99 | 0.87 | 0.72 | 0.94 | 0.56 | 0.85 | 0.96 |
| loss-based | 0.68 | 1.00 | 0.88 | 0.44 | 0.36 | 0.60 | 0.50 | 0.95 |
| rules (rdq check) | 0.71 | 0.99 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 0.50 |

## Scenario: skill

200 episodes, 5 seeds; subset methods keep 120 episodes (the number of good ones). rdq's rules kept 200.0 on average. Scoring took 11.77 s per dataset.

| Method | Success | 95% CI | Bad episodes kept |
|---|---|---|---|
| oracle | 0.994 | 0.990–0.997 | 0% |
| keep shortest | 0.889 | 0.874–0.902 | 24% |
| psd (Sojib et al.) | 0.884 | 0.869–0.897 | 26% |
| all data | 0.880 | 0.865–0.894 | 40% |
| rules (rdq check) | 0.880 | 0.865–0.894 | 40% |
| random, rules size | 0.880 | 0.865–0.894 | 40% |
| composite | 0.838 | 0.821–0.853 | 19% |
| rules + composite | 0.838 | 0.821–0.853 | 19% |
| consistency | 0.822 | 0.805–0.838 | 24% |
| loss-based | 0.776 | 0.757–0.794 | 23% |
| random | 0.735 | 0.715–0.754 | 41% |
| deminf-lite | 0.701 | 0.681–0.721 | 22% |
| hfr | 0.628 | 0.607–0.649 | 38% |

Detection AUROC (bad vs expert):

| Score | All | detour | failed | hesitant | shaky |
|---|---|---|---|---|---|
| length (longest first) | 0.79 | 0.99 | 0.67 | 0.99 | 0.52 |
| psd (Sojib et al.) | 0.74 | 0.96 | 0.56 | 0.94 | 0.51 |
| deminf-lite | 0.81 | 0.82 | 0.61 | 0.84 | 0.95 |
| consistency | 0.71 | 0.75 | 0.64 | 0.68 | 0.79 |
| hfr | 0.43 | 0.65 | 0.14 | 0.04 | 0.91 |
| dither | 0.54 | 0.66 | 0.50 | 0.50 | 0.50 |
| final-state outlier | 0.50 | 0.49 | 0.42 | 0.56 | 0.52 |
| composite | 0.83 | 0.87 | 0.66 | 0.85 | 0.96 |
| loss-based | 0.82 | 1.00 | 0.50 | 0.77 | 1.00 |
| rules (rdq check) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

## Scenario: structural

200 episodes, 5 seeds; subset methods keep 140 episodes (the number of good ones). rdq's rules kept 140.0 on average. Scoring took 9.25 s per dataset.

| Method | Success | 95% CI | Bad episodes kept |
|---|---|---|---|
| oracle | 0.997 | 0.993–0.999 | 0% |
| rules (rdq check) | 0.997 | 0.993–0.999 | 0% |
| rules + composite | 0.997 | 0.993–0.999 | 0% |
| deminf-lite | 0.932 | 0.921–0.943 | 16% |
| composite | 0.920 | 0.907–0.931 | 15% |
| loss-based | 0.881 | 0.867–0.895 | 26% |
| consistency | 0.764 | 0.745–0.782 | 17% |
| random, rules size | 0.221 | 0.203–0.240 | 29% |
| psd (Sojib et al.) | 0.213 | 0.196–0.232 | 26% |
| keep shortest | 0.211 | 0.194–0.229 | 30% |
| random | 0.185 | 0.169–0.203 | 29% |
| all data | 0.157 | 0.141–0.173 | 30% |
| hfr | 0.129 | 0.115–0.144 | 12% |

Detection AUROC (bad vs expert):

| Score | All | desync | frozen_log | glitch |
|---|---|---|---|---|
| length (longest first) | 0.49 | 0.51 | 0.49 | 0.47 |
| psd (Sojib et al.) | 0.60 | 0.53 | 0.59 | 0.68 |
| deminf-lite | 0.81 | 0.96 | 0.84 | 0.61 |
| consistency | 0.74 | 0.94 | 0.79 | 0.49 |
| hfr | 0.82 | 0.48 | 1.00 | 1.00 |
| dither | 0.50 | 0.50 | 0.51 | 0.50 |
| final-state outlier | 0.51 | 0.48 | 0.52 | 0.53 |
| composite | 0.80 | 0.98 | 0.85 | 0.58 |
| loss-based | 0.60 | 1.00 | 0.28 | 0.52 |
| rules (rdq check) | 1.00 | 1.00 | 1.00 | 1.00 |
