| Dataset | Episodes | Usable | Score | Findings (episodes) | Not checked |
|---|---|---|---|---|---|
| RoboCOIN/Agilex_Cobot_Magic_heat_sandwich | 301 | 64 | 67 | episodes.spikes (237); episodes.spikes (237); episodes.too_long (2); episodes.trajectory_outlier (9); tasks.too_short | - |
| RoboCOIN/Agilex_Cobot_Magic_organize_desk_fail | 117 | 25 | 58 | episodes.spikes (92); episodes.spikes (92); episodes.spikes_systematic; episodes.spikes_systematic; episodes.trajectory_outlier (2) | - |
| RoboCOIN/Agilex_Cobot_Magic_fold_short_sleeve_white | 50 | 17 | 73 | episodes.spikes (33); episodes.spikes (33); episodes.trajectory_outlier (4) | - |
| binhng/vrh31_bodyshop_v1 | 278 | 0 | 5 | episodes.no_motion (278); episodes.spikes (193); episodes.spikes (131); episodes.spikes_systematic; schema.names_shape_mismatch; episodes.too_long (5); episodes.trajectory_outlier (1); schema.action_state_dims_differ; values.dead_dimension | - |
| YOLO2431/self-repair-gripper-v2 | 303 | 297 | 98 | episodes.spikes (5); episodes.spikes (5); episodes.too_short (1); episodes.trajectory_outlier (1); schema.features_not_checked | - |
| RoboCOIN/Leju_Kuavo_4_hotel_services | 484 | 445 | 85 | episodes.spikes (39); info.load_warning; info.load_warning; episodes.too_long (3); episodes.trajectory_outlier (13); schema.action_state_dims_differ; values.dead_dimension; values.dead_dimension | action/state sync: actions do not track the state (delta or velocity control?) |
| lerobot/high_quality_folding | 1200 | 1083 | 98 | episodes.idle (2); episodes.spikes (57); episodes.spikes (1); episodes.too_short (4); sync.action_state_lag (57); sync.action_state_uncorrelated (1); episodes.too_long (17); episodes.trajectory_outlier (6) | - |
| zmfldlwx/community-v1-so101-pick_and_place-converted | 6288 | 6077 | 99 | episodes.idle (7); episodes.duplicate (84); episodes.idle_padding (81); episodes.spikes (1); episodes.too_short (16); sync.action_state_lag (23); sync.action_state_uncorrelated (3); episodes.trajectory_outlier (12) | - |
| lerobot/svla_so100_sorting | 52 | 50 | 97 | episodes.idle (2); episodes.idle_padding (1); episodes.too_short (1); episodes.trajectory_outlier (2) | - |
| lerobot/svla_so100_stacking | 56 | 51 | 94 | episodes.idle (4); episodes.too_short (3); episodes.trajectory_outlier (4) | - |
| lerobot/droid_100 | 100 | 41 | 86 | episodes.idle (1); episodes.spikes (11); episodes.too_short (1); tasks.placeholder (54); episodes.trajectory_outlier (1) | action/state sync: actions do not track the state (delta or velocity control?) |
| lerobot/pusht | 206 | 206 | 100 | none | - |
| lerobot/aloha_sim_insertion_human | 50 | 50 | 100 | none | - |

Datasets per check: episodes.trajectory_outlier 11, episodes.spikes 9, episodes.too_short 6, episodes.idle 5, episodes.too_long 4, episodes.spikes_systematic 2, values.dead_dimension 2, schema.action_state_dims_differ 2, sync.action_state_lag 2, sync.action_state_uncorrelated 2, episodes.idle_padding 2, tasks.too_short 1, schema.names_shape_mismatch 1, episodes.no_motion 1, schema.features_not_checked 1, info.load_warning 1, episodes.duplicate 1, tasks.placeholder 1