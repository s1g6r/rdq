# R6: upstream licences of public robot-learning datasets (LeRobot mirrors)

Checked 2026-10-12. Companion to `R6_upstream_licences.json` (same entries, machine-readable) and an extension of the knowledge base in `rdq/licenses.py` (DROID, BridgeData V2, LIBERO, AgiBot World, EgoDex, EPIC-KITCHENS, HOI4D and Ego4D are already covered and are not repeated here).

Rules followed: licences come only from primary sources (official site, paper, the authors' GitHub README/LICENSE, the authors' own HF card or Kaggle page). A code licence is never read as a data licence: when only the code licence is stated, the entry says `not stated`. *This is diligence support, not legal advice.*

## How to read the confidence column

- **high**: an explicit data licence from the original authors (README sentence, project page, LICENSE file, Kaggle licence field, or explicit text on the authors' own card).
- **medium**: an umbrella or indirect statement (the Open X-Embodiment README), a bare YAML tag on the authors' own card that may be the LeRobot template default, or an explicit but self-contradictory statement. For `not stated` entries, medium means the primary sources were checked and none gives data terms.
- **low**: the scope is unclear (e.g. a website footer), or the component's own release was not located or checked.

## Summary table

| key | dataset | upstream licence | commercial training | redistribution | conf. | Hub mirrors in top-600 |
|---|---|---|---|---|---|---|
| `oxe_language_table` | Language-Table (Open X-Embodiment component) | CC-BY-4.0 | yes | yes | medium | 3 |
| `oxe_fractal_rt1` | RT-1 Robot Action / fractal20220817_data | CC-BY-4.0 | yes | yes | medium | 3 |
| `oxe_kuka_qtopt` | QT-Opt Kuka grasping (OXE 'kuka') | CC-BY-4.0 | yes | yes | medium | 1 |
| `bc_z` | BC-Z | CC-BY-4.0 | yes | yes | high | 2 |
| `furniture_bench` | FurnitureBench | CC-BY-4.0 | yes | yes | medium | 1 |
| `taco_play` | TACO-Play (Task-Agnostic Real World Robot Play) | CC-BY-4.0 | yes | yes | high | 2 |
| `fmb` | FMB (Functional Manipulation Benchmark) | CC-BY-4.0 | yes | yes | high | 2 |
| `berkeley_cable_routing` | Berkeley Cable Routing | CC-BY-4.0 | yes | yes | high | 2 |
| `roboturk_real` | RoboTurk Real (Sawyer) dataset | MIT | yes | yes | high | 1 |
| `utaustin_mutex` | UT Austin MUTEX | CC-BY-4.0 | yes | yes | medium | 2 |
| `toto` | TOTO benchmark | CC-BY-4.0 | yes | yes | medium | 2 |
| `berkeley_autolab_ur5` | Berkeley Autolab UR5 | CC-BY-4.0 | yes | yes | medium | 2 |
| `berkeley_rpt` | Berkeley RPT (Robot Learning with Sensorimotor Pre-training) | CC-BY-4.0 | yes | yes | medium | 1 |
| `stanford_kuka_multimodal` | Stanford Kuka Multimodal (Making Sense of Vision and Touch) | CC-BY-4.0 | yes | yes | medium | 1 |
| `jaco_play` | Jaco Play (CLVR) | CC-BY-4.0 | yes | yes | high | 2 |
| `viola` | UT Austin VIOLA | CC-BY-4.0 | yes | yes | medium | 0 |
| `nyu_franka_play` | NYU Franka Play (play-to-policy / C-BeT) | CC-BY-4.0 | yes | yes | medium | 2 |
| `austin_buds_sailor_sirius` | UT Austin BUDS / Sailor / Sirius | CC-BY-4.0 | yes | yes | medium | 2 |
| `ucsd_kitchen` | UCSD Kitchen | CC-BY-4.0 | yes | yes | low | 1 |
| `ucsd_pick_and_place` | UCSD Pick and Place | CC-BY-4.0 | yes | yes | low | 1 |
| `cmu_play_fusion` | CMU PlayFusion | CC-BY-4.0 | yes | yes | medium | 1 |
| `iamlab_cmu_pickup_insert` | IAMLab CMU Pickup Insert | CC-BY-4.0 | yes | yes | low | 1 |
| `dlr_datasets` | DLR (SARA pour / grid clamp, EDAN shared control) | CC-BY-4.0 | yes | yes | low | 0 |
| `columbia_cairlab_pusht_real` | Columbia CAIRLab Push-T real | CC-BY-4.0 | yes | yes | medium | 0 |
| `stanford_hydra` | Stanford HYDRA | CC-BY-4.0 | yes | yes | medium | 2 |
| `dobbe_hony` | Dobb-E / Homes of New York (OXE 'dobbe') | CC-BY-4.0 | yes | yes | medium | 1 |
| `nyu_door_opening` | NYU Door Opening (VINN) | CC-BY-4.0 | yes | yes | low | 1 |
| `berkeley_fanuc_manipulation` | Berkeley Fanuc Manipulation | LicenseRef-BerkeleyFanuc-MIT-NonCommercial | unclear | yes | medium | 1 |
| `berkeley_mvp` | Berkeley MVP (Real-World Robot Learning with Masked Visual Pre-training) | CC-BY-4.0 | yes | yes | medium | 1 |
| `stanford_robocook` | Stanford RoboCook | CC-BY-4.0 | yes | yes | medium | 1 |
| `imperialcollege_sawyer_wrist_cam` | Imperial College Sawyer wrist cam | CC-BY-4.0 | yes | yes | low | 1 |
| `usc_cloth_sim` | USC Cloth Sim | CC-BY-4.0 | yes | yes | low | 1 |
| `aloha_sim_act` | ACT ALOHA sim demos (insertion, transfer cube; human and scripted) | not stated | unclear | unclear | medium | 8 |
| `mobile_aloha` | Mobile ALOHA datasets | not stated | unclear | unclear | medium | 6 |
| `aloha_static` | ALOHA static datasets (as ported by LeRobot) | not stated | unclear | unclear | low | 15 |
| `pusht_diffusion_policy` | Push-T (Diffusion Policy, Chi et al.) | not stated | unclear | unclear | medium | 3 |
| `xarm_fowm` | xArm lift/push offline datasets (FOWM, Feng et al.) | not stated | unclear | unclear | medium | 8 |
| `umi_cup_in_the_wild` | UMI in-the-wild cup arrangement | not stated | unclear | unclear | medium | 1 |
| `metaworld_mt50_lerobot` | Meta-World MT50 demos (LeRobot) | Apache-2.0 | yes | yes | low | 1 |
| `robocasa` | RoboCasa / RoboCasa365 | CC-BY-4.0 | yes | yes | high | 9 |
| `behavior_1k_2025_challenge` | BEHAVIOR-1K 2025 challenge demos | MIT | unclear | yes | medium | 4 |
| `abc_130k` | ABC-130K (XDOF) | Apache-2.0 | yes | yes | high | 6 |
| `l2d` | L2D (Learning to Drive, Yaak + Hugging Face) | Apache-2.0 | yes | yes | medium | 2 |
| `unitree_g1_datasets` | Unitree robotics datasets (unitreerobotics/*) | Apache-2.0 | yes | yes | medium | 37 |
| `robocoin` | RoboCOIN (BAAI / FlagOpen) | Apache-2.0 | yes | yes | high | 230 |
| `rh20t` | RH20T | LicenseRef-RH20T-split-CC-BY-SA-4.0-and-CC-BY-NC-4.0 | unclear | yes | high | 5 |
| `roboset` | RoboSet (RoboHive / RoboPen) | MIT | yes | yes | low | 0 |
| `galaxea_open_world` | Galaxea Open-World Dataset | CC-BY-NC-SA-4.0 | no | yes | high | 0 |
| `agibot_digital_world` | AgiBot Digital World | CC-BY-NC-SA-4.0 | no | yes | high | 0 |
| `dexmimicgen` | DexMimicGen datasets | CC-BY-4.0 | yes | yes | high | 0 |
| `mimicgen` | MimicGen datasets | CC-BY-4.0 | yes | yes | high | 0 |
| `robomimic` | robomimic datasets | MIT | yes | yes | medium | 0 |
| `lerobot_svla_so10x` | SmolVLA SO-100/SO-101 community datasets (lerobot/svla_*) | Apache-2.0 | yes | yes | medium | 4 |
| `kai0` | OpenDriveLab Kai0 (chi-zero) data | CC-BY-NC-SA-4.0 | no | yes | high | 1 |
| `hiw_500` | BitRobot HIW-500 (Humanoids in the Wild) | CC-BY-4.0 | yes | yes | high | 1 |
| `habit` | Config HABIT | CC-BY-4.0 | yes | yes | high | 1 |
| `tacverse` | TacVerse (Xense Robotics) | CC-BY-SA-4.0 | yes | yes | high | 1 |

## Open X-Embodiment: umbrella versus component terms

The OXE README (github.com/google-deepmind/open_x_embodiment, 'License and Disclaimer') says: *All software is licensed under the Apache License, Version 2.0* and *All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY).* No per-dataset licence column is publicly readable (the OXE spreadsheet could not be retrieved here), and the TFDS catalog's CC BY 4.0 footer covers the catalog page only, not the data.

How we resolved it:
1. **When the component has its own explicit data licence, that governs.** It comes from the rightsholder. Found for: BC-Z (CC BY 4.0, Google's Kaggle), TACO-Play (CC BY 4.0, author's Kaggle), FMB (CC BY 4.0), Berkeley Cable Routing (CC BY 4.0), Jaco Play (CC BY 4.0), RoboTurk Real (**MIT**, different from the umbrella) and Berkeley Fanuc (**'MIT' but described as non-commercial**, which contradicts the umbrella).
2. **When the component states nothing, the umbrella is the only statement.** The entry is recorded as CC BY 4.0 at medium confidence. This is contested: the umbrella sentence sits in a code repository's README, it is not clear that it was meant to (or legally could) relicense contributor-owned data held in gs://gresearch/robotics, and no contributor agreement is public. It is still the only published grant, and it agrees with the component-level terms wherever those exist (except RoboTurk and Fanuc).
3. **Google-owned components** (Language-Table, RT-1/fractal, QT-Opt Kuka): the umbrella comes from the same organisation (Google DeepMind), which makes it stronger than for third-party contributions. It is still not dataset-specific, so these stay at medium.
4. For components marked low, no standalone release was checked, so a component licence could still exist.

## Mirrors on the Hub that disagree with upstream

`conflict` means the declared tag drops duties or restrictions the upstream imposes; a CC BY dataset tagged apache-2.0 or mit loses its attribution and change-notice duties on paper. `no upstream grant` means the original authors state no data licence, so the mirror's tag has nothing to stand on (it usually repeats the upstream code licence or the LeRobot default). Downloads come from `hub_top_ids.txt`.

| Hub id | downloads | declared | upstream | conf. | status |
|---|---:|---|---|---|---|
| IPEC-COMMUNITY/language_table_lerobot | 111,332 | apache-2.0 | CC-BY-4.0 (`oxe_language_table`) | medium | conflict |
| IPEC-COMMUNITY/kuka_lerobot | 99,234 | apache-2.0 | CC-BY-4.0 (`oxe_kuka_qtopt`) | medium | conflict |
| IPEC-COMMUNITY/fractal20220817_data_lerobot | 72,331 | apache-2.0 | CC-BY-4.0 (`oxe_fractal_rt1`) | medium | conflict |
| saaduddinM/rbo_oxe_base_language_table_lerobot | 21,954 | apache-2.0 | CC-BY-4.0 (`oxe_language_table`) | medium | conflict |
| ember-lab-berkeley/robocasa365-pretrain-mg | 12,022 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| IPEC-COMMUNITY/bc_z_lerobot | 11,501 | apache-2.0 | CC-BY-4.0 (`bc_z`) | high | conflict |
| lerobot/berkeley_cable_routing | 4,368 | apache-2.0 | CC-BY-4.0 (`berkeley_cable_routing`) | high | conflict |
| IPEC-COMMUNITY/furniture_bench_dataset_lerobot | 3,894 | apache-2.0 | CC-BY-4.0 (`furniture_bench`) | medium | conflict |
| ygtxr1997/bc_z_lerobot | 3,550 | apache-2.0 | CC-BY-4.0 (`bc_z`) | high | conflict |
| IPEC-COMMUNITY/taco_play_lerobot | 3,471 | apache-2.0 | CC-BY-4.0 (`taco_play`) | high | conflict |
| ember-lab-berkeley/robocasa365-pretrain-composite | 3,465 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| IPEC-COMMUNITY/fmb_dataset_lerobot | 3,436 | apache-2.0 | CC-BY-4.0 (`fmb`) | high | conflict |
| lerobot/stanford_kuka_multimodal_dataset | 3,316 | mit | CC-BY-4.0 (`stanford_kuka_multimodal`) | medium | conflict |
| IPEC-COMMUNITY/berkeley_cable_routing_lerobot | 3,172 | apache-2.0 | CC-BY-4.0 (`berkeley_cable_routing`) | high | conflict |
| IPEC-COMMUNITY/roboturk_lerobot | 3,077 | apache-2.0 | MIT (`roboturk_real`) | high | minor (MIT allows sublicensing) |
| ember-lab-berkeley/robocasa365-target-composite-unseen | 2,834 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| infope/robocasa | 2,655 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| ember-lab-berkeley/robocasa365-target-composite-seen | 2,642 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| IPEC-COMMUNITY/utaustin_mutex_lerobot | 2,413 | apache-2.0 | CC-BY-4.0 (`utaustin_mutex`) | medium | conflict |
| IPEC-COMMUNITY/toto_lerobot | 2,132 | apache-2.0 | CC-BY-4.0 (`toto`) | medium | conflict |
| IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot | 2,085 | apache-2.0 | CC-BY-4.0 (`berkeley_autolab_ur5`) | medium | conflict |
| IPEC-COMMUNITY/berkeley_rpt_lerobot | 2,022 | apache-2.0 | CC-BY-4.0 (`berkeley_rpt`) | medium | conflict |
| ember-lab-berkeley/robocasa365-target-atomic | 1,509 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| IPEC-COMMUNITY/jaco_play_lerobot | 1,425 | apache-2.0 | CC-BY-4.0 (`jaco_play`) | high | conflict |
| IPEC-COMMUNITY/austin_sirius_dataset_lerobot | 1,388 | apache-2.0 | CC-BY-4.0 (`austin_buds_sailor_sirius`) | medium | conflict |
| IPEC-COMMUNITY/stanford_hydra_dataset_lerobot | 1,361 | apache-2.0 | CC-BY-4.0 (`stanford_hydra`) | medium | conflict |
| lerobot/robocasa_target_human_unified | 1,358 | apache-2.0 | CC-BY-4.0 (`robocasa`) | high | conflict |
| LehongWu/language_table_sim-lerobot | 1,323 | apache-2.0 | CC-BY-4.0 (`oxe_language_table`) | medium | conflict |
| IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot | 1,276 | apache-2.0 | CC-BY-4.0 (`iamlab_cmu_pickup_insert`) | low | conflict |
| BrunoM42/fractal20220817_data_lerobot | 1,152 | apache-2.0 | CC-BY-4.0 (`oxe_fractal_rt1`) | medium | conflict |
| IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot | 1,145 | apache-2.0 | CC-BY-4.0 (`nyu_franka_play`) | medium | conflict |
| IPEC-COMMUNITY/dobbe_lerobot | 1,128 | apache-2.0 | CC-BY-4.0 (`dobbe_hony`) | medium | conflict |
| lerobot/utaustin_mutex | 1,127 | mit | CC-BY-4.0 (`utaustin_mutex`) | medium | conflict |
| IPEC-COMMUNITY/cmu_play_fusion_lerobot | 1,116 | apache-2.0 | CC-BY-4.0 (`cmu_play_fusion`) | medium | conflict |
| IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot | 1,068 | apache-2.0 | LicenseRef-BerkeleyFanuc-MIT-NonCommercial (`berkeley_fanuc_manipulation`) | medium | conflict (review) |
| ember-lab-berkeley/robocasa365-pretrain-atomic | 1,057 | mit | CC-BY-4.0 (`robocasa`) | high | conflict |
| IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot | 1,048 | apache-2.0 | CC-BY-4.0 (`nyu_door_opening`) | low | conflict |
| lerobot/ucsd_pick_and_place_dataset | 1,004 | mit | CC-BY-4.0 (`ucsd_pick_and_place`) | low | conflict |
| Voxel51/robocasa-MG_100 | 974 | apache-2.0 | CC-BY-4.0 (`robocasa`) | high | conflict |
| lerobot/nyu_franka_play_dataset | 895 | mit | CC-BY-4.0 (`nyu_franka_play`) | medium | conflict |
| lerobot/stanford_hydra_dataset | 870 | mit | CC-BY-4.0 (`stanford_hydra`) | medium | conflict |
| IPEC-COMMUNITY/berkeley_mvp_lerobot | 833 | apache-2.0 | CC-BY-4.0 (`berkeley_mvp`) | medium | conflict |
| lerobot/toto | 797 | mit | CC-BY-4.0 (`toto`) | medium | conflict |
| lerobot/imperialcollege_sawyer_wrist_cam | 792 | apache-2.0 | CC-BY-4.0 (`imperialcollege_sawyer_wrist_cam`) | low | conflict |
| lerobot/stanford_robocook | 779 | mit | CC-BY-4.0 (`stanford_robocook`) | medium | conflict |
| IPEC-COMMUNITY/austin_sailor_dataset_lerobot | 777 | apache-2.0 | CC-BY-4.0 (`austin_buds_sailor_sirius`) | medium | conflict |
| allenai/MolmoAct2-RT-1-Dataset | 742 | apache-2.0 | CC-BY-4.0 (`oxe_fractal_rt1`) | medium | conflict |
| lerobot/usc_cloth_sim | 738 | apache-2.0 | CC-BY-4.0 (`usc_cloth_sim`) | low | conflict |
| lerobot/ucsd_kitchen_dataset | 735 | mit | CC-BY-4.0 (`ucsd_kitchen`) | low | conflict |

Totals over the top-600 list: 47 conflicting mirrors (401,178 downloads). Of these, 16 (59,439 downloads) conflict with a **high-confidence** upstream; the rest rely on the OXE umbrella. Another 1 needs review (Berkeley Fanuc), 1 is a minor mismatch, and 41 mirrors (283,810 downloads) declare a licence for data whose authors never stated one.

### Mirrors with no upstream grant (upstream says nothing about data)

| Hub id | downloads | declared | upstream entry |
|---|---:|---|---|
| lerobot/aloha_sim_insertion_human | 48,497 | mit | `aloha_sim_act` |
| lerobot/aloha_sim_transfer_cube_human | 46,649 | mit | `aloha_sim_act` |
| lerobot/aloha_sim_insertion_scripted | 40,058 | mit | `aloha_sim_act` |
| lerobot/pusht | 25,838 | mit | `pusht_diffusion_policy` |
| lerobot/aloha_sim_transfer_cube_scripted | 22,963 | apache-2.0 | `aloha_sim_act` |
| lerobot/aloha_sim_insertion_human_image | 10,011 | mit | `aloha_sim_act` |
| lerobot/aloha_sim_insertion_scripted_image | 8,165 | mit | `aloha_sim_act` |
| lerobot/xarm_lift_medium | 8,094 | mit | `xarm_fowm` |
| lerobot/pusht_image | 7,743 | mit | `pusht_diffusion_policy` |
| lerobot/aloha_sim_transfer_cube_human_image | 7,079 | mit | `aloha_sim_act` |
| lerobot/aloha_mobile_shrimp | 6,602 | apache-2.0 | `mobile_aloha` |
| lerobot/aloha_sim_transfer_cube_scripted_image | 6,545 | mit | `aloha_sim_act` |
| lerobot/aloha_mobile_cabinet | 4,567 | apache-2.0 | `mobile_aloha` |
| lerobot/aloha_static_coffee | 3,130 | mit | `aloha_static` |
| lerobot/xarm_lift_medium_replay | 2,819 | mit | `xarm_fowm` |
| lerobot/aloha_static_cups_open | 2,773 | mit | `aloha_static` |
| lerobot/xarm_push_medium | 2,683 | mit | `xarm_fowm` |
| lerobot/xarm_push_medium_replay | 2,156 | mit | `xarm_fowm` |
| lerobot/aloha_static_battery | 1,802 | apache-2.0 | `aloha_static` |
| lerobot/aloha_static_towel | 1,607 | apache-2.0 | `aloha_static` |
| lerobot/aloha_static_candy | 1,589 | mit | `aloha_static` |
| lerobot/aloha_mobile_wipe_wine | 1,529 | mit | `mobile_aloha` |
| lerobot/pusht_keypoints | 1,453 | apache-2.0 | `pusht_diffusion_policy` |
| lerobot/aloha_static_screw_driver | 1,435 | mit | `aloha_static` |
| lerobot/aloha_static_thread_velcro | 1,359 | apache-2.0 | `aloha_static` |
| lerobot/aloha_static_vinh_cup | 1,257 | mit | `aloha_static` |
| lerobot/xarm_lift_medium_image | 1,242 | mit | `xarm_fowm` |
| lerobot/aloha_static_tape | 1,236 | apache-2.0 | `aloha_static` |
| lerobot/xarm_push_medium_image | 1,149 | mit | `xarm_fowm` |
| lerobot/xarm_lift_medium_replay_image | 1,125 | mit | `xarm_fowm` |
| lerobot/aloha_static_fork_pick_up | 1,110 | apache-2.0 | `aloha_static` |
| lerobot/aloha_static_ziploc_slide | 1,096 | mit | `aloha_static` |
| lerobot/aloha_static_vinh_cup_left | 1,079 | mit | `aloha_static` |
| lerobot/xarm_push_medium_replay_image | 1,058 | mit | `xarm_fowm` |
| lerobot/aloha_static_pingpong_test | 1,025 | mit | `aloha_static` |
| lerobot/umi_cup_in_the_wild | 1,001 | apache-2.0 | `umi_cup_in_the_wild` |
| lerobot/aloha_static_pro_pencil | 981 | mit | `aloha_static` |
| lerobot/aloha_mobile_chair | 905 | mit | `mobile_aloha` |
| lerobot/aloha_static_coffee_new | 851 | mit | `aloha_static` |
| lerobot/aloha_mobile_wash_pan | 785 | mit | `mobile_aloha` |
| lerobot/aloha_mobile_elevator | 764 | mit | `mobile_aloha` |

### Consistent mirrors (for completeness)

behavior-1k/2025-challenge-demos (81,551, mit), yaak-ai/L2D (71,353, apache-2.0), configinc/HABIT (30,603, cc-by-4.0), BitRobot/HIW-500-LeRobot (20,735, cc-by-4.0), TacVerse/opendata (18,168, cc-by-sa-4.0), lerobot/svla_so100_pickplace (14,754, apache-2.0), lerobot/svla_so100_sorting (13,223, apache-2.0), lerobot/svla_so100_stacking (12,855, apache-2.0), lerobot/abc_130k_v3_train (12,630, apache-2.0), OpenDriveLab-org/Kai0 (11,188, cc-by-nc-sa-4.0), lerobot/svla_so101_pickplace (8,664, apache-2.0), lerobot/metaworld_mt50 (5,814, apache-2.0), Dario-Shit4/behavior-1k_2025-challenge-demos (4,978, mit), typoverflow/abc_130k_train (4,453, apache-2.0), lerobot/abc_130k_v3_smoke (3,453, apache-2.0), lerobot/taco_play (2,771, cc-by-4.0), k1000dai/behavior1k-only-rgb (1,881, mit), lerobot/berkeley_autolab_ur5 (1,489, cc-by-4.0), elonelonelon/2025-challenge-demos (1,459, mit), LinhanWang/ABC-130k-lerobot (1,447, apache-2.0), yaak-ai/L2D-v3 (1,346, apache-2.0), lerobot/jaco_play (1,267, cc-by-4.0), robot-lev/rh20t_cfg1 (1,246, other), Voxel51/rh20t-cfg1-28ep (1,064, other), robot-lev/rh20t_cfg2 (1,057, other), Voxel51/fmb-multi-302ep (980, cc-by-4.0), lerobot/abc_130k_v3_val (937, apache-2.0), robot-lev/rh20t_cfg5 (809, other), huzheyuan/abc130k_v3_train (782, apache-2.0), robot-lev/rh20t_cfg3 (752, other). In addition, all 230 RoboCOIN/* repos and all 37 unitreerobotics/* repos are the rightsholders' own releases.

## Entry details

### Language-Table (Open X-Embodiment component) (`oxe_language_table`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `language[-_]?table`
- Matched Hub ids (3): IPEC-COMMUNITY/language_table_lerobot, saaduddinM/rbo_oxe_base_language_table_lerobot, LehongWu/language_table_sim-lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Google's own release (github.com/google-research/language-table) has an Apache-2.0 LICENSE for code and its README states no data licence; data is served from the same gresearch/robotics bucket as OXE. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### RT-1 Robot Action / fractal20220817_data (`oxe_fractal_rt1`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `fractal20220817|(^|[/_-])rt-1([/_-]|$)`
- Matched Hub ids (3): IPEC-COMMUNITY/fractal20220817_data_lerobot, BrunoM42/fractal20220817_data_lerobot, allenai/MolmoAct2-RT-1-Dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). RT-1 project page (robotics-transformer1.github.io) links the data (gs://gresearch/rt-1-data-release) without a licence; robotics_transformer repo is Apache-2.0 for code only. allenai/MolmoAct2-RT-1-Dataset is a processed derivative. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### QT-Opt Kuka grasping (OXE 'kuka') (`oxe_kuka_qtopt`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `(^|/)kuka([-_](lerobot|qt[-_]?opt))?$`
- Matched Hub ids (1): IPEC-COMMUNITY/kuka_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone data licence for the QT-Opt data was found; Google contribution distributed through OXE. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### BC-Z (`bc_z`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "Attribution 4.0 International (CC BY 4.0)" (https://www.kaggle.com/datasets/google/bc-z-robot)
- Repo-id regex: `(^|[/_-])bc[-_]z([-_]|$)`
- Matched Hub ids (2): IPEC-COMMUNITY/bc_z_lerobot, ygtxr1997/bc_z_lerobot
- Notes: Kaggle licence field on the dataset published by the 'Google' Kaggle account (linked from sites.google.com/view/bc-z). Agrees with the OXE umbrella.

### FurnitureBench (`furniture_bench`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `furniture[-_]?bench`
- Matched Hub ids (1): IPEC-COMMUNITY/furniture_bench_dataset_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (clvrai/furniture-bench, MIT LICENSE for code; dataset docs and site show only '(c) Copyright 2023, CLVR @ KAIST') states no data licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### TACO-Play (Task-Agnostic Real World Robot Play) (`taco_play`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "Attribution 4.0 International (CC BY 4.0)" (https://www.kaggle.com/datasets/oiermees/taco-robot)
- Repo-id regex: `taco[-_]?play`
- Matched Hub ids (2): IPEC-COMMUNITY/taco_play_lerobot, lerobot/taco_play
- Notes: Kaggle licence field on the dataset published by first author Oier Mees (the LeRobot card's homepage). Agrees with the OXE umbrella.

### FMB (Functional Manipulation Benchmark) (`fmb`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "All data is provided under the Creative Commons Attribution 4.0 International License" (https://functional-manipulation-benchmark.github.io/)
- Repo-id regex: `(^|[/_-])fmb([-_]|$)`
- Matched Hub ids (2): IPEC-COMMUNITY/fmb_dataset_lerobot, Voxel51/fmb-multi-302ep
- Notes: Explicit data licence on the official project site; agrees with the OXE umbrella.

### Berkeley Cable Routing (`berkeley_cable_routing`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "All data is provided under the Creative Commons Attribution 4.0 International License" (https://sites.google.com/view/cablerouting/data)
- Repo-id regex: `cable[-_]routing`
- Matched Hub ids (2): lerobot/berkeley_cable_routing, IPEC-COMMUNITY/berkeley_cable_routing_lerobot
- Notes: Explicit statement on the official dataset page (linked from charlesxu0124/CableRouting, whose MIT LICENSE covers code). Agrees with the OXE umbrella.

### RoboTurk Real (Sawyer) dataset (`roboturk_real`)

- Licence: **MIT**; commercial training: yes; redistribution: yes; restrictions: attribution; confidence: high
- Statement: "This dataset and codebase are released under the MIT License." (https://github.com/StanfordVL/roboturk_real_dataset)
- Repo-id regex: `roboturk`
- Matched Hub ids (1): IPEC-COMMUNITY/roboturk_lerobot
- Notes: Component's own explicit data licence (MIT) differs from the OXE umbrella (CC BY 4.0). The rightsholder's own MIT grant governs the original; both are permissive. MIT allows sublicensing, so an Apache-2.0 tag on a mirror is not a hard conflict provided the MIT notice is kept.

### UT Austin MUTEX (`utaustin_mutex`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `utaustin[-_]mutex|(^|/)mutex([-_]|$)`
- Matched Hub ids (2): IPEC-COMMUNITY/utaustin_mutex_lerobot, lerobot/utaustin_mutex
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (UT-Austin-RPL/MUTEX, MIT LICENSE for code; project page; Box download) states no data licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### TOTO benchmark (`toto`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `(^|/)toto([-_]lerobot)?$`
- Matched Hub ids (2): IPEC-COMMUNITY/toto_lerobot, lerobot/toto
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (AGI-Labs/toto_benchmark, BSD-2-Clause LICENSE for code; toto-benchmark.org) states no data licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Berkeley Autolab UR5 (`berkeley_autolab_ur5`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `berkeley[-_]autolab[-_]ur5`
- Matched Hub ids (2): IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot, lerobot/berkeley_autolab_ur5
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component site (sites.google.com/view/berkeley-ur5) gives a citation request but no licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Berkeley RPT (Robot Learning with Sensorimotor Pre-training) (`berkeley_rpt`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `berkeley[-_]rpt`
- Matched Hub ids (1): IPEC-COMMUNITY/berkeley_rpt_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Project page states no licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Stanford Kuka Multimodal (Making Sense of Vision and Touch) (`stanford_kuka_multimodal`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `stanford[-_]kuka[-_]multimodal`
- Matched Hub ids (1): lerobot/stanford_kuka_multimodal_dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (stanford-iprl-lab/multimodal_representation, MIT LICENSE for code; project site) states no data licence. The LeRobot mirror's 'mit' tag mirrors the code licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Jaco Play (CLVR) (`jaco_play`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "The data is available under the Creative Commons Attribution 4.0 International License." (https://github.com/clvrai/clvr_jaco_play_dataset)
- Repo-id regex: `jaco[-_]?play`
- Matched Hub ids (2): IPEC-COMMUNITY/jaco_play_lerobot, lerobot/jaco_play
- Notes: README statement plus a CC BY 4.0 LICENSE file in the official repo. Agrees with the OXE umbrella.

### UT Austin VIOLA (`viola`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `(^|[/_-])viola([-_]|$)`
- Matched Hub ids: none in the top-600 list
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (UT-Austin-RPL/VIOLA, MIT LICENSE for code; project page) states no data licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### NYU Franka Play (play-to-policy / C-BeT) (`nyu_franka_play`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `nyu[-_]franka[-_]play`
- Matched Hub ids (2): IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot, lerobot/nyu_franka_play_dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component release (jeffacce/play-to-policy, MIT LICENSE for code; datasets on OSF; project page) states no data licence; OSF licence field could not be read. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### UT Austin BUDS / Sailor / Sirius (`austin_buds_sailor_sirius`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `austin[-_](buds|sailor|sirius)`
- Matched Hub ids (2): IPEC-COMMUNITY/austin_sirius_dataset_lerobot, IPEC-COMMUNITY/austin_sailor_dataset_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Component repos (UT-Austin-RPL/BUDS, sailor, sirius; MIT LICENSE for code) state no data licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### UCSD Kitchen (`ucsd_kitchen`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `ucsd[-_]kitchen`
- Matched Hub ids (1): lerobot/ucsd_kitchen_dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release located; not checked beyond OXE. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### UCSD Pick and Place (`ucsd_pick_and_place`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `ucsd[-_]pick[-_]and[-_]place`
- Matched Hub ids (1): lerobot/ucsd_pick_and_place_dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release located; not checked beyond OXE. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### CMU PlayFusion (`cmu_play_fusion`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `cmu[-_]play[-_]fusion`
- Matched Hub ids (1): IPEC-COMMUNITY/cmu_play_fusion_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Project page (play-fusion.github.io) states no licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### IAMLab CMU Pickup Insert (`iamlab_cmu_pickup_insert`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `iamlab[-_]cmu`
- Matched Hub ids (1): IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release checked. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### DLR (SARA pour / grid clamp, EDAN shared control) (`dlr_datasets`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `(^|/)dlr[-_](sara|edan)`
- Matched Hub ids: none in the top-600 list
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release checked. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Columbia CAIRLab Push-T real (`columbia_cairlab_pusht_real`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `columbia[-_]cairlab[-_]pusht`
- Matched Hub ids: none in the top-600 list
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Real Push-T data from the Diffusion Policy authors; their repo licence ('This repository is released under the MIT license.') covers code, and no data licence is stated. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Stanford HYDRA (`stanford_hydra`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `stanford[-_]hydra`
- Matched Hub ids (2): IPEC-COMMUNITY/stanford_hydra_dataset_lerobot, lerobot/stanford_hydra_dataset
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Project page states no licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Dobb-E / Homes of New York (OXE 'dobbe') (`dobbe_hony`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `(^|/)dobbe([-_]|$)`
- Matched Hub ids (1): IPEC-COMMUNITY/dobbe_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). dobb-e.com licenses source code MIT and website content separately; no licence is stated for the HoNY data. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### NYU Door Opening (VINN) (`nyu_door_opening`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `nyu[-_]door[-_]opening`
- Matched Hub ids (1): IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release checked. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Berkeley Fanuc Manipulation (`berkeley_fanuc_manipulation`)

- Licence: **LicenseRef-BerkeleyFanuc-MIT-NonCommercial**; commercial training: unclear; redistribution: yes; restrictions: attribution, non-commercial; confidence: medium
- Statement: "The dataset is released under the permissive MIT license, allowing free use, redistribution, and adaptation" (https://sites.google.com/berkeley.edu/fanuc-manipulation)
- Repo-id regex: `berkeley[-_]fanuc`
- Matched Hub ids (1): IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot
- Notes: Self-contradictory: the page names MIT but (per two separate fetches) limits the grant to non-commercial purposes; the raw page could not be retrieved to confirm the tail of the sentence. Contested against the OXE umbrella (CC BY 4.0). Treat commercial use as unclear and verify manually.

### Berkeley MVP (Real-World Robot Learning with Masked Visual Pre-training) (`berkeley_mvp`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `berkeley[-_]mvp`
- Matched Hub ids (1): IPEC-COMMUNITY/berkeley_mvp_lerobot
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Project page states no licence. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Stanford RoboCook (`stanford_robocook`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `robocook`
- Matched Hub ids (1): lerobot/stanford_robocook
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). Project page links a Google Drive download with no licence; hshi74/robocook LICENSE (MIT) covers code. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### Imperial College Sawyer wrist cam (`imperialcollege_sawyer_wrist_cam`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `imperialcollege[-_]sawyer`
- Matched Hub ids (1): lerobot/imperialcollege_sawyer_wrist_cam
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release checked. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### USC Cloth Sim (`usc_cloth_sim`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY)." (https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer)
- Repo-id regex: `usc[-_]cloth[-_]sim`
- Matched Hub ids (1): lerobot/usc_cloth_sim
- Notes: Licence comes only from the Open X-Embodiment umbrella README (Apache-2.0 for software, CC BY 4.0 for 'all other materials'). No standalone component release checked. Whether the umbrella line covers contributed datasets hosted in gs://gresearch/robotics is not explicit and is contested; treat as CC BY 4.0 at medium confidence.

### ACT ALOHA sim demos (insertion, transfer cube; human and scripted) (`aloha_sim_act`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: medium
- Statement: "You can find all scripted/human demo for simulated environments here." (https://github.com/tonyzhaozh/act)
- Repo-id regex: `(^|/)aloha[-_]sim[-_](insertion|transfer[-_]cube)`
- Matched Hub ids (8): lerobot/aloha_sim_insertion_human, lerobot/aloha_sim_transfer_cube_human, lerobot/aloha_sim_insertion_scripted, lerobot/aloha_sim_transfer_cube_scripted, lerobot/aloha_sim_insertion_human_image, lerobot/aloha_sim_insertion_scripted_image, lerobot/aloha_sim_transfer_cube_human_image, lerobot/aloha_sim_transfer_cube_scripted_image
- Notes: Data licence not stated. The ACT repo LICENSE is MIT (Copyright 2023 Tony Z. Zhao) and covers code; the demos are a Google Drive folder with no terms. The LeRobot 'mit' tag reflects the code licence, not a data grant. Scripted episodes can be regenerated with the MIT code.

### Mobile ALOHA datasets (`mobile_aloha`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: medium
- Statement: "[no licence statement; the project page links 'Datasets' to a Google Drive folder]" (https://mobile-aloha.github.io/)
- Repo-id regex: `(^|/)aloha[-_]mobile[-_]`
- Matched Hub ids (6): lerobot/aloha_mobile_shrimp, lerobot/aloha_mobile_cabinet, lerobot/aloha_mobile_wipe_wine, lerobot/aloha_mobile_chair, lerobot/aloha_mobile_wash_pan, lerobot/aloha_mobile_elevator
- Notes: Data licence not stated on the project page or in MarkFzp/mobile-aloha (MIT LICENSE for code).

### ALOHA static datasets (as ported by LeRobot) (`aloha_static`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: low
- Statement: "[no licence statement; the ALOHA page offers no dataset download]" (https://tonyzhaozh.github.io/aloha/)
- Repo-id regex: `(^|/)aloha[-_]static[-_]`
- Matched Hub ids (15): lerobot/aloha_static_coffee, lerobot/aloha_static_cups_open, lerobot/aloha_static_battery, lerobot/aloha_static_towel, lerobot/aloha_static_candy, lerobot/aloha_static_screw_driver, lerobot/aloha_static_thread_velcro, lerobot/aloha_static_vinh_cup, lerobot/aloha_static_tape, lerobot/aloha_static_fork_pick_up, lerobot/aloha_static_ziploc_slide, lerobot/aloha_static_vinh_cup_left, lerobot/aloha_static_pingpong_test, lerobot/aloha_static_pro_pencil, lerobot/aloha_static_coffee_new
- Notes: LeRobot cards cite the ALOHA paper/page, which publishes no dataset or terms; the static ALOHA demos appear to be the co-training data distributed with the Mobile ALOHA Drive folder (unconfirmed). tonyzhaozh/aloha is MIT for code. LeRobot mirrors are inconsistent among themselves (mit and apache-2.0).

### Push-T (Diffusion Policy, Chi et al.) (`pusht_diffusion_policy`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: medium
- Statement: "This repository is released under the MIT license." (https://github.com/real-stanford/diffusion_policy)
- Repo-id regex: `(^|/)pusht(_image|_keypoints)?$`
- Matched Hub ids (3): lerobot/pusht, lerobot/pusht_image, lerobot/pusht_keypoints
- Notes: Data licence not stated. The quoted MIT licence covers the repository (code); training data is a separate download (diffusion-policy.cs.columbia.edu/data/training/pusht.zip) with no terms on the project page. LeRobot's 'mit' tag reflects the code licence.

### xArm lift/push offline datasets (FOWM, Feng et al.) (`xarm_fowm`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: medium
- Statement: "This repository is licensed under the MIT license." (https://github.com/fyhMer/fowm)
- Repo-id regex: `(^|/)xarm[-_](lift|push)[-_]medium`
- Matched Hub ids (8): lerobot/xarm_lift_medium, lerobot/xarm_lift_medium_replay, lerobot/xarm_push_medium, lerobot/xarm_push_medium_replay, lerobot/xarm_lift_medium_image, lerobot/xarm_push_medium_image, lerobot/xarm_lift_medium_replay_image, lerobot/xarm_push_medium_replay_image
- Notes: xarm_lift_medium(_replay)/xarm_push_medium(_replay) are the offline datasets released with FOWM (Google Drive links in the README). The MIT statement covers the repository; no data licence is stated. LeRobot cards point to the TD-MPC homepage, but the datasets come from FOWM.

### UMI in-the-wild cup arrangement (`umi_cup_in_the_wild`)

- Licence: **not stated**; commercial training: unclear; redistribution: unclear; restrictions: none recorded; confidence: medium
- Statement: "This repository is released under the MIT license." (https://github.com/real-stanford/universal_manipulation_interface)
- Repo-id regex: `umi[-_]cup[-_]in[-_]the[-_]wild`
- Matched Hub ids (1): lerobot/umi_cup_in_the_wild
- Notes: The MIT statement covers code; the project page and README give no data licence.

### Meta-World MT50 demos (LeRobot) (`metaworld_mt50_lerobot`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: low
- Statement: "We provide a LeRobot-ready dataset for Meta-World MT50 on the HF Hub" (https://huggingface.co/docs/lerobot/metaworld)
- Repo-id regex: `metaworld[-_]?mt50`
- Matched Hub ids (1): lerobot/metaworld_mt50
- Notes: No independent upstream dataset: Hugging Face's own docs present lerobot/metaworld_mt50 as their release, and its card tags apache-2.0, which is LeRobot's template default. Who generated it and how is not documented. The Meta-World benchmark code is MIT.

### RoboCasa / RoboCasa365 (`robocasa`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "Assets and Datasets: CC BY 4.0" (https://github.com/robocasa/robocasa#license)
- Repo-id regex: `robocasa`
- Matched Hub ids (9): ember-lab-berkeley/robocasa365-pretrain-mg, ember-lab-berkeley/robocasa365-pretrain-composite, ember-lab-berkeley/robocasa365-target-composite-unseen, infope/robocasa, ember-lab-berkeley/robocasa365-target-composite-seen, ember-lab-berkeley/robocasa365-target-atomic, lerobot/robocasa_target_human_unified, ember-lab-berkeley/robocasa365-pretrain-atomic, Voxel51/robocasa-MG_100
- Notes: Official README: code MIT, assets and datasets CC BY 4.0. The ember-lab-berkeley cards call themselves a 'Mirror of RoboCasa365' and claim 'MIT, inherited from upstream RoboCasa365', which mixes up the code licence and the data licence.

### BEHAVIOR-1K 2025 challenge demos (`behavior_1k_2025_challenge`)

- Licence: **MIT**; commercial training: unclear; redistribution: yes; restrictions: attribution; confidence: medium
- Statement: "license: mit" (https://huggingface.co/datasets/behavior-1k/2025-challenge-demos)
- Repo-id regex: `behavior[-_]?1k|(^|/)2025[-_]challenge[-_]demos$`
- Matched Hub ids (4): behavior-1k/2025-challenge-demos, Dario-Shit4/behavior-1k_2025-challenge-demos, k1000dai/behavior1k-only-rgb, elonelonelon/2025-challenge-demos
- Notes: The authors' own HF card tags MIT, but only as a tag. The demos are renders of the BEHAVIOR Data Bundle, whose EULA (OmniGibson asset_utils.py) says 'Data may only be used for non-commercial academic research.' and forbids redistribution. Commercial training is therefore unclear.

### ABC-130K (XDOF) (`abc_130k`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "Released under the Apache License 2.0. See `LICENSE`." (https://huggingface.co/datasets/XDOF/ABC-130k)
- Repo-id regex: `abc[-_]?130k`
- Matched Hub ids (6): lerobot/abc_130k_v3_train, typoverflow/abc_130k_train, lerobot/abc_130k_v3_smoke, LinhanWang/ABC-130k-lerobot, lerobot/abc_130k_v3_val, huzheyuan/abc130k_v3_train
- Notes: Official XDOF card. The upstream is gated (contact-sharing click-through), but the mirrors are ungated. That is not a licence conflict.

### L2D (Learning to Drive, Yaak + Hugging Face) (`l2d`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "license: apache-2.0" (https://huggingface.co/datasets/yaak-ai/L2D)
- Repo-id regex: `(^|/)l2d([-_]v\d+)?$`
- Matched Hub ids (2): yaak-ai/L2D, yaak-ai/L2D-v3
- Notes: The original authors' card, but the tag sits in the stock LeRobot template, and apache-2.0 is the push_to_hub default. No LICENSE file, and neither the blog post nor yaak.ai states terms. Street-scene video may raise GDPR issues that are separate from copyright. yaak-ai/lerobot-driving-school is NOT L2D (a 10-episode 2023 ActionGPT sample, same org) and is deliberately not matched.

### Unitree robotics datasets (unitreerobotics/*) (`unitree_g1_datasets`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "license: apache-2.0" (https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset)
- Repo-id regex: `^unitreerobotics/`
- Matched Hub ids (37): unitreerobotics/G1_Dex3_ToastedBread_Dataset, unitreerobotics/G1_Dex1_Stack_Block, unitreerobotics/G1_Dex1_Wipe_Table, unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine, unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher, unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks_v0, unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket_v0, unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket, unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset, unitreerobotics/G1_WBT_Inspire_Put_Drinks_Into_Fridge_v0, unitreerobotics/G1_WBT_Brainco_Pickup_Pillow, unitreerobotics/G1_Dex1_Fold_Towel, unitreerobotics/G1_WBT_Inspire_Clean_The_Living_Room, unitreerobotics/G1_WBT_Dex1_Put_Clothes_into_Washing_Machine, unitreerobotics/G1_WBT_Brainco_Pick_Up_Medicine_v0, ... (22 more; full list in JSON)
- Notes: The rightsholder's own cards, but using the stock LeRobot template tag (possibly a default). Only Unitree's own org matches. Third-party data recorded on G1 robots (e.g. RoboCOIN/Unitree_G1edu_*, aarontung/g1-*, mncai/G1_*) is a different dataset.

### RoboCOIN (BAAI / FlagOpen) (`robocoin`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications, signed-agreement; confidence: high
- Statement: "This dataset is released under the apache-2.0 license." (https://huggingface.co/datasets/RoboCOIN/Cobot_Magic_make_hamburger)
- Repo-id regex: `^robocoin/`
- Matched Hub ids (230): RoboCOIN/Cobot_Magic_make_hamburger, RoboCOIN/Cobot_Magic_pull_zipper, RoboCOIN/AI2_Alphabot_2_insert_hose_into_hole, RoboCOIN/AgiBot-g1_storage_item_e, RoboCOIN/Agilex_Cobot_Magic_heat_burger, RoboCOIN/Cobot_Magic_pour_drink, RoboCOIN/Cobot_Magic_take_out_the_bread, RoboCOIN/Cobot_Magic_pot_storage_steamer, RoboCOIN/Cobot_Magic_pour_water_a, RoboCOIN/Cobot_Magic_clean_blackboard, RoboCOIN/Cobot_Magic_desktop_organization, RoboCOIN/AI2_Alphabot_2_insert_water_pipe, RoboCOIN/Agilex_Cobot_Magic_fold_clothes, RoboCOIN/Agilex_Cobot_Magic_basket_storage_peach, RoboCOIN/Cobot_Magic_steamer_storage_dumpling, ... (215 more; full list in JSON)
- Notes: Explicit text on the official cards. A click-through gate asks users to cite the paper and not to harm human subjects. The repo names carry robot brands (AgiBot, Galaxea, Unitree, Galbot, Leju...), but these are RoboCOIN data, not AgiBot World or Galaxea Open-World.

### RH20T (`rh20t`)

- Licence: **LicenseRef-RH20T-split-CC-BY-SA-4.0-and-CC-BY-NC-4.0**; commercial training: unclear; redistribution: yes; restrictions: attribution, indicate-modifications, share-alike, non-commercial; confidence: high
- Statement: "Commercial use of the RH20T-NC subset or models trained on it is not allowed." (https://rh20t.github.io/)
- Repo-id regex: `rh20t`
- Matched Hub ids (5): robot-lev/rh20t_cfg1, Voxel51/rh20t-cfg1-28ep, robot-lev/rh20t_cfg2, robot-lev/rh20t_cfg5, robot-lev/rh20t_cfg3
- Notes: Split by scene: RH20T-C (scene_0001-0005) is CC BY-SA 4.0 and allows commercial use; RH20T-NC (scene_0006-0010) is CC BY-NC 4.0 (the page's link points to the BY-NC-SA deed). The robot-lev mirrors keep this per-episode split. rh20t_api code is MIT.

### RoboSet (RoboHive / RoboPen) (`roboset`)

- Licence: **MIT**; commercial training: yes; redistribution: yes; restrictions: attribution; confidence: low
- Statement: "This work is licensed under the MIT-License." (https://robopen.github.io/roboset/)
- Repo-id regex: `(^|/)roboset`
- Matched Hub ids: none in the top-600 list
- Notes: Website footer statement; it is unclear whether it covers the data or only the site. robohive code is Apache-2.0.

### Galaxea Open-World Dataset (`galaxea_open_world`)

- Licence: **CC-BY-NC-SA-4.0**; commercial training: no; redistribution: yes; restrictions: attribution, indicate-modifications, non-commercial, share-alike, signed-agreement; confidence: high
- Statement: "All the data and code within this repo are under CC BY-NC-SA 4.0" (https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
- Repo-id regex: `galaxea[-_]?open[-_]?world`
- Matched Hub ids: none in the top-600 list
- Notes: 'Galaxea Community License Agreement', gated. Repo ids naming a Galaxea robot (e.g. RoboCOIN/Galaxea_R1_Lite_*) are not this dataset.

### AgiBot Digital World (`agibot_digital_world`)

- Licence: **CC-BY-NC-SA-4.0**; commercial training: no; redistribution: yes; restrictions: attribution, indicate-modifications, non-commercial, share-alike; confidence: high
- Statement: "All the data and code within this repo are under CC BY-NC-SA 4.0" (https://huggingface.co/datasets/agibot-world/AgiBotDigitalWorld)
- Repo-id regex: `agibot[-_]?digital[-_]?world`
- Matched Hub ids: none in the top-600 list
- Notes: Official card. This is separate from AgiBot World (already covered).

### DexMimicGen datasets (`dexmimicgen`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "the datasets are released under CC-BY 4.0" (https://github.com/NVlabs/dexmimicgen#license)
- Repo-id regex: `dexmimicgen`
- Matched Hub ids: none in the top-600 list
- Notes: The code is under the NVIDIA Source Code License (non-commercial); only the datasets are CC BY 4.0.

### MimicGen datasets (`mimicgen`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "the datasets are released under CC-BY 4.0" (https://github.com/NVlabs/mimicgen#license)
- Repo-id regex: `(^|[/_-])mimicgen`
- Matched Hub ids: none in the top-600 list
- Notes: The README changelog says that on 04/04/2024 the 'Dataset license changed to CC-BY 4.0, which is less restrictive'. Copies taken before then may carry the older terms. Code: NVIDIA Source Code License.

### robomimic datasets (`robomimic`)

- Licence: **MIT**; commercial training: yes; redistribution: yes; restrictions: attribution; confidence: medium
- Statement: "license: mit" (https://huggingface.co/datasets/robomimic/robomimic_datasets)
- Repo-id regex: `robomimic`
- Matched Hub ids: none in the top-600 list
- Notes: Tag only, on the official robomimic org card (amandlek/robomimic also tags MIT). The docs site and README state no data licence. The code is MIT.

### SmolVLA SO-100/SO-101 community datasets (lerobot/svla_*) (`lerobot_svla_so10x`)

- Licence: **Apache-2.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: medium
- Statement: "license: apache-2.0" (https://huggingface.co/datasets/lerobot/svla_so100_pickplace)
- Repo-id regex: `(^|/)svla[-_]so10[01][-_]`
- Matched Hub ids (4): lerobot/svla_so100_pickplace, lerobot/svla_so100_sorting, lerobot/svla_so100_stacking, lerobot/svla_so101_pickplace
- Notes: Hugging Face's own first-party release, so the tag is the creator's declaration (though it is the LeRobot template default).

### OpenDriveLab Kai0 (chi-zero) data (`kai0`)

- Licence: **CC-BY-NC-SA-4.0**; commercial training: no; redistribution: yes; restrictions: attribution, indicate-modifications, non-commercial, share-alike; confidence: high
- Statement: "The data and checkpoint are under CC BY-NC-SA 4.0." (https://github.com/OpenDriveLab/kai0)
- Repo-id regex: `(^|/)kai0`
- Matched Hub ids (1): OpenDriveLab-org/Kai0
- Notes: Code is Apache-2.0. The HF card's licence paragraph is an empty template, but its YAML tag (cc-by-nc-sa-4.0) agrees with the GitHub README.

### BitRobot HIW-500 (Humanoids in the Wild) (`hiw_500`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "The public HIW-500 dataset is released under the CC BY 4.0 license" (https://huggingface.co/datasets/BitRobot/HIW-500)
- Repo-id regex: `hiw[-_]?500`
- Matched Hub ids (1): BitRobot/HIW-500-LeRobot
- Notes: Official card; the LeRobot conversion BitRobot/HIW-500-LeRobot comes from the same org and carries the same text.

### Config HABIT (`habit`)

- Licence: **CC-BY-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications; confidence: high
- Statement: "license: cc-by-4.0" (https://huggingface.co/datasets/configinc/HABIT)
- Repo-id regex: `(^|/)habit$`
- Matched Hub ids (1): configinc/HABIT
- Notes: Official card by the authors (Config, KAIST), with a YAML tag plus a 'License: CC BY 4.0' badge. The project page states no licence.

### TacVerse (Xense Robotics) (`tacverse`)

- Licence: **CC-BY-SA-4.0**; commercial training: yes; redistribution: yes; restrictions: attribution, indicate-modifications, share-alike; confidence: high
- Statement: "Released under CC BY-SA 4.0." (https://huggingface.co/datasets/TacVerse/TacVerse)
- Repo-id regex: `^tacverse/`
- Matched Hub ids (1): TacVerse/opendata
- Notes: The landing card adds that derivative datasets must use the same CC BY-SA 4.0 terms.

## Caveats

- **Tool-default tags.** `apache-2.0` is LeRobot's `push_to_hub` default, so a bare `license: apache-2.0` in the stock LeRobot card template is weak evidence of intent, even on a rightsholder's own repo (L2D, Unitree, lerobot/svla_*, lerobot/metaworld_mt50). Those entries are medium or low. ABC-130K and RoboCOIN carry explicit licence sentences and are high.
- **Code licences are not data licences.** ACT/ALOHA, Mobile ALOHA, Push-T (Diffusion Policy), xArm (FOWM), UMI, and many OXE components publish an MIT/BSD/Apache LICENSE for code only. The LeRobot `mit` tags on lerobot/pusht, lerobot/aloha_sim_*, lerobot/xarm_* and lerobot/stanford_kuka_multimodal_dataset reproduce those code licences.
- **ALOHA 2** (aloha-2.github.io) released hardware designs and a MuJoCo model, not a dataset; no entry was needed. Repos such as physical-intelligence/aloha_pen_uncap_diverse or Voxel51/aloha_pen_uncap were recorded on ALOHA hardware but are not ALOHA project datasets, and the ALOHA regexes deliberately do not match them.
- **Brand names are not dataset names.** RoboCOIN/AgiBot-g1_*, RoboCOIN/Galaxea_R1_Lite_*, RoboCOIN/Unitree_G1edu_* and third-party G1 repos are excluded from AgiBot World, Galaxea Open-World and Unitree. yaak-ai/lerobot-driving-school is a different (2023) Yaak dataset, not L2D.
- **BEHAVIOR-1K.** The authors tag the demos MIT, but the demos render the BEHAVIOR Data Bundle, whose EULA restricts it to non-commercial academic research and forbids redistribution. Commercial training is recorded as unclear.
- **Berkeley Fanuc.** The raw page could not be retrieved (sites.google.com is blocked from this sandbox). The 'non-commercial' tail of the licence sentence comes from two independent WebFetch summaries and should be confirmed by hand before acting on it.
- **RH20T** is split by scene. The commercial flag is null at dataset level because it depends on the episode (scene_0001-0005 CC BY-SA 4.0; scene_0006-0010 CC BY-NC 4.0).
- **MimicGen** changed its dataset licence to CC BY 4.0 on 2024-04-04. Copies taken earlier may carry older terms.
- **Licence-laundering chains are visible.** saaduddinM/rbo_oxe_base_language_table_lerobot declares 'apache-2.0 (inherits from source)', meaning from IPEC-COMMUNITY/language_table_lerobot, which in turn mislabels the CC BY 4.0 OXE data. The ember-lab-berkeley RoboCasa365 cards claim 'MIT, inherited from upstream RoboCasa365' because they read the code licence as the data licence.
- Not verified (time): the OXE per-dataset spreadsheet; the Kaggle pages for other OXE components; the OSF licence field for NYU Franka Play; and the standalone releases of UCSD, IAMLab, DLR, Imperial, USC and NYU-door data.

## Sources

- https://functional-manipulation-benchmark.github.io/
- https://github.com/NVlabs/dexmimicgen#license
- https://github.com/NVlabs/mimicgen#license
- https://github.com/OpenDriveLab/kai0
- https://github.com/StanfordVL/roboturk_real_dataset
- https://github.com/clvrai/clvr_jaco_play_dataset
- https://github.com/fyhMer/fowm
- https://github.com/google-deepmind/open_x_embodiment#license-and-disclaimer
- https://github.com/real-stanford/diffusion_policy
- https://github.com/real-stanford/universal_manipulation_interface
- https://github.com/robocasa/robocasa#license
- https://github.com/tonyzhaozh/act
- https://huggingface.co/datasets/BitRobot/HIW-500
- https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset
- https://huggingface.co/datasets/RoboCOIN/Cobot_Magic_make_hamburger
- https://huggingface.co/datasets/TacVerse/TacVerse
- https://huggingface.co/datasets/XDOF/ABC-130k
- https://huggingface.co/datasets/agibot-world/AgiBotDigitalWorld
- https://huggingface.co/datasets/behavior-1k/2025-challenge-demos
- https://huggingface.co/datasets/configinc/HABIT
- https://huggingface.co/datasets/lerobot/svla_so100_pickplace
- https://huggingface.co/datasets/robomimic/robomimic_datasets
- https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset
- https://huggingface.co/datasets/yaak-ai/L2D
- https://huggingface.co/docs/lerobot/metaworld
- https://mobile-aloha.github.io/
- https://rh20t.github.io/
- https://robopen.github.io/roboset/
- https://sites.google.com/berkeley.edu/fanuc-manipulation
- https://sites.google.com/view/cablerouting/data
- https://tonyzhaozh.github.io/aloha/
- https://www.kaggle.com/datasets/google/bc-z-robot
- https://www.kaggle.com/datasets/oiermees/taco-robot
- https://raw.githubusercontent.com/StanfordVL/BEHAVIOR-1K/HEAD/OmniGibson/omnigibson/utils/asset_utils.py (BEHAVIOR Data Bundle EULA)
- https://github.com/google-research/language-table, https://robotics-transformer1.github.io/, https://sites.google.com/view/bc-z/home (component pages with no data licence)
- https://clvrai.github.io/furniture-bench/, https://ut-austin-rpl.github.io/MUTEX/, https://toto-benchmark.org/, https://sites.google.com/view/berkeley-ur5/home, https://sites.google.com/view/visionandtouch, https://ut-austin-rpl.github.io/VIOLA/, https://play-to-policy.github.io/, https://play-fusion.github.io/, https://sites.google.com/view/hydra-il-2023, https://dobb-e.com/, https://robotic-pretrained-transformer.github.io/, https://tetexiao.com/projects/real-mvp, https://hshi74.github.io/robocook/, https://umi-gripper.github.io/, https://mobile-aloha.github.io/, https://tonyzhaozh.github.io/aloha/, https://aloha-2.github.io/, https://diffusion-policy.cs.columbia.edu/, https://robocasa.ai/, https://behavior.stanford.edu/challenge/dataset.html (checked; no data licence stated)