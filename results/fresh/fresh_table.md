
4B (seeds 0,1,2, 130 items x 8 samples)

| arm | area | area per seed | natural length | length per seed | acc 8,192 | cut-off at 8,192 |
|---|---|---|---|---|---|---|
| untouched | 0.199 | - | 7741 | - | 0.355 | 0.870 |
| warm-up | 0.225 | - | 7663 | - | 0.387 | 0.855 |
| ctrl | 0.285 | 0.266 / 0.296 / 0.294 | 7424 | 7488 / 7435 / 7348 | 0.475 | 0.786 |
| wide | 0.407 | 0.419 / 0.391 / 0.411 | 5981 | 5886 / 6187 / 5872 | 0.621 | 0.515 |
| wide1 | 0.395 | 0.403 / 0.378 / 0.405 | 6094 | 6079 / 6397 / 5807 | 0.608 | 0.525 |
| ctrlpm | 0.297 | 0.262 / 0.303 / 0.327 | 7325 | 7512 / 7308 / 7155 | 0.488 | 0.759 |

- H5 (confirmatory): wide1 - ctrl norm_auc_log2 = +0.1102 (seeds +0.1367, +0.0823, +0.1115; CI [0.0886, 0.1324]): supported
- H6 (confirmatory): wide1 - ctrl natural_len = -1329.2151 (seeds -1409, -1038, -1541; CI [-1585.4896, -1080.7941]): supported

Contrasts (rule verdicts):
- area wide - ctrl: +0.1220 CI [0.1009, 0.1442] seeds [0.1533, 0.0949, 0.1178]: effect, positive
- area wide1 - ctrl: +0.1102 CI [0.0886, 0.1324] seeds [0.1367, 0.0823, 0.1115]: effect, positive
- area wide - wide1: +0.0118 CI [0.0042, 0.0199] seeds [0.0165, 0.0126, 0.0063]: effect, positive
- area wide1 - ctrlpm: +0.0983 CI [0.0782, 0.1190] seeds [0.1408, 0.0755, 0.0786]: effect, positive
- area wide - ctrlpm: +0.1101 CI [0.0908, 0.1307] seeds [0.1573, 0.0881, 0.0848]: effect, positive
- area ctrlpm - ctrl: +0.0119 CI [0.0064, 0.0176] seeds [-0.004, 0.0068, 0.033]: no effect
- area ctrl - untouched: +0.0866 CI [0.0684, 0.1059] seeds [0.0674, 0.0973, 0.095]: effect, positive
- area wide - untouched: +0.2086 CI [0.1736, 0.2452] seeds [0.2207, 0.1922, 0.2128]: effect, positive
- area wide1 - untouched: +0.1968 CI [0.1616, 0.2328] seeds [0.2041, 0.1796, 0.2065]: effect, positive
- area ctrlpm - untouched: +0.0985 CI [0.0777, 0.1200] seeds [0.0634, 0.1041, 0.128]: effect, positive
- area ctrl - warmup: +0.0600 CI [0.0458, 0.0753] seeds [0.0409, 0.0708, 0.0685]: effect, positive
- area wide - warmup: +0.1820 CI [0.1505, 0.2151] seeds [0.1941, 0.1657, 0.1862]: effect, positive
- area wide1 - warmup: +0.1702 CI [0.1384, 0.2030] seeds [0.1776, 0.1531, 0.18]: effect, positive
- area ctrlpm - warmup: +0.0720 CI [0.0557, 0.0891] seeds [0.0368, 0.0776, 0.1014]: effect, positive
- area warmup - untouched: +0.0265 CI [0.0170, 0.0363] seeds [0.0265]: descriptive only: one readout per side, the 3-seed rule does not apply
- natural length wide - ctrl: -1442.0478 CI [-1721.0718, -1174.7763] seeds [-1601.8885, -1248.5702, -1475.6846]: effect, negative
- natural length wide1 - ctrl: -1329.2151 CI [-1585.4896, -1080.7941] seeds [-1409.0394, -1038.0202, -1540.5856]: effect, negative
- natural length wide - wide1: -112.8327 CI [-168.7912, -61.0420] seeds [-192.849, -210.55, 64.901]: no effect
- natural length wide1 - ctrlpm: -1230.6327 CI [-1466.5853, -1004.2306] seeds [-1433.35, -910.9346, -1347.6135]: effect, negative
- natural length wide - ctrlpm: -1343.4654 CI [-1601.9113, -1098.1952] seeds [-1626.199, -1121.4846, -1282.7125]: effect, negative
- natural length ctrlpm - ctrl: -98.5824 CI [-146.0898, -56.8478] seeds [24.3106, -127.0856, -192.9721]: no effect
- natural length ctrl - untouched: -317.6670 CI [-422.3699, -222.7012] seeds [-253.3798, -305.9837, -393.6375]: effect, negative
- natural length wide - untouched: -1759.7147 CI [-2111.4138, -1427.2498] seeds [-1855.2683, -1554.5538, -1869.3221]: effect, negative
- natural length wide1 - untouched: -1646.8821 CI [-1976.9924, -1332.8330] seeds [-1662.4192, -1344.0038, -1934.2231]: effect, negative
- natural length ctrlpm - untouched: -416.2494 CI [-546.7854, -296.7033] seeds [-229.0692, -433.0692, -586.6096]: effect, negative
- natural length ctrl - warmup: -239.9699 CI [-331.1986, -157.4589] seeds [-175.6827, -228.2865, -315.9404]: effect, negative
- natural length wide - warmup: -1682.0176 CI [-2016.7425, -1361.7786] seeds [-1777.5712, -1476.8567, -1791.625]: effect, negative
- natural length wide1 - warmup: -1569.1849 CI [-1883.1756, -1268.5731] seeds [-1584.7221, -1266.3067, -1856.526]: effect, negative
- natural length ctrlpm - warmup: -338.5522 CI [-456.6793, -231.8392] seeds [-151.3721, -355.3721, -508.9125]: effect, negative
- natural length warmup - untouched: -77.6971 CI [-128.8587, -31.8690] seeds [-77.6971]: descriptive only: one readout per side, the 3-seed rule does not apply

Per source (descriptive, area / natural length seed means):
- aime2024 (30 items): untouched 0.109; ctrl 0.180 / 7942; wide 0.305 / 6727; wide1 0.302 / 6749; ctrlpm 0.198 / 7810
- aime2025 (30 items): untouched 0.109; ctrl 0.175 / 7845; wide 0.267 / 6754; wide1 0.259 / 6866; ctrlpm 0.182 / 7730
- aime2026 (30 items): untouched 0.059; ctrl 0.132 / 8078; wide 0.259 / 7180; wide1 0.242 / 7255; ctrlpm 0.139 / 8069
- amc2023 (40 items): untouched 0.438; ctrl 0.561 / 6227; wide 0.700 / 3944; wide1 0.682 / 4154; ctrlpm 0.576 / 6098

2B (seeds 0,1,2,3,4,5, 130 items x 8 samples)

| arm | area | area per seed | natural length | length per seed | acc 8,192 | cut-off at 8,192 |
|---|---|---|---|---|---|---|
| untouched | 0.067 | - | 8083 | - | 0.122 | 0.969 |
| ctrl | 0.174 | 0.183 / 0.173 / 0.167 / 0.176 / 0.168 / 0.177 | 7704 | 7680 / 7680 / 7741 / 7685 / 7734 / 7702 | 0.309 | 0.868 |
| wide | 0.196 | 0.198 / 0.187 / 0.199 / 0.171 / 0.206 / 0.217 | 7397 | 7281 / 7463 / 7319 / 7659 / 7460 / 7199 | 0.337 | 0.799 |
| wide1 | 0.186 | 0.179 / 0.202 / 0.184 / 0.187 / 0.187 / 0.175 | 7578 | 7590 / 7458 / 7625 / 7585 / 7504 / 7707 | 0.322 | 0.833 |

- H5 (secondary replication (H5-2B)): wide1 - ctrl norm_auc_log2 = +0.0117 (seeds -0.0033, +0.0286, +0.0176, +0.0107, +0.0191, -0.0023; CI [0.0063, 0.0177]): not supported (no effect by the rule)
- H6 (secondary replication (H6-2B)): wide1 - ctrl natural_len = -125.6575 (seeds -90, -222, -117, -100, -230, +5.4413; CI [-173.4529, -82.6792]): not supported (no effect by the rule)

Contrasts (rule verdicts):
- area wide - ctrl: +0.0223 CI [0.0152, 0.0302] seeds [0.0152, 0.0137, 0.0323, -0.005, 0.0376, 0.0399]: no effect
- area wide1 - ctrl: +0.0117 CI [0.0063, 0.0177] seeds [-0.0033, 0.0286, 0.0176, 0.0107, 0.0191, -0.0023]: no effect
- area wide - wide1: +0.0106 CI [0.0058, 0.0160] seeds [0.0185, -0.0148, 0.0147, -0.0157, 0.0185, 0.0422]: no effect
- area ctrl - untouched: +0.1073 CI [0.0786, 0.1388] seeds [0.116, 0.1067, 0.1, 0.1093, 0.1012, 0.1103]: effect, positive
- area wide - untouched: +0.1296 CI [0.0962, 0.1663] seeds [0.1312, 0.1205, 0.1323, 0.1043, 0.1388, 0.1502]: effect, positive
- area wide1 - untouched: +0.1190 CI [0.0879, 0.1533] seeds [0.1127, 0.1353, 0.1176, 0.12, 0.1204, 0.108]: effect, positive
- natural length wide - ctrl: -306.8607 CI [-404.0915, -218.6966] seeds [-399.2606, -217.0923, -422.6125, -25.7885, -273.3048, -503.1058]: effect, negative
- natural length wide1 - ctrl: -125.6575 CI [-173.4529, -82.6792] seeds [-90.4548, -222.476, -116.7596, -99.9962, -229.7, 5.4413]: no effect
- natural length wide - wide1: -181.2032 CI [-242.8611, -125.8622] seeds [-308.8058, 5.3837, -305.8529, 74.2077, -43.6048, -508.5471]: no effect
- natural length ctrl - untouched: -379.1287 CI [-558.7964, -219.9933] seeds [-402.6115, -402.5538, -341.4981, -397.9587, -349.2923, -380.8577]: effect, negative
- natural length wide - untouched: -685.9894 CI [-941.1215, -458.0525] seeds [-801.8721, -619.6462, -764.1106, -423.7471, -622.5971, -883.9635]: effect, negative
- natural length wide1 - untouched: -504.7862 CI [-716.2482, -317.5940] seeds [-493.0663, -625.0298, -458.2577, -497.9548, -578.9923, -375.4163]: effect, negative

Per source (descriptive, area / natural length seed means):
- aime2024 (30 items): untouched 0.005; ctrl 0.057 / 8131; wide 0.076 / 8034; wide1 0.069 / 8108
- aime2025 (30 items): untouched 0.013; ctrl 0.102 / 8071; wide 0.107 / 7849; wide1 0.100 / 7973
- aime2026 (30 items): untouched 0.002; ctrl 0.056 / 8169; wide 0.075 / 8086; wide1 0.067 / 8142
- amc2023 (40 items): untouched 0.202; ctrl 0.404 / 6758; wide 0.445 / 6063; wide1 0.427 / 6461
