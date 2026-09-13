# Dataset Statistics Report

## Source and selection

- Class-folder root: `data/raw/PlantVillage`
- Selected classes: 13
- Valid selected images: 20113
- Corrupt images: 0
- Exact duplicate groups in selected source data: 14

## Split method

Deterministic class-stratified 70/15/15 split (seed 42). Exact duplicate groups are assigned as a unit. Near-duplicate detection is not included and remains a limitation.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Pepper__bell___Bacterial_spot | 698 | 150 | 149 | 997 |
| Pepper__bell___healthy | 1035 | 222 | 221 | 1478 |
| Potato___Early_blight | 700 | 150 | 150 | 1000 |
| Potato___Late_blight | 700 | 150 | 150 | 1000 |
| Tomato_Bacterial_spot | 1489 | 319 | 319 | 2127 |
| Tomato_Early_blight | 700 | 150 | 150 | 1000 |
| Tomato_Late_blight | 1336 | 286 | 287 | 1909 |
| Tomato_Leaf_Mold | 666 | 143 | 143 | 952 |
| Tomato_Septoria_leaf_spot | 1240 | 266 | 265 | 1771 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1173 | 251 | 252 | 1676 |
| Tomato__Target_Spot | 983 | 211 | 210 | 1404 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 2246 | 481 | 481 | 3208 |
| Tomato_healthy | 1114 | 239 | 238 | 1591 |
| **Total** | **14080** | **3018** | **3015** | **20113** |

## Leakage checks

- File-path overlap: train/validation 0, train/test 0, validation/test 0.
- Exact-duplicate groups crossing partitions: 0.

## Preprocessing

All images use RGB tensor conversion and ImageNet normalization (mean 0.485/0.456/0.406; standard deviation 0.229/0.224/0.225) after resizing to 224×224. Training only uses a random crop with scale 0.85–1.00, horizontal flip, ±12° rotation, and modest brightness/contrast jitter. Validation and test transforms are deterministic.
