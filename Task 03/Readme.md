## Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

DevelopersHub Corporation — AI/ML Engineering Internship (Advanced Task 3)

## Objective

Real estate prices depend not only on structured attributes (bedrooms, bathrooms, square footage, location) but also on visual factors (kerb appeal, interior finish, natural light) that are only visible in photos. The objective of this task is to build a **multimodal regression model** that predicts a house's sale price by fusing:

- **Tabular features** — number of bedrooms, bathrooms, square footage, and zip code
- **Image features** — extracted with a Convolutional Neural Network (CNN) from photos of the house (bedroom, bathroom, kitchen, and frontal/exterior view)

Both feature streams are combined (**feature fusion**) into a single network, trained end-to-end to predict price, and evaluated using **MAE** and **RMSE**.

## Dataset

Public **Houses Dataset** (Ahmed & Moustafa, 2016) — purpose-built for this exact image + tabular housing-price task.

- **535 houses**
- 4 images per house (bathroom, bedroom, kitchen, frontal exterior)
- A text file (`HousesInfo.txt`) with: `bedrooms bathrooms area zipcode price`

Source: https://github.com/emanhamed/Houses-dataset

## Methodology / Approach

### 1. Environment Setup
Ran on Google Colab with GPU runtime enabled.

**TensorFlow version: 2.20.0**
**GPU available: []**

> Note: no GPU was actually allocated in this run (training ran on CPU), which is part of why training was slower per epoch than expected on Colab.

### 2. Dataset Loading
The dataset was cloned directly from GitHub — no manual download required.

**Cloning into 'Houses-dataset'...**
**remote: Enumerating objects: 2133, done.**
**remote: Counting objects: 100% (2133/2133), done.**
**remote: Compressing objects: 100% (2133/2133), done.**
**remote: Total 2133 (delta 0), reused 2129 (delta 0), pack-reused 0 (from 0)**
**Receiving objects: 100% (2133/2133), 176.26 MiB | 29.78 MiB/s, done.**
**Updating files: 100% (2144/2144), done.**
**['139_frontal.jpg', '379_frontal.jpg', '400_frontal.jpg', '447_kitchen.jpg', '490_kitchen.jpg', '415_frontal.jpg', '25_bathroom.jpg', '157_kitchen.jpg', '323_bathroom.jpg', '494_bedroom.jpg']**

Tabular data loaded into a DataFrame:

**(535, 6)**

| bedrooms | bathrooms | area | zipcode | price  | house_id |
|----------|-----------|------|---------|--------|----------|
| 4        | 4.0       | 4053 | 85255   | 869500 | 1        |
| 4        | 3.0       | 3343 | 36372   | 865200 | 2        |
| 3        | 4.0       | 3923 | 85266   | 889000 | 3        |
| 5        | 5.0       | 4022 | 85262   | 910000 | 4        |
| 3        | 4.0       | 4116 | 85266   | 971226 | 5        |

### 3. Preprocessing

**Tabular preprocessing:**
- `zipcode` is high-cardinality, so instead of one-hot encoding (which would explode dimensionality on only 535 rows) we used **target encoding**: each zipcode replaced by the mean training-set house price for that zipcode, fit only on the training split to avoid data leakage.
- `bedrooms`, `bathrooms`, `area`, and the encoded zipcode were scaled to [0, 1] with `MinMaxScaler`.
- `price` (the target) was also scaled to [0, 1] for stable training, then inverted back to dollars for evaluation.

**Image preprocessing:**
Each house's 4 photos were combined into a single 2x2 montage (bathroom, bedroom top row; frontal, kitchen bottom row) so the CNN sees the whole house in one image, resized to 224x224 for MobileNetV2.

**Image tensor shape: (535, 224, 224, 3)**

Train/test split (80/20):

**Train tabular: (428, 4) | Train images: (428, 224, 224, 3)**
**Test tabular: (107, 4) | Test images: (107, 224, 224, 3)**

### 4. Model Development

**Why transfer learning?** With only ~535 houses (428 for training), training a CNN from scratch would overfit badly. We used **MobileNetV2 pretrained on ImageNet** as a frozen feature extractor, since it already knows general visual features (edges, textures, shapes, materials) that transfer well to house photos, and freezing it keeps trainable parameters small relative to dataset size.

**Architecture (feature fusion):**
- **Image branch:** MobileNetV2 (frozen, `include_top=False`, global average pooling) → Dense(128) → BatchNorm → Dropout → Dense(32)
- **Tabular branch:** Dense(16) → Dense(8) on the 4 scaled tabular features
- **Fusion:** Concatenate both branches → Dense(32) → Dropout → Dense(1, linear) to predict scaled price

**Model parameter summary:**

| | Params |
|---|---|
| Total params | 2,428,153 (9.26 MB) |
| Trainable params | 169,913 (663.72 KB) |
| Non-trainable params | 2,258,240 (8.61 MB) |

### 5. Training
Trained for up to 60 epochs (Adam optimizer, MSE loss, early stopping on validation loss). Loss steadily decreased across training:

Epoch 1/60  — loss: 0.4705, mae: 0.5343, val_loss: 0.0861, val_mae: 0.2320
Epoch 20/60 — loss: 0.1103, mae: 0.2582, val_loss: 0.0536, val_mae: 0.1853
Epoch 40/60 — loss: 0.0528, mae: 0.1785, val_loss: 0.0285, val_mae: 0.1314
**Epoch 60/60 — loss: 0.0286, mae: 0.1314, val_loss: 0.0166, val_mae: 0.0999**

## Key Results / Observations

Final evaluation on the held-out test set (values inverse-scaled back to real dollars):

**MAE:  $577,959.87**
**RMSE: $726,647.43**
**MAPE: 200.97%**

### Observations
- Training loss (scaled MSE/MAE) decreased smoothly and consistently over 60 epochs, and validation loss tracked training loss closely — showing the model was **not overfitting** in the scaled [0,1] space.
- However, on the real dollar scale the test-set error is large: **MAE ≈ $578K, RMSE ≈ $727K, MAPE ≈ 201%**. The predicted-vs-actual scatter plot confirms this — points are scattered widely around the ideal diagonal line rather than tightly following it, and the residual distribution is wide and not tightly centered at zero.
- This gap between low scaled-loss and poor real-dollar accuracy is a strong sign of **outliers/skew in the price distribution** dominating the MinMax-scaled target, plus the small dataset size (428 training houses) limiting how well the frozen-backbone model can generalize.
- **Possible improvements:** use a log-transform on price instead of MinMax scaling (housing prices are typically right-skewed), fine-tune the last few MobileNetV2 layers instead of leaving it fully frozen, try per-image (not just montage) feature extraction, and/or gather a larger dataset to reduce variance in the test metrics.

## How to Run
1. Open `Task3_Multimodal_Housing_Price_Prediction.ipynb` in Google Colab.
2. Set runtime to GPU: `Runtime > Change runtime type > GPU`.
3. Run all cells top to bottom — the notebook clones the dataset from GitHub automatically.

## Skills Demonstrated
- Multimodal machine learning (image + tabular fusion)
- Convolutional Neural Networks / transfer learning (MobileNetV2)
- Feature engineering (target encoding, scaling)
- Regression modeling and evaluation (MAE, RMSE, MAPE)
## Colab Notebook link: 
https://colab.research.google.com/drive/1ojmr1V5meSZt_ywwJnYh-IHcnmEjHE5r?usp=sharing
