# 圖表索引

全部 23 張圖皆自 notebook 的執行輸出擷取，未重新繪製。每張圖對應的程式碼與完整數據見括號內的 notebook。

## 資料準備

| 檔案 | 內容 | 來源 |
|---|---|---|
| `01_video_split_distribution.png` | FF++ 官方 train／val／test 的影片數分布 | [01](../notebooks/01_FFPP_Deepfakes_資料準備.ipynb) |
| `01_frame_distribution.png` | 抽幀後各 split 的 Real／Fake 圖片數 | [01](../notebooks/01_FFPP_Deepfakes_資料準備.ipynb) |
| `02_face_crop_success_rate.png` | YOLO 人臉裁切成功率（依 split 與類別） | [02](../notebooks/02_FFPP_Deepfakes_YOLO人臉裁切.ipynb) |

## CLIP 基準（ViT-B/16 凍結）

| 檔案 | 內容 | 來源 |
|---|---|---|
| `03_baseline_training_curve.png` | 訓練／驗證 loss 與驗證 AUC，8 epoch 早停 | [03](../notebooks/03_FFPP_CLIP_Baseline訓練.ipynb) |
| `03_baseline_confusion_matrix.png` | FF++ 測試集混淆矩陣（frame／video） | [03](../notebooks/03_FFPP_CLIP_Baseline訓練.ipynb) |
| `03_baseline_roc_ffpp.png` | FF++ 測試集 ROC 曲線 | [03](../notebooks/03_FFPP_CLIP_Baseline訓練.ipynb) |
| `04_baseline_hidf_confusion_matrix.png` | HiDF 混淆矩陣 | [04](../notebooks/04_CLIP_跨資料集測試.ipynb) |
| `04_baseline_hidf_roc.png` | HiDF ROC 曲線 | [04](../notebooks/04_CLIP_跨資料集測試.ipynb) |
| `04_baseline_celebdf_confusion_matrix.png` | Celeb-DF v2 混淆矩陣 | [04](../notebooks/04_CLIP_跨資料集測試.ipynb) |
| `04_baseline_celebdf_roc.png` | Celeb-DF v2 ROC 曲線 | [04](../notebooks/04_CLIP_跨資料集測試.ipynb) |
| `04_baseline_cross_dataset_auc.png` | 三個資料集的 AUC 對照長條圖 | [04](../notebooks/04_CLIP_跨資料集測試.ipynb) |

## CLIP Feature Adapter（對照實驗，負向結果）

| 檔案 | 內容 | 來源 |
|---|---|---|
| `05_feature_adapter_training_curve.png` | 驗證 loss 於第 3 epoch 反轉，5 epoch 早停 | [05](../notebooks/05_FFPP_CLIP_Adapter訓練.ipynb) |

因未通過驗證集門檻，本實驗未執行測試，故無混淆矩陣與 ROC。

## Forensics Adapter（ViT-L/14 凍結 + TinyViT）

| 檔案 | 內容 | 來源 |
|---|---|---|
| `07_adapter_training_curve.png` | 驗證 AUC 自第 1 epoch 起即接近上限，15 epoch 早停 | [07](../notebooks/07_ForensicsAdapter_完整訓練與測試.ipynb) |
| `07_adapter_confusion_matrix.png` | FF++ 測試集混淆矩陣（frame／video） | [07](../notebooks/07_ForensicsAdapter_完整訓練與測試.ipynb) |
| `07_adapter_roc_ffpp.png` | FF++ 測試集 ROC 曲線 | [07](../notebooks/07_ForensicsAdapter_完整訓練與測試.ipynb) |
| `08_adapter_hidf_confusion_matrix.png` | HiDF 混淆矩陣 | [08](../notebooks/08_ForensicsAdapter_跨資料集測試.ipynb) |
| `08_adapter_hidf_roc.png` | HiDF ROC 曲線 | [08](../notebooks/08_ForensicsAdapter_跨資料集測試.ipynb) |
| `08_adapter_celebdf_confusion_matrix.png` | Celeb-DF v2 混淆矩陣 | [08](../notebooks/08_ForensicsAdapter_跨資料集測試.ipynb) |
| `08_adapter_celebdf_roc.png` | Celeb-DF v2 ROC 曲線 | [08](../notebooks/08_ForensicsAdapter_跨資料集測試.ipynb) |
| `08_adapter_cross_dataset_auc.png` | 三個資料集的 AUC 對照長條圖 | [08](../notebooks/08_ForensicsAdapter_跨資料集測試.ipynb) |

`04_baseline_cross_dataset_auc.png` 與 `08_adapter_cross_dataset_auc.png` 為同一組資料集在兩個模型下的結果，可直接對照。其餘 `04_` 與 `08_` 開頭的同名圖表亦然。

## 門檻校準

| 檔案 | 內容 | 來源 |
|---|---|---|
| `09C_threshold_search_and_score_distribution.png` | 門檻搜尋曲線與校準集的機率分布 | [09C](../notebooks/09C_手機資料門檻校準與正式測試.ipynb) |
| `09C_formal_test_confusion_matrices.png` | 預設門檻 0.5 與校準門檻 0.463 的混淆矩陣對照 | [09C](../notebooks/09C_手機資料門檻校準與正式測試.ipynb) |
| `09C_formal_test_roc.png` | 正式測試集 ROC 曲線（AUC 不受門檻影響，僅繪製一次） | [09C](../notebooks/09C_手機資料門檻校準與正式測試.ipynb) |

## 未收錄的圖表

原 notebook 另有人臉影像的預覽圖與誤判樣本圖，因 FaceForensics++、Celeb-DF v2、HiDF 的授權條款限制影像再散布，且獨立測試集涉及肖像權，皆未收錄於公開版本。產生這些圖的程式碼保留在 notebook 中，可在本機重現。
