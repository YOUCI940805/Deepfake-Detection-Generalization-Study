# Deepfake-Detection-Generalization-Study

在 FaceForensics++ 上訓練 deepfake 偵測模型，並檢驗它離開訓練分布之後會發生什麼事。

比較兩種以 CLIP 為基礎的方法（凍結線性基準、Forensics Adapter），在 FF++、HiDF、Celeb-DF v2 三個資料集上評估，並額外驗證跨 domain 部署時分類門檻的失效問題。

---

## 主要發現

**一、FF++ 內部測試無法區分方法優劣**

兩個模型在 FF++ Deepfakes 上的 video-level AUC 分別為 0.9805 與 0.9878，差距僅 0.007。Forensics Adapter 在第 1 個 epoch 的驗證 AUC（0.9965）就已高於基準訓練到最好的成績（0.9826），後續 14 個 epoch 只推進 0.0018。任務對這個架構而言接近飽和，內部指標已經測不出差異。

**二、方法差異只在跨資料集時顯現**

| 資料集 | CLIP 基準 | Forensics Adapter | 差距 |
|---|---|---|---|
| FF++ (video) | 0.9805 | 0.9878 | +0.007 |
| HiDF | 0.8083 | 0.9735 | **+0.165** |
| Celeb-DF v2 (frame) | 0.6770 | 0.7868 | **+0.110** |
| Celeb-DF v2 (video) | 0.6958 | 0.8063 | **+0.111** |

同樣兩個模型，在訓練分布內差 0.007，在訓練分布外差 0.11–0.17。Adapter 的價值不在準確率，在跨分布時的穩定性。

**三、預設門檻 0.5 在跨 domain 時失效，且與模型能力無關**

在一組獨立蒐集的測試影像上，Forensics Adapter 的 AUC 為 0.824，但輸出機率全部壓縮在 0.40–0.54 的狹窄區間內。

![門檻搜尋與機率分布](figures/09C_threshold_search_and_score_distribution.png)

右圖可見兩類的分數雖然可分，卻整體擠在 0.5 附近且中位數偏向 Real 側。排序能力與機率校準是兩件事——模型能正確排序樣本，不代表輸出的機率值可以直接用固定門檻切分。

以預設門檻 0.5 分類時：

| 門檻 | Accuracy | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| 0.5（預設） | 0.560 | 0.120 | 1.000 | 0.214 | 0.824 |
| 0.463（校準） | 0.745 | 0.670 | 0.788 | 0.724 | 0.824 |

![門檻對照](figures/09C_formal_test_confusion_matrices.png)

100 張 Fake 中有 88 張被漏掉，同時沒有任何一張 Real 被誤判。模型並未失去鑑別力（AUC 0.824），是門檻不再落在合適的切分位置。

僅用一組 100 張的獨立校準資料重新選擇門檻，將門檻從 0.5 移動 0.037 到 0.463，F1 從 0.214 提升到 0.724，完全不需重新訓練。

同樣的現象在 HiDF 與 Celeb-DF v2 上也可觀察到：recall 偏低而 precision 偏高，且換用更強的方法後依然存在。**門檻校準是與模型能力相對獨立的部署問題。**

---

## 實驗設定

**訓練資料**

FaceForensics++ c23，Real 取自 `original_sequences/youtube`，Fake 取自 `manipulated_sequences/Deepfakes`。不納入 Face2Face、FaceSwap、NeuralTextures，讓比較對象維持單一偽造方法。

沿用 FF++ 官方 `train.json`／`val.json`／`test.json`（720／140／140 部影片），每部影片等距抽取 16 幀，經 YOLOv11n-face 裁切主要人臉後，共 31,996 張。

**資料洩漏檢查**

Deepfakes 檔名 `000_003` 代表兩個來源身分。程式將 Fake 的兩個 ID 都展開，確認沒有任何身分同時出現在不同 split。檢查結果為 0 個跨 split 身分。

**評估模型**

| 模型 | 主幹 | 可訓練參數 | 說明 |
|---|---|---|---|
| CLIP Baseline | ViT-B/16（凍結） | 2,050 | LayerNorm + 線性分類頭 |
| CLIP Feature Adapter | ViT-B/16（凍結） | 133,762 | 512→128→512 殘差瓶頸，**對照實驗，結果為負向** |
| Forensics Adapter | ViT-L/14（凍結） | 7,546,823 | 官方完整實作 |

**外部測試集**

| 資料集 | 建立方式 | 規模 |
|---|---|---|
| HiDF<br>(Human-Indistinguishable<br>Deepfake Dataset) | 取自作者先前專題所建之 8:1:1 切分的 **val 子集**。本次模型未接觸其任何子集 | Real 313、Fake 313，共 626 張 |
| Celeb-DF v2 | 官方 `List_of_testing_videos.txt`，保留全部 178 部 Real，以 seed 42 從 340 部 Fake 抽出 178 部，每部取 3 幀 | Real 534、Fake 534，共 1,068 張 |
| 獨立測試集 | Real 為手機實拍、Fake 為網路蒐集，經 YOLO 裁切後分為互斥的校準集與測試集 | 校準 100 張、測試 200 張 |

建立過程見 [`00A`](notebooks/00A_HiDF資料集切分.ipynb) 與 [`00B`](notebooks/00B_CelebDFv2_平衡測試集建立.ipynb)。兩者的已知限制列於下方「已知限制」第五、六點。

**評估規範**

- 標籤固定 `Real=0`、`Fake=1`，程式會驗證對應關係。
- 最佳權重只依**驗證集** video-level AUC 選取，測試集不參與選模。
- 跨資料集測試不重新訓練、不選模、不調門檻，門檻固定 0.5。
- 09C 的門檻校準使用與正式測試互斥的獨立校準集，以 SHA-256 驗證兩組無重疊。

---

## 完整結果

### FF++ Deepfakes 內部測試

| 模型 | 層級 | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| CLIP Baseline | Frame | 0.8967 | 0.9173 | 0.8719 | 0.8940 | 0.9674 |
| CLIP Baseline | Video | 0.9214 | 0.9403 | 0.9000 | 0.9197 | 0.9805 |
| Forensics Adapter | Frame | 0.9371 | 0.9631 | 0.9089 | 0.9352 | 0.9814 |
| Forensics Adapter | Video | 0.9536 | 0.9774 | 0.9286 | 0.9524 | 0.9878 |

CLIP Feature Adapter 未執行測試，原因見下一節。

### 跨資料集測試

門檻固定 0.5，模型未見過這兩個資料集的任何樣本。

| 模型 | 資料集 | Accuracy | Recall | AUC | AUC 相對 FF++ 下降 |
|---|---|---|---|---|---|
| CLIP Baseline | HiDF | 0.6502 | 0.3291 | 0.8083 | 0.159 |
| CLIP Baseline | Celeb-DF v2 | 0.5665 | 0.2472 | 0.6770 | 0.290 |
| Forensics Adapter | HiDF | 0.8658 | 0.7540 | 0.9735 | 0.008 |
| Forensics Adapter | Celeb-DF v2 | 0.6751 | 0.3951 | 0.7868 | 0.195 |

<p align="center">
  <img src="figures/04_baseline_cross_dataset_auc.png" width="45%">
  <img src="figures/08_adapter_cross_dataset_auc.png" width="45%">
</p>

左為 CLIP 基準，右為 Forensics Adapter。

兩個外部資料集的困難程度差異很大：Forensics Adapter 在 HiDF 上幾乎沒有衰減（AUC 下降 0.008），在 Celeb-DF v2 上仍下降 0.195。不宜以單一資料集代表泛化能力。

### 對照實驗：CLIP Feature Adapter 的負向結果

在凍結 CLIP 的**最終輸出特徵**之後接一個 512→128→512 的瓶頸 Adapter，測試「只在特徵層動手腳」能否帶來增益。Adapter 最後一層零初始化，訓練開始前輸出與基準完全相同。

![Feature Adapter 訓練曲線](figures/05_feature_adapter_training_curve.png)

| Epoch | Train Frame AUC | Val Loss | Val Video AUC |
|---|---|---|---|
| 1 | 0.9853 | 0.2353 | 0.9803 |
| 2 | 0.9859 | 0.2423 | 0.9785 |
| 3 | 0.9867 | **0.2301** | **0.9816** |
| 4 | 0.9872 | 0.2473 | 0.9772 |
| 5 | 0.9887 | 0.2619 | 0.9746 |

訓練集 AUC 單調上升，驗證 loss 從第 3 個 epoch 起反轉，5 個 epoch 後早停。最佳驗證 AUC 0.9816，未超過基準的 0.9826。

**因未通過驗證集門檻，本實驗未動用測試集。** 選模規則在實驗開始前已固定，在明知未達基準的情況下仍去跑測試集，等同於期待測試集給出對自己有利的隨機波動。

**解釋**：CLIP 的最後一層特徵是為圖文語意對齊而訓練的，偽造痕跡所依賴的局部、高頻、空間性資訊在傳遞到這一層時已大幅衰減。在資訊已流失的地方增加參數量，只會讓模型更快記住訓練集。

這個負向結果支持了 Forensics Adapter 的設計動機：該方法從 CLIP 的第 1、8、15 層注入淺層特徵，正是為了取得最終特徵裡已經消失的那些資訊。

---

## 已知限制

**一、主幹規模不同，增益無法單獨歸因於 Adapter**

CLIP 基準使用 ViT-B/16，Forensics Adapter 使用 ViT-L/14。本研究測得的差距混合了「更大的主幹」與「Adapter 結構」兩個因素。若要分離，需補一個凍結 CLIP ViT-L/14 的線性基準。

**二、Boundary 監督在本次訓練中幾乎沒有生效**

Forensics Adapter 的 blending-boundary（x-ray）監督項，在 15 個 epoch 之間僅從 0.003940 降至 0.003876，降幅 1.6%，且第 2 個 epoch 之後完全平坦。

最可能的原因是 x-ray 目標圖中絕大多數像素為 0，MSE 只要整體預測接近 0 即可達到很低的值，有效梯度訊號極弱。這意味著本次訓練中模型的實際效能，未必來自論文所主張的邊界監督機制。

![Forensics Adapter 訓練曲線](figures/07_adapter_training_curve.png)

右圖的驗證 AUC 從第 1 個 epoch 起就貼在頂端，整段訓練幾乎是平的。

依權重分解第 1 個 epoch 的總 loss：

| 項目 | 原始值 | 權重 | 貢獻 | 佔比 |
|---|---|---|---|---|
| Adapter patch contrastive | 0.7328 | ×20 | 14.66 | 62.3% |
| CLIP contrastive | 0.7186 | ×10 | 7.19 | 30.6% |
| 分類 | 0.0893 | ×10 | 0.89 | 3.8% |
| Boundary MSE | 0.00394 | ×200 | 0.79 | 3.3% |

總 loss 曲線的起伏幾乎完全反映對比項的變化，與分類能力無關。

**三、實驗設定與原論文不同，不宣稱重現論文數字**

本研究使用每部影片 16 幀與 YOLOv11n-face 偵測，原論文使用 32 幀與 RetinaFace。硬體限制為 RTX 4070 12GB，批次大小 4（梯度累積 4 步，等效 16）。

**四、獨立測試集的資料來源存在系統性差異**

[`09A`](notebooks/09A_YOLO手機照片人臉裁切.ipynb)–[`09C`](notebooks/09C_手機資料門檻校準與正式測試.ipynb) 使用的 300 張測試影像中，Real 為手機實拍照片，Fake 為網路蒐集的合成人臉影像。兩類的來源不同，除真偽外還系統性地存在裝置、解析度、壓縮歷程與拍攝條件的差異。

因此**該組資料的 AUC 0.824 不作為泛化能力的估計**，無法排除模型部分利用了來源差異而非偽造痕跡。

該組資料的用途是**部署可行性檢查**：驗證輸出機率在面對訓練分布外影像時是否仍可沿用預設門檻。這個結論建立在機率分布的形狀上，與 AUC 的絕對數值無關，因此不受上述限制影響。

**五、三個測試集的人臉前處理並不一致**

| 資料集 | 人臉裁切 | 補正方形 | margin | 偵測門檻 |
|---|---|---|---|---|
| FF++（訓練） | YOLOv11n-face | 是 | 0.30 | 0.50 |
| HiDF | 無，直接使用原始圖片 | — | — | — |
| Celeb-DF v2 | YOLOv11n-face | **否** | 0.20 | 0.60 |
| 獨立測試集 | YOLOv11n-face | 是 | 0.15 | 0.50 |

Celeb-DF v2 的裁切在擴張人臉框後未補成正方形即縮放為 224×224，長寬比非 1:1 的人臉會被壓縮變形，而訓練資料的人臉比例未失真。此差異無法排除為 Celeb-DF v2 表現較低（AUC 0.787）的部分原因，不應將該結果完全歸因於資料集本身的困難度。

HiDF 則完全未經人臉裁切，但取得最佳的跨資料集成績（AUC 0.9735）。因此前處理不一致與效能高低之間並非單向關係，只能說這個變因存在且未被控制。

詳見 [`00A`](notebooks/00A_HiDF資料集切分.ipynb)、[`00B`](notebooks/00B_CelebDFv2_平衡測試集建立.ipynb) 與 [`02`](notebooks/02_FFPP_Deepfakes_YOLO人臉裁切.ipynb)、[`09A`](notebooks/09A_YOLO手機照片人臉裁切.ipynb)。

**六、HiDF 的 Fake 存在嚴重的種族分布偏斜**

| 種族 | Real | Fake |
|---|---|---|
| White | 694 | **2,427** |
| Black | 694 | 236 |
| Asian | 694 | 358 |
| Latino | 691 | 90 |
| Indian | 353 | 15 |

Real 各族接近均衡，Fake 有 78% 為 White。原因是 HiDF 的 Fake 可用影像庫存本身即高度不均（White 5,126 對 Indian 15），平均抽樣無法補足。

模型有可能部分依賴與種族相關的特徵區分 Real／Fake，而非偽造痕跡本身。HiDF 上的高分應據此打折解讀。

另需說明：HiDF 的 train／val／test 是將圖片路徑打散後依數量切分，並非以身分為單位，同一 ID 可能同時出現在三個子集。本次研究的模型從未接觸 HiDF 的任何子集，故不受影響；但若以此 val 集評估曾在 HiDF train 上訓練過的模型，成績會因身分洩漏而偏高。

**七、外部資料集的規模有限**

HiDF 測試集 626 張、Celeb-DF v2 平衡測試集 1,068 張。樣本量不足以支撐精確的效能估計，僅用於觀察趨勢。

---

## Repo 結構

```text
├── notebooks/                          實驗流程，含完整執行輸出
│   ├── 00A_HiDF資料集切分.ipynb                外部測試集來源（先前專題）
│   ├── 00B_CelebDFv2_平衡測試集建立.ipynb        外部測試集來源
│   ├── 01_FFPP_Deepfakes_資料準備.ipynb
│   ├── 02_FFPP_Deepfakes_YOLO人臉裁切.ipynb
│   ├── 03_FFPP_CLIP_Baseline訓練.ipynb
│   ├── 04_CLIP_跨資料集測試.ipynb
│   ├── 05_FFPP_CLIP_Adapter訓練.ipynb          對照實驗（負向結果）
│   ├── 06_FFPP_ForensicsAdapter_Mask資料準備.ipynb
│   ├── 07_ForensicsAdapter_完整訓練與測試.ipynb
│   ├── 08_ForensicsAdapter_跨資料集測試.ipynb
│   ├── 09A_YOLO手機照片人臉裁切.ipynb
│   ├── 09B_手機資料重新切分.ipynb
│   └── 09C_手機資料門檻校準與正式測試.ipynb
├── results/                            各實驗的原始輸出
│   ├── 03_clip_baseline/
│   ├── 05_feature_adapter/
│   ├── 07_forensics_adapter/
│   ├── 08_cross_dataset/
│   │   ├── HiDF/
│   │   └── CelebDF_v2/
│   └── 09C_threshold_calibration/
├── figures/                            結果圖表，索引見 figures/README.md
├── docs/
│   └── third_party_modifications.md    對官方 ForensicsAdapter 的本機修改紀錄
├── README.md
├── LICENSE                             MIT
├── NOTICE                              第三方程式碼與資料集的授權說明
└── requirements.txt
```

Notebook 保留完整執行輸出，可直接在 GitHub 上檢視所有指標，不需執行任何程式。

**未包含的內容**

- 模型權重（`best_model.pth`）：ViT-L/14 checkpoint 超過 GitHub 檔案大小限制。
- 任何資料集與影像：FaceForensics++ 與 Celeb-DF v2 的授權條款限制影像再散布，需向原作者申請。基於同樣理由，notebook 中原有的人臉預覽與誤判樣本影像已移除，程式碼保留，可在本機重現。
- 官方 ForensicsAdapter 原始碼：請見下方連結，本 repo 不重新散布。

---

## 重現方式

Notebook 中的路徑為作者本機環境（Windows），重現時需自行調整。資料集需另行取得：

- [FaceForensics++](https://github.com/ondyari/FaceForensics)：需向原作者申請
- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)：需向原作者申請
- [HiDF](https://github.com/DSAIL-SKKU/HiDF)（Human-Indistinguishable Deepfake Dataset）：[Zenodo](https://zenodo.org/records/16140829)，CC BY-NC 4.0。本次使用其 val 子集，切分方式見 [`00A`](notebooks/00A_HiDF資料集切分.ipynb)
- [ForensicsAdapter 官方實作](https://github.com/OUC-VAS/ForensicsAdapter)：[`07`](notebooks/07_ForensicsAdapter_完整訓練與測試.ipynb)、[`08`](notebooks/08_ForensicsAdapter_跨資料集測試.ipynb)、[`09C`](notebooks/09C_手機資料門檻校準與正式測試.ipynb) 需要

執行順序為 [`00A`](notebooks/00A_HiDF資料集切分.ipynb)／[`00B`](notebooks/00B_CelebDFv2_平衡測試集建立.ipynb)（建立外部測試集）→ [`01`](notebooks/01_FFPP_Deepfakes_資料準備.ipynb) → [`09C`](notebooks/09C_手機資料門檻校準與正式測試.ipynb)。每份 notebook 的安全開關（`RUN_TRAINING`、`RUN_TEST`、`SMOKE_TEST` 等）預設為保守值，需依序確認資料檢查通過後再啟用。

環境見 `requirements.txt`。實驗環境為 Python 3.10、PyTorch 2.5.1+cu121、RTX 4070 12GB。

---

## 引用與授權

本研究使用了 Forensics Adapter 的官方實作。對其程式碼所做的三處本機相容性修改記錄於 [`docs/third_party_modifications.md`](docs/third_party_modifications.md)，皆不改動模型架構或損失函數定義。

```bibtex
@InProceedings{Cui_2025_CVPR,
  author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
  title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {19207-19217}
}
```

本研究使用的 HiDF 資料集：

```bibtex
@inproceedings{Kang_2025_HiDF,
  author    = {Kang, Chaewon and Jeong, Seoyoon and Lee, Jonghyun and
               Choi, Daejin and Woo, Simon S. and Han, Jinyoung},
  title     = {HiDF: A Human-Indistinguishable Deepfake Dataset},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  year      = {2025},
  doi       = {10.1145/3711896.3737399}
}
```

### 授權

本 repository 自身的 notebook、文件與圖表採用 MIT 授權（見 [`LICENSE`](LICENSE)）。

但本研究依賴的第三方專案與資料集各有更嚴格的條款，MIT 授權不能解除這些限制：

| 來源 | 條款 |
|---|---|
| ForensicsAdapter 官方實作 | CC BY-NC 4.0，限學術研究，禁止商業使用 |
| FaceForensics++ | 需申請，限學術研究，禁止再散布影像 |
| Celeb-DF v2 | 需申請，限學術研究，禁止再散布影像 |
| HiDF | CC BY-NC 4.0，禁止商業使用 |
| YOLOv11n-Face 權重 | 適用 akanametov/yolo-face 之條款 |

[`07`](notebooks/07_ForensicsAdapter_完整訓練與測試.ipynb)、[`08`](notebooks/08_ForensicsAdapter_跨資料集測試.ipynb)、[`09C`](notebooks/09C_手機資料門檻校準與正式測試.ipynb) 需搭配 ForensicsAdapter 官方程式碼才能執行，因此**實際執行本專案的完整流程時仍受其非商業條款約束**。MIT 涵蓋的是本 repository 的原創部分，不是整套流程。

完整說明見 [`NOTICE`](NOTICE)。
