# 對官方 ForensicsAdapter 實作的本機修改

本研究的 `07`、`08`、`09C` 使用 Forensics Adapter 的官方實作：

- Repository：https://github.com/OUC-VAS/ForensicsAdapter
- 論文：Cui, X., Li, Y., Luo, A., Zhou, J., & Dong, J. (2025). *Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection*. CVPR 2025, pp. 19207–19217.
- 授權：CC BY-NC 4.0，僅供學術研究使用

官方原始碼未包含在本 repo 中，需自行取得。

為在本機環境執行，以下三處需要修改或接管。**三項皆為環境相容性處理，不改動模型架構、損失函數定義或任何超參數。**

---

## 1. CLIP 權重下載路徑

**位置**：`model/ds.py`

**問題**：官方程式將 CLIP 權重的下載目錄寫死為原作者的環境路徑 `/data/cuixinjie/weights`，在其他機器上不存在也無法寫入。

**處理**：在載入模組前將該字串替換為本機可寫入的快取目錄，並在首次修改時保留原檔備份為 `ds.py.codex_backup`。

**影響範圍**：僅權重的存放位置，不影響載入的權重內容。

**相關程式碼**：`07` 的第 4 節。

---

## 2. Windows 環境下的 Unicode 路徑

**位置**：官方 `dataset/abstract_dataset.py` 內部使用的 `cv2.imread` 與 `open()`

**問題**：本研究的資料路徑包含中文字元。在 Windows 環境下：

- `cv2.imread` 無法直接讀取含非 ASCII 字元的路徑，會回傳 `None`。
- 官方讀取 JSON 時未指定 `encoding`，Windows 預設使用 cp950，導致 UTF-8 內容解碼失敗。

**處理**：

- 以支援 Unicode 的版本接管 `cv2.imread`：偵測到路徑含非 ASCII 字元時，改用 `np.fromfile` + `cv2.imdecode` 讀取。
- 以動態載入的方式取得官方 Dataset 模組，並接管其命名空間中的 `open()`，在未指定編碼時固定使用 UTF-8。

**影響範圍**：僅檔案讀取行為。不修改官方 `.py` 檔案內容，也不改變 Dataset 回傳的資料。

**相關程式碼**：`07` 的第 4 節。

---

## 3. Albumentations 版本相容性

**位置**：`DeepfakeAbstractBaseDataset.init_data_aug_method`

**問題**：官方程式固定使用 `albumentations==1.1.0`。在較新版本中，舊式 `IsotropicResize` 的 `always_apply=False` 會被解讀為機率 0，導致 `A.OneOf` 內三個選項的機率全部為 0，計算選擇機率時發生除以零的錯誤。

本研究的環境安裝的是 albumentations 2.0.8。

**處理**：覆寫 `init_data_aug_method`，在 `use_data_augmentation=False` 時直接跳過該增強流程的初始化。

**影響範圍**：本研究的所有實驗皆設定 `use_data_augmentation=False`，此增強流程本來就不會被使用，跳過初始化不改變任何實際行為。若需啟用資料增強，應改為安裝 `albumentations==1.1.0`。

**相關程式碼**：`07` 的第 4 節。

---

## 4. 記憶體使用（非修改，屬使用方式差異）

官方 `DS` 類別在 `inference=True` 時會持續累加內部的 `prob`、`label`、`correct`、`total`，用於其自身的統計流程。

本研究自行計算所有指標，未使用官方的統計輸出。因此在每次評估的前後主動清空這些累加變數，避免在多輪評估中記憶體單調增長。

這不是對官方程式碼的修改，而是外部呼叫端的資源管理。

**相關程式碼**：`07` 的第 8 節 `evaluate` 函式。
