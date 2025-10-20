# 資料探勘作業一 - 決策樹分類器 (h1_m11423012)

本專案使用 UCI Adult 資料集 比較多種決策樹演算法的效能，並實作剪枝分析。

## 專案內容

* **資料前處理** (`Preprocessing.py`):
    * 載入 `adult.data` 與 `adult.test` 資料集。
    * 清理遺失值 (`?`) 與測試集中的錯誤字元 (`.`)。
    * 合併訓練/測試集後，進行 One-Hot Encoding。
* **演算法比較**:
    * **CART (Gini)** (`confusion_CART.py`): 使用 Gini 不純度。
    * **CART (Entropy)** (`confusion_ID3.py`): 使用 Information Gain (`criterion='entropy'`)，作為 ID3 的模擬實作。
    * **C4.5 (模擬)** (`confusion_C45.py`): 同上，使用 `criterion='entropy'`。
    * **C5.0** (`confusion_C5_0.py`): 透過 `rpy2` 呼叫 R 語言的 `C50` 函式庫。
* **成本複雜度剪枝 (CCP)**:
    * 針對 CART (Gini) 模型進行 CCP 分析 (`confusion_CART.py`)。
    * 繪製 Alpha 值與準確率的關係曲線。
    * 比較「未剪枝」、「最佳剪枝」、「過度剪枝」三種情境。
* **結果匯出** (`export_results.py`):
    * 將所有模型的預測結果（正確類別 vs 預測類別）自動匯出至 `DataMining_Results.xlsx` 的不同工作表中。
    * 儲存所有演算法的混淆矩陣 (Confusion Matrix) 圖片。

---

## 如何執行 (Execution)

警告： 本專案包含 R 語言的呼叫，請務必同時設定 Python 與 R 的環境。
### 1. Python 環境設定

安裝所有必要的 Python 套件：

```bash
pip install pandas scikit-learn matplotlib seaborn numpy openpyxl rpy2
### 2. R 環境設定 (C5.0 執行所必需)
confusion_C5_0.py 需要 R 語言以及 C50 套件。

安裝 R 語言: 至 R 語言官網 下載並安裝 R。

安裝 R 的 C50 套件:install.packages("C50")

3. 資料集準備
本專案預期 AdultData 資料夾與 .py 檔案位於同一層級。
h1_m11423012/
├── AdultData/
│   ├── adult.data
│   └── adult.test
├── confusion_CART.py
├── confusion_C5_0.py
├── confusion_CART.py
├── confusion_ID3.py
└── README.md
4. 執行所有模型
依序執行以下腳本，即可重新產生所有 .png 圖片檔與 DataMining_Results.xlsx。
python confusion_CART.py
python confusion_ID3.py
python confusion_C45.py
python confusion_C5_0.py




※演算法實作說明
C5.0: 本專案使用 rpy2 套件呼叫 R 語言中的 C50 函式庫，此為 C5.0 演算法的標準實作。
