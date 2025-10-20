# -*- coding: utf-8 -*-
"""
在 Spyder 中透過 rpy2 呼叫 R 的 C5.0 套件 (最終穩定版)
"""
# =============================================================================
# 1. 導入必要的函式庫
# =============================================================================
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 🔹 加上 confusion_matrix
import numpy as np
import matplotlib.pyplot as plt  # 🔹 匯入繪圖
import seaborn as sns            # 🔹 匯入 seaborn
# 導入 rpy2 相關工具
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# 導入您自己的預處理函式 (請確保 Preprocessing.py 也已更新為 dtype=int 版本)
from Preprocessing import load_preprocess_data
from export_results import export_to_excel


# =============================================================================
# 2. 載入並轉換資料
# =============================================================================
print("--- 步驟 1: 載入並預處理 Python 資料 ---")
X_train, y_train, X_test, y_test = load_preprocess_data()

X_train = X_train.astype(int)
X_test = X_test.astype(int)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("\n--- 步驟 2: 將 Python DataFrame 轉換為 R DataFrame (格式轉換) ---")
with localconverter(robjects.default_converter + pandas2ri.converter):
  r_x_train = robjects.conversion.py2rpy(X_train)
  r_y_train = robjects.conversion.py2rpy(y_train)
  r_x_test = robjects.conversion.py2rpy(X_test)

# =============================================================================
# 3. 在 Python 中呼叫 R 進行模型訓練與預測
# =============================================================================
print("\n--- 步驟 3: 導入 R 的 C50 套件並訓練模型 ---")
base = importr('base')
C50 = importr('C50')

r_train_df = base.cbind(r_x_train, r_y_train)
new_colnames = list(X_train.columns) + ['income']
r_train_df.colnames = new_colnames

formula = robjects.Formula('income ~ .')
r_train_df[r_train_df.colnames.index('income')] = base.as_factor(
    r_train_df[r_train_df.colnames.index('income')]
)

c5_model = C50.C5_0(formula, data=r_train_df, trials=1)
print("R 模型訓練完成！")

tree_sizes = list(c5_model.rx2('size'))
avg_tree_size = np.mean(tree_sizes)
print(f"C5.0 模型共進行了 {len(tree_sizes)} 次 boosting")
print(f"決策樹的平均大小 (節點數): {avg_tree_size:.2f}")

print("\n--- 步驟 4: 使用 R 模型進行預測 ---")
with localconverter(robjects.default_converter + pandas2ri.converter):
    predictions_r = C50.predict_C5_0(c5_model, newdata=r_x_test)
with localconverter(robjects.default_converter + pandas2ri.converter):
    train_pred_r = C50.predict_C5_0(c5_model, newdata=r_x_train)

# =============================================================================
# 4. 將結果轉回 Python 並評估
# =============================================================================
print("\n--- 步驟 5: 將預測結果轉回 Python 並進行評估 ---")
predictions_py = [int(p) for p in list(predictions_r)]
train_pred_py = [int(p) for p in list(train_pred_r)]

train_acc = accuracy_score(y_train, train_pred_py)
print(f"訓練資料準確率 (Training Accuracy): {train_acc:.4f}")

accuracy = accuracy_score(y_test, predictions_py)
report = classification_report(y_test, predictions_py, target_names=['<=50K', '>50K'])

print(f"\n最終模型準確率 (Accuracy): {accuracy:.4f}\n")
print("最終分類報告 (Classification Report):\n", report)

# 🔹 新增：混淆矩陣（圖表）
cm = confusion_matrix(y_test, predictions_py)
cm_df = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Pred <=50K', 'Pred >50K'],
                    yticklabels=['True <=50K', 'True >50K'])
plt.title("C5.0 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# 🔹 儲存混淆矩陣圖
cm_path = "C50_confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"混淆矩陣圖已儲存為：{cm_path}")

# =============================================================================
# 匯出至EXCEL
# =============================================================================
output_filename = 'DataMining_Results.xlsx'
sheet_name_c50 = 'C5.0'

export_result = export_to_excel(
    y_true=y_test,
    y_pred=predictions_py,
    file_name=output_filename,
    sheet_name=sheet_name_c50,
    model_name='C5.0'
)
