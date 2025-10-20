# -*- coding: utf-8 -*-
from Preprocessing import load_preprocess_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix   # 🔹 加入 confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt   # 🔹 匯入繪圖
import seaborn as sns             # 🔹 匯入 seaborn
from export_results import export_to_excel

# 載入預處理資料
X_train, y_train, X_test, y_test = load_preprocess_data()

# 建立 C4.5 決策樹模型
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# 預測訓練與測試集
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 計算準確率
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# 取得決策樹資訊
tree_depth = clf.get_depth()
num_leaves = clf.get_n_leaves()

# 分類報告
report = classification_report(y_test, y_test_pred, target_names=['<=50K', '>50K'])
report_dict = classification_report(
    y_test, y_test_pred, target_names=['<=50K', '>50K'], output_dict=True
)
report_df = pd.DataFrame(report_dict).T

# 🔹 新增：混淆矩陣
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=['True <=50K', 'True >50K'], columns=['Pred <=50K', 'Pred >50K'])

# 匯出結果
print("================== C4.5 決策樹結果 ==================")
print(f"訓練集準確率: {train_acc:.4f}")
print(f"測試集準確率: {test_acc:.4f}")
print(f"決策樹深度: {tree_depth}")
print(f"葉節點數量: {num_leaves}")
print("\n=================== 測試集分類報告 ===================")
print(report)

# 🔹 新增：印出混淆矩陣
print("\n=================== 混淆矩陣 ===================")
print(cm_df)

# 🔹 新增：繪製並輸出混淆矩陣圖
plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("C4.5 Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# 儲存圖檔
cm_image_path = "C45_confusion_matrix.png"
plt.savefig(cm_image_path, dpi=300)
plt.close()

print(f"\n混淆矩陣圖已儲存為：{cm_image_path}")

# =============================================================================
# 匯出至EXCEL (修正部分)
# =============================================================================
print("\n--- 將結果匯出至 Excel ---")
output_filename = 'DataMining_Results.xlsx'  # 使用大家共用的檔名
sheet_name_c45 = 'C4.5'  # 為 C4.5 模型建立一個新的工作表名稱

# 呼叫共用函式，傳入作業要求的資料
export_to_excel(
    y_true=y_test,
    y_pred=y_test_pred,
    file_name=output_filename,
    sheet_name=sheet_name_c45,
    model_name='C4.5'
)

# 🔹 可選：將混淆矩陣數值表也寫進 Excel（非必要）
#with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl') as writer:
#   cm_df.to_excel(writer, sheet_name='C4.5_CM')
