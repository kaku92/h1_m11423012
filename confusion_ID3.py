# -*- coding: utf-8 -*-
"""
ID3模型(模擬)
真正的ID3因為版本過久且衝突產生無法處理的問題
"""
# train_models.py

# 1. 匯入必要的套件 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 從我們自訂的 preprocessing.py 檔案中，匯入函式
from Preprocessing import load_preprocess_data
from export_results import export_to_excel
# 2. 載入並取得前置處理過的資料
# 這一行會執行 preprocessing.py 裡的所有程式碼，並回傳結果
x_train, y_train, x_test, y_test = load_preprocess_data()



# 3. 訓練 ID3 決策樹模型
print("\n--- 正在訓練 ID3 模型 ---")
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 使用 .fit() 方法進行學習 (使用的資料和 CART 完全一樣)
id3_model.fit(x_train, y_train)

print("ID3 模型訓練完成！")
# 取得決策樹的深度
tree_depth = id3_model.get_depth()
# 取得葉節點的數量
leaf_nodes = id3_model.get_n_leaves()
print(f"決策樹的深度為: {tree_depth}")
print(f"決策樹的葉節點數量為: {leaf_nodes}")
# --- 使用 ID3 模型進行預測 ---
y_train_pred = id3_model.predict(x_train)
y_test_pred = id3_model.predict(x_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred,target_names=['<=50K', '>50K'])
print(f"\n--- ID3 模型評估結果 ---")
print(f"訓練資料正確率 (Train Accuracy): {train_accuracy:.4f}")
print(f"測試資料正確率 (Test Accuracy): {test_accuracy:.4f}")
print("分類報告 (Classification Report):\n", report)

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])

# 繪製並儲存圖檔
fig, ax = plt.subplots(figsize=(6,6))   # 可選擇圖大小
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title("ID3 Confusion Matrix")

# 儲存圖檔
cm_image_path = "ID3_confusion_matrix.png"
plt.savefig(cm_image_path, dpi=300)
plt.show()
plt.close()

# =============================================================================
#匯出至EXCEL
# =============================================================================
print("\n--- 將結果匯出至 Excel ---")
output_filename = 'DataMining_Results.xlsx'
sheet_name_id3 = 'ID3' # 使用不同的工作表名稱

export_to_excel(
    y_true=y_test,
    y_pred=y_test_pred,
    file_name=output_filename,
    sheet_name=sheet_name_id3,
    model_name='ID3'
)
