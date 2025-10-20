# -*- coding: utf-8 -*-
"""
CART模型 + 混淆矩陣輸出
"""
# train_models.py

# 1. 匯入必要的套件
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# 從我們自訂的 preprocessing.py 檔案中，匯入函式
from Preprocessing import load_preprocess_data
from export_results import export_to_excel

# 畫圖&設定中文字
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"中文設定警告: {e}")

# 2. 載入並取得前置處理過的資料
x_train, y_train, x_test, y_test = load_preprocess_data()

# 3. 訓練 CART 決策樹模型
print("\n--- 正在訓練 CART 模型 ---")
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(x_train, y_train)
print("CART 模型訓練完成！")

# 取得決策樹資訊
tree_depth = cart_model.get_depth()
leaf_nodes = cart_model.get_n_leaves()
print(f"決策樹的深度為: {tree_depth}")
print(f"決策樹的葉節點數量為: {leaf_nodes}")

# 4. 使用訓練好的模型進行預測
y_train_pred = cart_model.predict(x_train)
y_test_pred = cart_model.predict(x_test)

# 5. 評估模型效能
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, target_names=['<=50K', '>50K'])

print(f"訓練資料正確率 (Train Accuracy): {train_accuracy:.4f}")
print(f"測試資料正確率 (Test Accuracy): {test_accuracy:.4f}")
print("測試資料分類報告 (Classification Report):\n", report)

# --- 主模型混淆矩陣 ---
cm_main = confusion_matrix(y_test, y_test_pred)
disp_main = ConfusionMatrixDisplay(cm_main, display_labels=['<=50K', '>50K'])
fig, ax = plt.subplots(figsize=(6,6))
disp_main.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title("CART 主模型混淆矩陣")
plt.savefig("test_cart_confusion_matrix_main.png", dpi=300)
plt.show()
plt.close()

# =============================================================================
# 匯出至 EXCEL
# =============================================================================
output_filename = 'DataMining_Results.xlsx'
sheet_name_cart = 'CART' 

export_to_excel(
    y_true=y_test,
    y_pred=y_test_pred,
    file_name=output_filename,
    sheet_name=sheet_name_cart,
    model_name='CART'
)

# =============================================================================
# 成本複雜度剪枝 (CCP) 分析
# =============================================================================
print("\n--- 開始 CART 成本複雜度剪枝 (CCP) 分析 ---")
path = cart_model.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # 移除 alpha 最大值
clfs = []

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='gini', ccp_alpha=alpha, random_state=42)
    clf.fit(x_train, y_train)
    clfs.append(clf)

train_scores = [clf.score(x_train, y_train) for clf in clfs]
test_scores = [clf.score(x_test, y_test) for clf in clfs]

best_alpha_index = test_scores.index(max(test_scores))
best_alpha = ccp_alphas[best_alpha_index]
best_score = max(test_scores)
print(f"剪枝分析完成！最佳測試準確率: {best_score:.4f} (ccp_alpha ≈ {best_alpha:.6f})")

# 繪製 CCP 剪枝曲線
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(ccp_alphas, train_scores, marker='o', label='訓練準確率', drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='s', label='測試準確率', drawstyle="steps-post")
ax.set_xscale('log')
ax.axvline(best_alpha, linestyle='--', color='red', label=f'最佳 Alpha ≈ {best_alpha:.6f}')
ax.set_title('CART CCP 剪枝：準確率 vs. ccp_alpha', fontsize=16)
ax.set_xlabel('ccp_alpha (對數尺度)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)
plt.savefig("test_cart_ccp_accuracy_curve_log.png")
plt.show()
plt.close()

# 比較三種特定剪枝參數設定
alpha_A = 0.0       # 未剪枝
alpha_B = best_alpha # 最佳剪枝
alpha_C = 0.001     # 過度剪枝
selected_alphas = [
    {"name": "未剪枝", "alpha": alpha_A, "filename": "test_cart_tree_unpruned.png"},
    {"name": "最佳剪枝", "alpha": alpha_B, "filename": "test_cart_tree_best_pruned.png"},
    {"name": "過度剪枝", "alpha": alpha_C, "filename": "test_cart_tree_over_pruned.png"}
]

ccp_results = []
for config in selected_alphas:
    alpha = config["alpha"]
    name = config["name"]
    filename_tree = config["filename"]
    
    print(f"\n--- 訓練與評估模型: {name} (alpha={alpha:.6f}) ---")
    ccp_model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    ccp_model.fit(x_train, y_train)
    y_pred_test = ccp_model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=['<=50K', '>50K'])
    print(f"測試資料準確率: {accuracy:.4f}")
    print("分類報告:\n", report)
    
    result_summary = {
        "設定": name,
        "ccp_alpha": f"{alpha:.6f}",
        "測試準確率": f"{accuracy:.4f}",
        "樹的深度": ccp_model.get_depth(),
        "葉節點數量": ccp_model.get_n_leaves()
    }
    ccp_results.append(result_summary)
    
    # 繪製決策樹
    plt.figure(figsize=(20, 12))
    plot_tree(ccp_model,
              filled=True,
              feature_names=x_train.columns.tolist(),
              class_names=['<=50K', '>50K'],
              rounded=True,
              fontsize=10,
              max_depth=3)
    plt.title(f'決策樹視覺化 - {name}\n(Alpha={alpha:.6f}, 顯示頂部3層)', fontsize=20)
    plt.savefig(filename_tree, dpi=300)
    plt.close()
    
    # --- 產生混淆矩陣 ---
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K'])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f"CART {name} 混淆矩陣")
    cm_filename = f"cart_confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(cm_filename, dpi=300)
    plt.close()

# 最終比較
results_df = pd.DataFrame(ccp_results)
print("\n--- 三種剪枝設定最終比較 ---")
print(results_df.to_string())
