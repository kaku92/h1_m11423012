# -*- coding: utf-8 -*-
"""
匯出成Xlsx
"""
import pandas as pd
import os

def export_to_excel(y_true, y_pred, file_name, sheet_name, model_name):

    try:
        # 假設 y_true 是 pandas Series
        correct_labels = y_true.reset_index(drop=True)
        
    except AttributeError:
        correct_labels = y_true
    results_df = pd.DataFrame({
        '正確類別': correct_labels,
        '預測類別': y_pred
    })
    #補上預測結果正確
    results_df['是否正確'] = (results_df['正確類別'] == results_df['預測類別']).map({True: '✅', False: '❌'})

    #寫入EXCEL
    try:
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        full_path = os.path.abspath(file_name)
        print(f"{model_name} 的預測結果已成功儲存至 '{full_path}' 的 '{sheet_name}' 工作表中。")
    except FileNotFoundError:
        # 如果檔案完全不存在，第一次寫入用w
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='w') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"已建立新檔案 '{file_name}' 並將 {model_name} 的預測結果儲存至 '{sheet_name}' 工作表中。")
    except Exception as e:
        print(f"寫入 Excel 失敗: {e}")
        
    return f"{model_name} 檔案儲存成功."


