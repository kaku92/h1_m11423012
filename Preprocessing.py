# -*- coding: utf-8 -*-
"""
資料前處裡
遭遇錯誤使否可以重新執行? Y
"""
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder

def load_preprocess_data():
# 1. 欄位名稱定義
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]    
    
    # 2. 載入訓練與測試資料
    train_path='AdultData/adult.data'
    test_path='AdultData/adult.test'
    train_df = pd.read_csv(train_path, header=None, names=column_names, 
                           sep=r',\s*', engine='python', na_values='?')
    
    
    # 載入測試資料，skiprows=1 跳過第一行註解
    test_df = pd.read_csv(test_path, header=None, names=column_names, 
                          sep=r',\s*', engine='python', na_values='?', skiprows=1)
    
    train_df_org=train_df
    test_df_org=test_df
    # 3. 【移除不需要的欄位】 權重因子(fnlwgt)、教育程度數值(education)
    columns_to_drop = ['fnlwgt', 'education']#native-country 可考慮資料幾乎為US
    
    train_df = train_df.drop(columns=columns_to_drop, axis=1)
    
    test_df = test_df.drop(columns=columns_to_drop, axis=1)
    print(f"\n移除了 {len(columns_to_drop)} 個欄位: {columns_to_drop}")
    
        
    
    # 4. 資料清理: 刪除含有 NaN 的任何一行
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
  
    
    # 修正測試資料集 income 欄位的句點
    test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

    # 重設索引 
    # 防止刪除資料後索引會不連續，重設索引可以讓它變回從0開始的連續編號
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    

    # 5. 資料特徵編碼
    # 先合併 --> 編碼
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # (1) 編碼目標欄位 (income)
    le = LabelEncoder()
    combined_df['income'] = le.fit_transform(combined_df['income'])

    # (2) 編碼特徵欄位 (One-Hot Encoding)
    # 找出所有物件型別(object)的欄位，這些就是我們要編碼的類別特徵
    categorical_cols = combined_df.select_dtypes(include=['object']).columns

    # 使用 pandas 的 get_dummies 進行 One-Hot Encoding
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

    # --- 編碼完成，將資料拆分回訓練集與測試集 ---
    train_df = combined_df[combined_df['source_train'] == 1].copy()
    test_df = combined_df[combined_df['source_train'] == 0].copy()

    train_df.drop(columns=['source_train'], inplace=True) # get_dummies 會自動轉換 source 欄位
    test_df.drop(columns=['source_train'], inplace=True)

    # (3) 最終分割成 X (特徵) 和 y (目標)
    X_train = train_df.drop('income', axis=1)
    y_train = train_df['income']
    X_test = test_df.drop('income', axis=1)
    y_test = test_df['income']

    print("\n資料編碼與最終分割完成！")
    print(f"訓練資料最終維度: {X_train.shape}")
    print(f"測試資料最終維度: {X_test.shape}")
       
    #--防呆--
    if X_train.empty:
        raise ValueError("錯誤：經過前置處理後，訓練資料集為空！無法繼續進行模型訓練。")
    if X_test.empty:
        raise ValueError("錯誤：經過前置處理後，測試資料集為空！無法繼續進行模型訓練。")
        
    
    if(len(train_df_org) < len(X_train)):    
        error_message = (
        f"\n--- 警告: 訓練數據不符合邏輯 ---\n"
        f"原始訓練資料筆數: {len(train_df_org)}\n"
        f"實際最終筆數: {len(X_train)}")
        raise ValueError(error_message)
    if(len(test_df_org) < len(X_test)):    
        error_message = (
        f"\n--- 警告: 訓練數據不符合邏輯 ---\n"
        f"原始訓練資料筆數: {len(test_df_org)}\n"
        f"實際最終筆數: {len(X_test)}")
        raise ValueError(error_message)
    
    train_loss_ratio = 1 - (len(X_train) / len(train_df_org))
    test_loss_ratio = 1 - (len(X_test) / len(test_df_org))
    print('訓練資料遺失率: ',train_loss_ratio)
    print('測試資料遺失率: ',test_loss_ratio)
    print("--------------------前處理完成--------------------")


    return X_train, y_train, X_test, y_test



