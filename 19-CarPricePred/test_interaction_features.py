#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试交互特征的创建和类型标记
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering_and_catboost import create_car_features, create_statistical_features, encode_categorical_features, feature_selection

def test_interaction_features():
    """测试交互特征的创建和类型标记"""
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'brand': ['大众', '丰田', '大众', '奔驰'],
        'bodyType': ['轿车', 'SUV', '轿车', '轿车'],
        'gearbox': ['自动', '手动', '自动', '自动'],
        'kilometer': [50000, 80000, 30000, 60000],
        'vehicle_age_years': [3, 5, 2, 4],
        'power': [120, 150, 100, 180],
        'regDate': [202001, 201901, 202101, 202001],
        'creatDate': [202301, 202301, 202301, 202301],
        'model': ['朗逸', 'RAV4', '速腾', 'C级'],
        'fuelType': ['汽油', '汽油', '汽油', '汽油'],
        'notRepairedDamage': ['无', '无', '有', '无'],
        'v_0': [1.2, 1.5, 1.1, 1.8],
        'price': [120000, 180000, 150000, 250000]
    })
    
    print('原始数据:')
    print(test_data)
    print()
    
    # 测试特征创建
    data_features = create_car_features(test_data.copy())
    print('创建特征后:')
    print('新增特征数量:', len(data_features.columns))
    print('所有特征:', data_features.columns.tolist())
    print()
    
    # 检查新增的交互特征
    interaction_cols = [col for col in data_features.columns if any(term in col for term in ['brand_bodyType', 'brand_gearbox', 'age_km', 'km_per_year', 'age_segment'])]
    print('交互特征:')
    for col in interaction_cols:
        print(f"  {col}: {data_features[col].dtype}")
    print()
    
    # 检查具体的数据类型
    if 'brand_bodyType' in data_features.columns:
        print(f'brand_bodyType 数据类型: {data_features[\"brand_bodyType\"].dtype}')
        print(f'brand_bodyType 唯一值: {data_features[\"brand_bodyType\"].unique()}')
    if 'brand_gearbox' in data_features.columns:
        print(f'brand_gearbox 数据类型: {data_features[\"brand_gearbox\"].dtype}')
        print(f'brand_gearbox 唯一值: {data_features[\"brand_gearbox\"].unique()}')
    if 'age_km_segment' in data_features.columns:
        print(f'age_km_segment 数据类型: {data_features[\"age_km_segment\"].dtype}')
        print(f'age_km_segment 唯一值: {data_features[\"age_km_segment\"].unique()}')
    if 'km_per_year' in data_features.columns:
        print(f'km_per_year 数据类型: {data_features[\"km_per_year\"].dtype}')
        print(f'km_per_year 值: {data_features[\"km_per_year\"].tolist()}')
    
    print('\n' + '='*50)
    print('测试特征选择和类型转换...')
    
    # 测试特征选择函数
    data_selected, categorical_cols = feature_selection(data_features, ['brand', 'bodyType', 'gearbox', 'model'])
    
    print('分类特征列表:', categorical_cols)
    print()
    
    # 检查分类特征的数据类型
    for col in categorical_cols:
        if col in data_selected.columns:
            print(f'{col}: {data_selected[col].dtype}')
    
    print('\n测试完成！')

if __name__ == '__main__':
    test_interaction_features()