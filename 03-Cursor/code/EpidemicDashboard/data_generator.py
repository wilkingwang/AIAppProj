import pandas as pd
import json
import numpy as np
from datetime import datetime
import os

def load_excel_data():
    """加载Excel数据"""
    try:
        df = pd.read_excel('香港各区疫情数据_20250322.xlsx')
        print(f"成功读取数据：{len(df)}行{len(df.columns)}列")
        return df
    except Exception as e:
        print(f"读取Excel文件失败：{e}")
        return None

def process_data(df):
    """处理数据为可视化格式"""
    if df is None:
        return None
    
    # 自动检测列名
    date_col = None
    district_col = None
    new_cases_col = None
    
    for col in df.columns:
        col_str = str(col).strip()
        if '日期' in col_str or 'date' in col_str.lower():
            date_col = col
        elif '区' in col_str and '确诊' not in col_str:
            district_col = col
        elif '新增' in col_str or '新增确诊' in col_str:
            new_cases_col = col
    
    if not all([date_col, district_col, new_cases_col]):
        print("未找到必要的列，使用默认列名...")
        # 使用默认列名
        date_col = df.columns[0] if len(df.columns) > 0 else '报告日期'
        district_col = df.columns[1] if len(df.columns) > 1 else '地区'
        new_cases_col = df.columns[2] if len(df.columns) > 2 else '新增确诊'
    
    # 数据清理
    df[new_cases_col] = pd.to_numeric(df[new_cases_col], errors='coerce').fillna(0)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 移除空值
    df = df.dropna(subset=[date_col, district_col, new_cases_col])
    
    return df, date_col, district_col, new_cases_col

def generate_dashboard_data():
    """生成大屏所需的数据"""
    df = load_excel_data()
    if df is None:
        return False
    
    processed_data = process_data(df)
    if processed_data is None:
        return False
    
    df, date_col, district_col, new_cases_col = processed_data
    
    # 获取唯一日期并排序
    dates = df[date_col].dt.strftime('%Y-%m-%d').unique().tolist()
    dates.sort()
    
    # 获取所有地区
    districts = df[district_col].unique().tolist()
    
    # 准备趋势图数据
    trend_data = []
    for district in districts[:5]:  # 只取前5个地区
        district_data = df[df[district_col] == district]
        district_dates = district_data.groupby(date_col)[new_cases_col].sum()
        
        # 确保所有日期都有数据
        full_dates = pd.Series(index=pd.to_datetime(dates), data=0)
        full_dates.update(district_dates)
        
        trend_data.append({
            'name': district,
            'data': full_dates.values.tolist()
        })
    
    # 计算累计数据
    total_data = []
    for district in districts:
        total = df[df[district_col] == district][new_cases_col].sum()
        total_data.append({
            'name': district,
            'value': int(total)
        })
    
    # 按累计确诊排序
    total_data.sort(key=lambda x: x['value'], reverse=True)
    
    # 每日新增总和
    daily_total = df.groupby(date_col)[new_cases_col].sum()
    full_daily = pd.Series(index=pd.to_datetime(dates), data=0)
    full_daily.update(daily_total)
    daily_new = full_daily.values.tolist()
    
    # 香港18区地理坐标（简化版）
    hong_kong_coords = {
        '中西区': [114.1547, 22.2819],
        '湾仔区': [114.1780, 22.2775],
        '东区': [114.2250, 22.2800],
        '南区': [114.1667, 22.2333],
        '油尖旺区': [114.1716, 22.3121],
        '深水埗区': [114.1594, 22.3300],
        '九龙城区': [114.1880, 22.3260],
        '黄大仙区': [114.2019, 22.3367],
        '观塘区': [114.2167, 22.3167],
        '葵青区': [114.1333, 22.3667],
        '荃湾区': [114.1167, 22.3667],
        '屯门区': [113.9667, 22.4000],
        '元朗区': [114.0333, 22.4500],
        '北区': [114.1500, 22.5000],
        '大埔区': [114.1667, 22.4500],
        '沙田区': [114.1833, 22.3833],
        '西贡区': [114.2667, 22.3167],
        '离岛区': [113.9500, 22.2833]
    }
    
    # 为地图数据添加坐标
    map_data = []
    for item in total_data:
        district_name = item['name']
        if district_name in hong_kong_coords:
            map_data.append({
                'name': district_name,
                'value': [hong_kong_coords[district_name][0], hong_kong_coords[district_name][1], item['value']]
            })
    
    # 生成最终数据
    dashboard_data = {
        'dates': dates,
        'trend': trend_data,
        'pie': total_data[:5],  # 饼图只显示前5个
        'map': map_data,  # 地图数据
        'daily': daily_new,
        'rank': total_data,
        'summary': {
            'total_cases': int(df[new_cases_col].sum()),
            'today_cases': int(df[df[date_col] == df[date_col].max()][new_cases_col].sum()),
            'total_districts': len(districts),
            'latest_date': dates[-1] if dates else None
        }
    }
    
    # 保存数据文件
    with open('dashboard_data.json', 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    
    print("数据文件已生成：dashboard_data.json")
    print(f"数据概览：")
    print(f"- 日期范围：{dates[0]} 到 {dates[-1]}")
    print(f"- 涉及地区：{len(districts)}个")
    print(f"- 累计确诊：{dashboard_data['summary']['total_cases']}")
    print(f"- 最新日期：{dashboard_data['summary']['latest_date']}")
    
    return True

if __name__ == "__main__":
    generate_dashboard_data()