from flask import Flask, render_template, jsonify
import json
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# 地区中英文映射
DISTRICT_NAMES = {
    "Central and Western": "中西区",
    "Eastern": "东区",
    "Southern": "南区",
    "Wan Chai": "湾仔区",
    "Sham Shui Po": "深水埗区",
    "Kowloon City": "九龙城区",
    "Kwun Tong": "观塘区",
    "Wong Tai Sin": "黄大仙区",
    "Yau Tsim Mong": "油尖旺区",
    "Islands": "离岛区",
    "Kwai Tsing": "葵青区",
    "North": "北区",
    "Sai Kung": "西贡区",
    "Sha Tin": "沙田区",
    "Tai Po": "大埔区",
    "Tsuen Wan": "荃湾区",
    "Tuen Mun": "屯门区",
    "Yuen Long": "元朗区"
}

# 加载GeoJSON数据并转换地区名称为中文
with open('hongkong.json', 'r', encoding='utf-8') as f:
    hongkong_geo = json.load(f)

# 将GeoJSON中的英文地区名转换为中文
for feature in hongkong_geo['features']:
    en_name = feature['properties']['name']
    if en_name in DISTRICT_NAMES:
        feature['properties']['name'] = DISTRICT_NAMES[en_name]

# 模拟疫情数据
def generate_covid_data():
    districts = list(DISTRICT_NAMES.keys())
    
    data = []
    start_date = datetime(2022, 1, 1)
    
    for i in range(180):  # 6个月数据
        current_date = start_date + timedelta(days=i)
        for district in districts:
            # 模拟真实的疫情增长趋势
            base_cases = random.randint(0, 5)
            if district in ["Yau Tsim Mong", "Kowloon City", "Central and Western"]:
                base_cases += random.randint(0, 10)  # 核心区域病例更多
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'district': district,
                'new_cases': base_cases,
                'total_cases': random.randint(50, 500) + base_cases * 10
            })
    
    return pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/geojson')
def get_geojson():
    return jsonify(hongkong_geo)

@app.route('/api/covid-data')
def get_covid_data():
    df = generate_covid_data()
    
    # 地图热力图数据
    map_data = df.groupby('district').agg({
        'total_cases': 'max',
        'new_cases': 'sum'
    }).reset_index()
    map_data['district_cn'] = map_data['district'].map(DISTRICT_NAMES)
    map_data = map_data.to_dict('records')
    
    # 趋势图数据
    trend_data = df.groupby('date').agg({
        'new_cases': 'sum'
    }).reset_index()
    trend_data = trend_data.to_dict('records')
    
    # 地区排名数据
    district_rank = df.groupby('district').agg({
        'total_cases': 'max',
        'new_cases': 'sum'
    }).reset_index()
    district_rank['district_cn'] = district_rank['district'].map(DISTRICT_NAMES)
    district_rank = district_rank.sort_values('total_cases', ascending=False)
    district_rank = district_rank.head(10).to_dict('records')
    
    # 每日新增对比
    daily_comparison = df.groupby(['date', 'district'])['new_cases'].sum().unstack(fill_value=0)
    daily_comparison = daily_comparison.reset_index().to_dict('records')
    
    # 实时统计数据
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date]
    highest_district_en = latest_data.loc[latest_data['new_cases'].idxmax(), 'district']
    highest_district_cn = DISTRICT_NAMES.get(highest_district_en, highest_district_en)
    
    stats = {
        'total_cases': int(latest_data['total_cases'].sum()),
        'new_cases_today': int(latest_data['new_cases'].sum()),
        'active_districts': int((latest_data['new_cases'] > 0).sum()),
        'highest_district': highest_district_cn,
        'highest_cases': int(latest_data['new_cases'].max())
    }
    
    return jsonify({
        'map_data': map_data,
        'trend_data': trend_data,
        'district_rank': district_rank,
        'daily_comparison': daily_comparison,
        'stats': stats
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)