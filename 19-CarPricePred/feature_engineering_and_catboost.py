#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
二手车价格预测 - 高级特征工程与CatBoost建模
增强版：添加精细地理位置特征和高级特征工程技术
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """
    加载原始数据
    """
    print("正在加载数据...")
    # 加载训练集
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    # 加载测试集
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    数据预处理
    """
    print("\n开始数据预处理...")
    
    # 合并训练集和测试集进行特征工程
    train_data['source'] = 'train'
    test_data['source'] = 'test'
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 保存SaleID
    train_ids = train_data['SaleID']
    test_ids = test_data['SaleID']
    
    # 从训练集获取y值
    y = train_data['price']
    
    return data, y, train_ids, test_ids

# 加载数据
train_data, test_data = load_data()

# 预处理数据
data, y, train_ids, test_ids = preprocess_data(train_data, test_data)
data


# In[2]:


def create_time_features(data):
    """
    创建时间特征 - 优化版
    增加了周期性时间特征和更多时间差特征
    """
    print("创建优化的时间特征...")
    
    # 转换日期格式
    data['regDate'] = pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')
    data['creatDate'] = pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce')
    
    # 处理无效日期
    data.loc[data['regDate'].isnull(), 'regDate'] = pd.to_datetime('20160101', format='%Y%m%d')
    data.loc[data['creatDate'].isnull(), 'creatDate'] = pd.to_datetime('20160101', format='%Y%m%d')
    
    # 车辆年龄（天数）
    data['vehicle_age_days'] = (data['creatDate'] - data['regDate']).dt.days
    
    # 修复异常值
    data.loc[data['vehicle_age_days'] < 0, 'vehicle_age_days'] = 0
    
    # 车辆年龄（年）
    data['vehicle_age_years'] = data['vehicle_age_days'] / 365
    
    # 注册年份和月份
    data['reg_year'] = data['regDate'].dt.year
    data['reg_month'] = data['regDate'].dt.month
    data['reg_day'] = data['regDate'].dt.day
    data['reg_dayofweek'] = data['regDate'].dt.dayofweek  # 星期几 (0-6, 0是星期一)
    data['reg_quarter'] = data['regDate'].dt.quarter  # 季度 (1-4)
    
    # 创建年份和月份
    data['creat_year'] = data['creatDate'].dt.year
    data['creat_month'] = data['creatDate'].dt.month
    data['creat_day'] = data['creatDate'].dt.day
    data['creat_dayofweek'] = data['creatDate'].dt.dayofweek  # 星期几 (0-6, 0是星期一)
    data['creat_quarter'] = data['creatDate'].dt.quarter  # 季度 (1-4)
    
    # 是否为新车（使用年限<1年）
    data['is_new_car'] = (data['vehicle_age_years'] < 1).astype(int)
    
    # 季节特征
    data['reg_season'] = data['reg_month'].apply(lambda x: (x%12 + 3)//3)
    data['creat_season'] = data['creat_month'].apply(lambda x: (x%12 + 3)//3)
    
    # 每年行驶的公里数
    data['km_per_year'] = data['kilometer'] / (data['vehicle_age_years'] + 0.1)
    
    # 车龄分段 - 更细致的分段
    data['age_segment'] = pd.cut(data['vehicle_age_years'], 
                                bins=[-0.01, 0.5, 1, 2, 3, 5, 7, 10, 15, 100], 
                                labels=['0-0.5年', '0.5-1年', '1-2年', '2-3年', '3-5年', '5-7年', '7-10年', '10-15年', '15年以上'])
    
    # 周期性时间特征 - 使用正弦和余弦变换捕捉月份和季节的周期性
    # 月份的周期性 (1-12)
    data['reg_month_sin'] = np.sin(2 * np.pi * data['reg_month'] / 12)
    data['reg_month_cos'] = np.cos(2 * np.pi * data['reg_month'] / 12)
    data['creat_month_sin'] = np.sin(2 * np.pi * data['creat_month'] / 12)
    data['creat_month_cos'] = np.cos(2 * np.pi * data['creat_month'] / 12)
    
    # 季节的周期性 (1-4)
    data['reg_season_sin'] = np.sin(2 * np.pi * data['reg_season'] / 4)
    data['reg_season_cos'] = np.cos(2 * np.pi * data['reg_season'] / 4)
    data['creat_season_sin'] = np.sin(2 * np.pi * data['creat_season'] / 4)
    data['creat_season_cos'] = np.cos(2 * np.pi * data['creat_season'] / 4)
    
    # 星期几的周期性 (0-6)
    data['reg_dayofweek_sin'] = np.sin(2 * np.pi * data['reg_dayofweek'] / 7)
    data['reg_dayofweek_cos'] = np.cos(2 * np.pi * data['reg_dayofweek'] / 7)
    data['creat_dayofweek_sin'] = np.sin(2 * np.pi * data['creat_dayofweek'] / 7)
    data['creat_dayofweek_cos'] = np.cos(2 * np.pi * data['creat_dayofweek'] / 7)
    
    # 时间差特征
    # 距离春节的时间（简化处理，以2月为春节月份）
    data['reg_days_from_spring_festival'] = abs(data['reg_month'] - 2) * 30 + data['reg_day']
    data['creat_days_from_spring_festival'] = abs(data['creat_month'] - 2) * 30 + data['creat_day']
    
    # 是否在周末注册/创建
    data['reg_is_weekend'] = (data['reg_dayofweek'] >= 5).astype(int)
    data['creat_is_weekend'] = (data['creat_dayofweek'] >= 5).astype(int)
    
    # 是否在月初/月末
    data['reg_is_month_start'] = (data['reg_day'] <= 5).astype(int)
    data['reg_is_month_end'] = (data['reg_day'] >= 25).astype(int)
    data['creat_is_month_start'] = (data['creat_day'] <= 5).astype(int)
    data['creat_is_month_end'] = (data['creat_day'] >= 25).astype(int)
    
    return data

# 创建时间特征
data = create_time_features(data)

def create_geo_features(data):
    """
    创建更精细的地理位置特征
    基于省份、城市等地理信息创建特征
    """
    print("创建精细地理位置特征...")
    
    # 提取省份信息（假设前两位数字代表省份编码）
    if 'regionCode' in data.columns:
        # 提取省份编码（前两位）
        data['province_code'] = data['regionCode'].astype(str).str[:2]
        
        # 提取城市编码（中间两位）
        data['city_code'] = data['regionCode'].astype(str).str[2:4]
        
        # 提取区县编码（最后两位）
        data['district_code'] = data['regionCode'].astype(str).str[4:6]
        
        # 将编码转换为分类特征
        data['province_code'] = data['province_code'].astype('category')
        data['city_code'] = data['city_code'].astype('category')
        data['district_code'] = data['district_code'].astype('category')
        
        # 计算每个省份的车辆数量
        province_counts = data.groupby('province_code').size()
        data['province_vehicle_count'] = data['province_code'].map(province_counts)
        
        # 计算每个城市的车辆数量
        city_counts = data.groupby('city_code').size()
        data['city_vehicle_count'] = data['city_code'].map(city_counts)
        
        # 计算每个区县的车辆数量
        district_counts = data.groupby('district_code').size()
        data['district_vehicle_count'] = data['district_code'].map(district_counts)
        
        # 创建省份与城市的组合特征
        data['province_city'] = data['province_code'].astype(str) + '_' + data['city_code'].astype(str)
        data['province_city'] = data['province_city'].astype('category')
        
        # 创建高级地理特征 - 省份与车型的交互
        if 'model' in data.columns:
            data['province_model'] = data['province_code'].astype(str) + '_' + data['model'].astype(str)
            data['province_model'] = data['province_model'].astype('category')
    
    # 使用经纬度信息（如果有的话）
    if 'latitude' in data.columns and 'longitude' in data.columns:
        # 计算与主要城市的距离
        major_cities = {
            '北京': (39.9042, 116.4074),
            '上海': (31.2304, 121.4737),
            '广州': (23.1291, 113.2644),
            '深圳': (22.5431, 114.0579),
            '成都': (30.5728, 104.0668),
            '重庆': (29.4316, 106.9123),
            '杭州': (30.2741, 120.1551)
        }
        
        for city, (lat, lon) in major_cities.items():
            # 使用欧几里得距离作为简化计算
            data[f'distance_to_{city}'] = np.sqrt(
                (data['latitude'] - lat)**2 + (data['longitude'] - lon)**2
            )
    
    # 使用KMeans聚类创建地理位置聚类特征
    if 'regionCode' in data.columns:
        # 将regionCode转换为数值
        region_codes = data['regionCode'].astype(float).values.reshape(-1, 1)
        
        # 应用KMeans聚类（选择5个聚类）
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        data['geo_cluster'] = kmeans.fit_predict(region_codes)
        data['geo_cluster'] = data['geo_cluster'].astype('category')
        
        # 计算到聚类中心的距离作为特征
        cluster_centers = kmeans.cluster_centers_
        for i, center in enumerate(cluster_centers):
            data[f'distance_to_cluster_{i}'] = abs(data['regionCode'].astype(float) - center[0])
        
        # 创建地理位置密度特征
        region_density = data.groupby('regionCode').size() / len(data)
        data['region_density'] = data['regionCode'].map(region_density)
        
        # 创建地理位置热度特征 - 基于车辆价格
        if 'price' in data.columns:
            region_price_mean = data.groupby('regionCode')['price'].mean()
            data['region_price_level'] = data['regionCode'].map(region_price_mean)
    
    return data

# 创建地理位置特征
data = create_geo_features(data)

data[['regDate', 'creatDate', 'reg_year', 'reg_month', 'reg_day', 'creat_year', 'creat_month', 'creat_day', 
      'vehicle_age_days', 'vehicle_age_years', 'is_new_car', 'reg_season', 'creat_season', 'km_per_year', 'age_segment']]


# In[3]:




def create_car_features(data):
    """
    创建车辆特征
    增强版：添加更多车辆特征和交互特征
    """
    print("创建增强车辆特征...")
    
    # 缺失值处理
    numerical_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    for feature in numerical_features:
        # 标记缺失值
        data[f'{feature}_missing'] = data[feature].isnull().astype(int)
        # 填充缺失值
        data[feature] = data[feature].fillna(data[feature].median())
    
    # 将model转换为数值型特征
    data['model_num'] = data['model'].astype('category').cat.codes
    
    # 品牌与车型组合
    data['brand_model'] = data['brand'].astype(str) + '_' + data['model'].astype(str)
        
    # 相对年份特征
    current_year = datetime.datetime.now().year
    data['car_age_from_now'] = current_year - data['reg_year']
    
    # 处理异常值
    numerical_cols = ['power', 'kilometer', 'v_0']
    for col in numerical_cols:
        Q1 = data[col].quantile(0.05)
        Q3 = data[col].quantile(0.95)
        IQR = Q3 - Q1
        data[f'{col}_outlier'] = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).astype(int)
        data[col] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    # 创建交互特征
    # 功率与公里数的比率 - 表示车辆的使用强度
    data['power_km_ratio'] = data['power'] / (data['kilometer'] + 1)
    
    # 功率与车龄的比率 - 表示车辆的性能保持程度
    data['power_age_ratio'] = data['power'] / (data['vehicle_age_years'] + 0.1)
    
    # 公里数与车龄的比率 - 表示年均行驶里程
    data['km_age_ratio'] = data['kilometer'] / (data['vehicle_age_years'] + 0.1)
    
    # 创建多项式特征（针对重要的数值特征）
    important_features = ['power', 'kilometer', 'vehicle_age_years']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    
    # 提取这些特征并创建多项式特征
    poly_features = poly.fit_transform(data[important_features])
    
    # 获取特征名称
    feature_names = poly.get_feature_names_out(important_features)
    
    # 将多项式特征添加到数据中
    for i, name in enumerate(feature_names):
        if '_' in name:  # 只保留交互特征，跳过原始特征
            data[f'poly_{name}'] = poly_features[:, i]
    
    return data

# 创建车辆特征
data = create_car_features(data)
data


# In[4]:


def create_statistical_features(data, train_idx):
    """
    创建多级别统计特征
    增加了品牌+型号、车身类型、燃油类型等多个级别的统计特征
    """
    print("创建多级别统计特征...")
    
    # 仅使用训练集数据创建统计特征
    train_data = data.iloc[train_idx].reset_index(drop=True)
    
    # 定义要计算的统计量
    agg_funcs = {
        'price': ['mean', 'median', 'std', 'min', 'max', 'count', 
                 lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }
    
    # 重命名聚合函数的输出列
    agg_funcs_names = {
        'price': {
            'mean': 'price_mean', 
            'median': 'price_median', 
            'std': 'price_std', 
            'min': 'price_min', 
            'max': 'price_max', 
            'count': 'price_count',
            '<lambda_0>': 'price_25percentile',
            '<lambda_1>': 'price_75percentile'
        }
    }
    
    # 1. 品牌级别统计
    brand_stats = train_data.groupby('brand').agg(agg_funcs)
    brand_stats.columns = ['brand_' + agg_funcs_names['price'][col[1]] for col in brand_stats.columns]
    brand_stats = brand_stats.reset_index()
    
    # 2. 品牌+型号级别统计
    brand_model_stats = train_data.groupby(['brand', 'model']).agg(agg_funcs)
    brand_model_stats.columns = ['brand_model_' + agg_funcs_names['price'][col[1]] for col in brand_model_stats.columns]
    brand_model_stats = brand_model_stats.reset_index()
    
    # 3. 车身类型级别统计
    body_type_stats = train_data.groupby('bodyType').agg(agg_funcs)
    body_type_stats.columns = ['body_type_' + agg_funcs_names['price'][col[1]] for col in body_type_stats.columns]
    body_type_stats = body_type_stats.reset_index()
    
    # 4. 燃油类型级别统计
    fuel_type_stats = train_data.groupby('fuelType').agg(agg_funcs)
    fuel_type_stats.columns = ['fuel_type_' + agg_funcs_names['price'][col[1]] for col in fuel_type_stats.columns]
    fuel_type_stats = fuel_type_stats.reset_index()
    
    # 5. 品牌+车身类型级别统计
    brand_body_stats = train_data.groupby(['brand', 'bodyType']).agg(agg_funcs)
    brand_body_stats.columns = ['brand_body_' + agg_funcs_names['price'][col[1]] for col in brand_body_stats.columns]
    brand_body_stats = brand_body_stats.reset_index()
    
    # 6. 车龄分段级别统计
    age_segment_stats = train_data.groupby('age_segment').agg(agg_funcs)
    age_segment_stats.columns = ['age_segment_' + agg_funcs_names['price'][col[1]] for col in age_segment_stats.columns]
    age_segment_stats = age_segment_stats.reset_index()
    
    # 合并所有统计特征
    data = data.merge(brand_stats, on='brand', how='left')
    data = data.merge(brand_model_stats, on=['brand', 'model'], how='left')
    data = data.merge(body_type_stats, on='bodyType', how='left')
    data = data.merge(fuel_type_stats, on='fuelType', how='left')
    data = data.merge(brand_body_stats, on=['brand', 'bodyType'], how='left')
    data = data.merge(age_segment_stats, on='age_segment', how='left')
    
    # 计算价格比率特征
    data['brand_price_ratio'] = data['brand_price_mean'] / data['brand_price_mean'].mean()
    data['brand_model_price_ratio'] = data['brand_model_price_mean'] / data['brand_model_price_mean'].mean()
    data['body_type_price_ratio'] = data['body_type_price_mean'] / data['body_type_price_mean'].mean()
    
    # 计算价格范围特征
    data['brand_price_range'] = data['brand_price_max'] - data['brand_price_min']
    data['brand_model_price_range'] = data['brand_model_price_max'] - data['brand_model_price_min']
    data['body_type_price_range'] = data['body_type_price_max'] - data['body_type_price_min']
    
    # 计算价格变异系数 (CV = 标准差/均值)
    data['brand_price_cv'] = data['brand_price_std'] / (data['brand_price_mean'] + 1e-5)
    data['brand_model_price_cv'] = data['brand_model_price_std'] / (data['brand_model_price_mean'] + 1e-5)
    data['body_type_price_cv'] = data['body_type_price_std'] / (data['body_type_price_mean'] + 1e-5)
    
    # 计算四分位数范围 (IQR = Q3 - Q1)
    data['brand_price_iqr'] = data['brand_price_75percentile'] - data['brand_price_25percentile']
    data['brand_model_price_iqr'] = data['brand_model_price_75percentile'] - data['brand_model_price_25percentile']
    data['body_type_price_iqr'] = data['body_type_price_75percentile'] - data['body_type_price_25percentile']
    
    return data

# 找回训练集的索引
train_idx = data[data['source'] == 'train'].index
test_idx = data[data['source'] == 'test'].index

# 创建统计特征
data = create_statistical_features(data, train_idx)

# 创建高级特征
def create_advanced_features(data):
    """
    创建高级特征工程技术
    包括对数变换、平方根变换、分箱特征等
    处理NaN值以确保特征创建过程不会出错
    """
    print("创建高级特征...")
    
    # 检查并填充NaN值
    numerical_cols = ['power', 'kilometer']
    for col in numerical_cols:
        if data[col].isna().any():
            print(f"检测到{col}列有NaN值，使用中位数填充")
            data[col] = data[col].fillna(data[col].median())
    
    # 对数变换 - 适用于偏斜分布的特征
    for col in numerical_cols:
        data[f'log_{col}'] = np.log1p(data[col])
    
    # 平方根变换 - 减轻右偏分布
    for col in numerical_cols:
        data[f'sqrt_{col}'] = np.sqrt(data[col])
    
    # 分箱特征 - 将连续特征转换为分类特征
    try:
        # 功率分箱
        data['power_bin'] = pd.qcut(data['power'], q=10, labels=False, duplicates='drop')
        
        # 公里数分箱
        data['kilometer_bin'] = pd.qcut(data['kilometer'], q=10, labels=False, duplicates='drop')
        
        # 价格比率分箱 (仅对训练集有效)
        if 'price_ratio' in data.columns:
            # 确保没有NaN值
            data['price_ratio_bin'] = pd.qcut(data['price_ratio'].fillna(-1), q=10, labels=False, duplicates='drop')
    except Exception as e:
        print(f"创建分箱特征时出错: {e}")
        print("尝试使用替代方法创建分箱特征...")
        
        # 使用cut作为替代方法（基于固定间隔而非分位数）
        data['power_bin'] = pd.cut(data['power'], bins=10, labels=False)
        data['kilometer_bin'] = pd.cut(data['kilometer'], bins=10, labels=False)
        
        if 'price_ratio' in data.columns:
            data['price_ratio_bin'] = pd.cut(data['price_ratio'].fillna(-1), bins=10, labels=False)
    
    return data

data = create_advanced_features(data)
data


# In[5]:


def encode_categorical_features(data):
    """
    编码分类特征
    """
    print("编码分类特征...")
    
    # 目标编码的替代方案 - 频率编码
    categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_cols:
        # 填充缺失值
        data[col] = data[col].fillna('未知')
        
        # 频率编码
        freq_encoding = data.groupby(col).size() / len(data)
        data[f'{col}_freq'] = data[col].map(freq_encoding)
    
    # 将分类变量转换为CatBoost可以识别的格式
    for col in categorical_cols:
        data[col] = data[col].astype('str')
    
    return data, categorical_cols

# 编码分类特征
data, categorical_cols = encode_categorical_features(data)
data


# In[6]:


categorical_cols


# In[7]:


def feature_selection(data, categorical_cols):
    """
    特征选择和最终数据准备
    """
    print("特征选择和最终数据准备...")
    
    # 删除不再需要的列, 所有车offerType=0,seller只有1个为1，其他都为0
    drop_cols = ['regDate', 'creatDate', 'price', 'SaleID', 'name', 'offerType', 'seller', 'source']
    data = data.drop(drop_cols, axis=1, errors='ignore')
    
    # 确保所有分类特征都被正确标记
    # 添加age_segment到分类特征列表中
    if 'age_segment' not in categorical_cols and 'age_segment' in data.columns:
        categorical_cols.append('age_segment')
    
    # 确保brand_model也被标记为分类特征
    if 'brand_model' not in categorical_cols and 'brand_model' in data.columns:
        categorical_cols.append('brand_model')
    
    # 添加地理位置相关的分类特征
    geo_categorical_features = ['province_code', 'city_code', 'district_code', 'province_city', 'geo_cluster', 'province_model']
    for col in geo_categorical_features:
        if col in data.columns and col not in categorical_cols:
            categorical_cols.append(col)
            
    # 添加车辆计数特征到分类特征列表
    count_features = ['province_vehicle_count', 'city_vehicle_count', 'district_vehicle_count']
    for col in count_features:
        if col in data.columns and col not in categorical_cols:
            # 将这些计数特征转换为字符串类型，以便CatBoost将其视为分类特征
            data[col] = data[col].astype('str')
            categorical_cols.append(col)
    
    # 添加分箱特征
    bin_features = [col for col in data.columns if col.endswith('_bin')]
    for col in bin_features:
        if col in data.columns and col not in categorical_cols:
            categorical_cols.append(col)
    
    # 转换分类特征
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    return data, categorical_cols

def apply_feature_selection(X_train, y_train, X_test, k=100):
    """
    应用特征选择，选择最重要的特征
    处理NaN值并应用SelectKBest
    """
    print(f"应用特征选择，选择前{k}个最重要的特征...")
    
    # 只对数值型特征应用特征选择
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # 处理NaN值 - 在应用特征选择前填充数值特征的NaN值
    X_train_numerical = X_train[numerical_cols].copy()
    
    # 检查并填充NaN值
    has_nan = X_train_numerical.isna().any().any()
    if has_nan:
        print("检测到NaN值，正在填充...")
        # 使用中位数填充NaN值
        X_train_numerical = X_train_numerical.fillna(X_train_numerical.median())
    
    # 创建特征选择器
    selector = SelectKBest(f_regression, k=min(k, len(numerical_cols)))
    
    # 应用特征选择
    X_train_selected = selector.fit_transform(X_train_numerical, y_train)
    
    # 获取选择的特征索引
    selected_indices = selector.get_support(indices=True)
    selected_features = [numerical_cols[i] for i in selected_indices]
    
    # 合并选择的数值特征和所有分类特征
    final_features = selected_features + categorical_cols
    
    print(f"特征数量从 {X_train.shape[1]} 减少到 {len(final_features)}")
    
    # 确保测试集也处理了NaN值
    if has_nan and X_test is not None:
        for col in selected_features:
            if col in X_test.columns and X_test[col].isna().any():
                X_test[col] = X_test[col].fillna(X_train[col].median())
    
    return X_train[final_features], X_test[final_features], final_features

# 特征选择和最终准备
data, cat_features = feature_selection(data, categorical_cols)
data


# In[11]:


#data[['offerType', 'seller']]
#data['offerType'].value_counts()
#data['seller'].value_counts()


# In[13]:


# 分离训练集和测试集
X_train_full = data.iloc[train_idx].reset_index(drop=True)
X_test = data.iloc[test_idx].reset_index(drop=True)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y, test_size=0.1, random_state=42
)

# 应用特征选择
X_train, X_test, selected_features = apply_feature_selection(X_train, y_train, X_test, k=150)

# 更新验证集特征，只保留选择的特征
X_val = X_val[selected_features]

# 更新分类特征列表，只保留选择的特征中的分类特征
cat_features = [feat for feat in cat_features if feat in selected_features]

# 保存处理后的数据
joblib.dump(X_train, 'processed_data/fe_enhanced_X_train.joblib')
joblib.dump(X_val, 'processed_data/fe_enhanced_X_val.joblib')
joblib.dump(y_train, 'processed_data/fe_enhanced_y_train.joblib')
joblib.dump(y_val, 'processed_data/fe_enhanced_y_val.joblib')
joblib.dump(X_test, 'processed_data/fe_enhanced_test_data.joblib')
joblib.dump(test_ids, 'processed_data/fe_enhanced_sale_ids.joblib')
joblib.dump(cat_features, 'processed_data/fe_enhanced_cat_features.joblib')

print("增强特征工程后的数据已保存")


# In[9]:


def train_catboost_model(X_train, X_val, y_train, y_val, cat_features):
    """
    训练CatBoost模型
    """
    print("\n开始训练增强特征的CatBoost模型...")
    
    # 创建数据池
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # 设置模型参数
    params = {
        'iterations': 10000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bayesian',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 100,
        'verbose': 100,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1
    }
    
    # 创建模型
    model = CatBoostRegressor(**params)
    
    # 训练模型
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=True
    )
    
    # 保存模型
    model.save_model('processed_data/fe_enhanced_catboost_model_v2.cbm')
    print("增强模型已保存到 processed_data/fe_enhanced_catboost_model_v2.cbm")
    
    return model

# 训练CatBoost模型
model = train_catboost_model(X_train, X_val, y_train, y_val, cat_features)


# In[10]:


def evaluate_model(model, X_val, y_val, cat_features):
    """
    评估模型性能
    """
    # 创建验证数据池
    val_pool = Pool(X_val, cat_features=cat_features)
    
    # 预测
    y_pred = model.predict(val_pool)
    
    # 计算评估指标
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("\n模型评估结果：")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R2分数: {r2:.4f}")
    
    # 绘制预测值与实际值的对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('CatBoost预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig('fe_catboost_prediction_vs_actual.png')
    plt.close()
    
    return rmse, mae, r2

# 评估模型
rmse, mae, r2 = evaluate_model(model, X_val, y_val, cat_features)


# In[11]:


def plot_feature_importance(model, X_train):
    """
    绘制特征重要性图
    """
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 保存特征重要性到CSV
    importance_df.to_csv('fe_catboost_feature_importance.csv', index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('CatBoost Top 20 特征重要性')
    plt.tight_layout()
    plt.savefig('fe_catboost_feature_importance.png')
    plt.close()
    
    return importance_df
    
# 绘制特征重要性
importance_df = plot_feature_importance(model, X_train)


# In[14]:


def predict_test_data(model, X_test, test_ids, cat_features):
    """
    预测测试集数据
    """
    print("\n正在预测测试集...")
    
    # 创建测试数据池
    test_pool = Pool(X_test, cat_features=cat_features)
    
    # 预测
    predictions = model.predict(test_pool)
    
    # 创建提交文件
    submit_data = pd.DataFrame({
        'SaleID': test_ids,
        'price': predictions
    })
    
    # 保存预测结果
    submit_data.to_csv('fe_enhanced_catboost_submit_result.csv', index=False)
    print("预测结果已保存到 fe_enhanced_catboost_submit_result.csv")

# 预测测试集
predict_test_data(model, X_test, test_ids, cat_features)

print("\n增强特征模型训练、评估和预测完成！")
print(f"Top 10 重要特征:\n{importance_df.head(10)}")
print("\n增强特征包括:")
print("1. 精细地理位置特征 - 省份、城市、区县编码及其统计量")
print("2. 地理位置聚类特征 - 使用KMeans聚类")
print("3. 高级特征工程 - 对数变换、平方根变换、分箱特征")
print("4. 特征选择 - 使用SelectKBest选择最重要的特征")


# In[37]:


# X_test['brand_model'].isnull().sum()
# cat_features
# X_test['vehicle_age_years'].describe()
# #X_test['vehicle_age_years'].isnull().sum()
# data.loc[data['age_segment'].isnull(), 'vehicle_age_years']


# In[12]:


# # 车龄分段
# data['age_segment'] = pd.cut(data['vehicle_age_years'], 
#                             bins=[-0.01, 1, 3, 5, 10, 100], 
#                             labels=['0-1年', '1-3年', '3-5年', '5-10年', '10年以上'])


# In[16]:


#cat_features

