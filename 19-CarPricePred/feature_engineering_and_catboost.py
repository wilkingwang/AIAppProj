#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
二手车价格预测 - 高级特征工程与CatBoost建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

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
    创建时间特征
    """
    print("创建时间特征...")
    
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
    
    # 创建年份和月份
    data['creat_year'] = data['creatDate'].dt.year
    data['creat_month'] = data['creatDate'].dt.month
    data['creat_day'] = data['creatDate'].dt.day
    
    # 是否为新车（使用年限<1年）
    data['is_new_car'] = (data['vehicle_age_years'] < 1).astype(int)
    
    # 季节特征
    data['reg_season'] = data['reg_month'].apply(lambda x: (x%12 + 3)//3)
    data['creat_season'] = data['creat_month'].apply(lambda x: (x%12 + 3)//3)
    
    # 每年行驶的公里数
    data['km_per_year'] = data['kilometer'] / (data['vehicle_age_years'] + 0.1)
    
    # 车龄分段
    data['age_segment'] = pd.cut(data['vehicle_age_years'], 
                                bins=[-0.01, 1, 3, 5, 10, 100], 
                                labels=['0-1年', '1-3年', '3-5年', '5-10年', '10年以上'])
    
    return data

# 创建时间特征
data = create_time_features(data)
data[['regDate', 'creatDate', 'reg_year', 'reg_month', 'reg_day', 'creat_year', 'creat_month', 'creat_day', 
      'vehicle_age_days', 'vehicle_age_years', 'is_new_car', 'reg_season', 'creat_season', 'km_per_year', 'age_segment']]


# In[3]:


def create_car_features(data):
    """
    创建车辆特征
    """
    print("创建车辆特征...")
    
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
    
    # 品牌高级特征
    # 计算每个品牌的车型数量
    brand_model_counts = data.groupby('brand')['model'].nunique().reset_index()
    brand_model_counts.columns = ['brand', 'brand_model_count']
    data = data.merge(brand_model_counts, on='brand', how='left')
    
    # 计算每个品牌的平均功率
    brand_power_stats = data.groupby('brand')['power'].agg(['mean', 'std', 'max']).reset_index()
    brand_power_stats.columns = ['brand', 'brand_power_mean', 'brand_power_std', 'brand_power_max']
    data = data.merge(brand_power_stats, on='brand', how='left')
    
    # 计算每个品牌的平均里程
    brand_km_stats = data.groupby('brand')['kilometer'].agg(['mean', 'std']).reset_index()
    brand_km_stats.columns = ['brand', 'brand_km_mean', 'brand_km_std']
    data = data.merge(brand_km_stats, on='brand', how='left')
    
    # 车型高级特征
    # 计算每个车型的平均功率和里程
    model_stats = data.groupby(['brand', 'model']).agg(
        model_power_mean=('power', 'mean'),
        model_km_mean=('kilometer', 'mean'),
        model_count=('SaleID', 'count')
    ).reset_index()
    data = data.merge(model_stats, on=['brand', 'model'], how='left')
    
    # 计算车型在品牌中的受欢迎程度（占比）
    brand_counts = data.groupby('brand')['SaleID'].count().reset_index()
    brand_counts.columns = ['brand', 'brand_total_count']
    model_counts = data.groupby(['brand', 'model'])['SaleID'].count().reset_index()
    model_counts.columns = ['brand', 'model', 'model_count']
    model_counts = model_counts.merge(brand_counts, on='brand', how='left')
    model_counts['model_popularity'] = model_counts['model_count'] / model_counts['brand_total_count']
    model_counts = model_counts[['brand', 'model', 'model_popularity']]
    data = data.merge(model_counts, on=['brand', 'model'], how='left')
    
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
    
    # 新增交互特征 - 品牌与车身类型交互
    print("添加品牌与车身类型交互特征...")
    # 创建品牌与车身类型的组合特征
    data['brand_bodyType'] = data['brand'].astype(str) + '_' + data['bodyType'].astype(str)
    
    # 计算每个品牌-车身类型组合的车辆数量
    brand_bodyType_counts = data.groupby(['brand', 'bodyType']).size().reset_index(name='brand_bodyType_count')
    data = data.merge(brand_bodyType_counts, on=['brand', 'bodyType'], how='left')
    
    # 计算每个品牌-车身类型组合的平均功率
    brand_bodyType_power = data.groupby(['brand', 'bodyType'])['power'].mean().reset_index(name='brand_bodyType_power_mean')
    data = data.merge(brand_bodyType_power, on=['brand', 'bodyType'], how='left')
    
    # 计算每个品牌-车身类型组合的平均里程
    brand_bodyType_km = data.groupby(['brand', 'bodyType'])['kilometer'].mean().reset_index(name='brand_bodyType_km_mean')
    data = data.merge(brand_bodyType_km, on=['brand', 'bodyType'], how='left')
    
    # 计算该品牌-车身类型组合在该品牌中的占比
    brand_total_counts = data.groupby('brand').size().reset_index(name='brand_total_count')
    data = data.merge(brand_total_counts, on='brand', how='left')
    data['brand_bodyType_ratio'] = data['brand_bodyType_count'] / data['brand_total_count']
    
    # 新增交互特征 - 品牌与变速箱交互
    print("添加品牌与变速箱交互特征...")
    # 创建品牌与变速箱的组合特征
    data['brand_gearbox'] = data['brand'].astype(str) + '_' + data['gearbox'].astype(str)
    
    # 计算每个品牌-变速箱组合的车辆数量
    brand_gearbox_counts = data.groupby(['brand', 'gearbox']).size().reset_index(name='brand_gearbox_count')
    data = data.merge(brand_gearbox_counts, on=['brand', 'gearbox'], how='left')
    
    # 计算每个品牌-变速箱组合的平均功率
    brand_gearbox_power = data.groupby(['brand', 'gearbox'])['power'].mean().reset_index(name='brand_gearbox_power_mean')
    data = data.merge(brand_gearbox_power, on=['brand', 'gearbox'], how='left')
    
    # 计算该品牌-变速箱组合在该品牌中的占比
    data['brand_gearbox_ratio'] = data['brand_gearbox_count'] / data['brand_total_count']
    
    # 新增交互特征 - 车龄与行驶里程交互
    print("添加车龄与行驶里程交互特征...")
    # 创建车龄与里程的直接交互（乘积）
    data['age_km'] = data['vehicle_age_years'] * data['kilometer']
    
    # 创建车龄与里程的比率特征（每年平均行驶里程）
    data['km_per_year'] = data['kilometer'] / (data['vehicle_age_years'] + 0.1)  # 加0.1避免除零
    
    # 创建车龄分段（更细致的分段）
    data['age_segment_detailed'] = pd.cut(data['vehicle_age_years'], 
                                         bins=[-0.01, 0.5, 1, 2, 3, 5, 8, 15, 100], 
                                         labels=['0-0.5年', '0.5-1年', '1-2年', '2-3年', '3-5年', '5-8年', '8-15年', '15年以上'])
    
    # 创建里程分段
    data['km_segment'] = pd.cut(data['kilometer'], 
                               bins=[0, 20000, 50000, 80000, 120000, 200000, 1000000], 
                               labels=['0-2万', '2-5万', '5-8万', '8-12万', '12-20万', '20万以上'])
    
    # 创建车龄-里程组合特征
    data['age_km_segment'] = data['age_segment_detailed'].astype(str) + '_' + data['km_segment'].astype(str)
    
    # 车龄与里程的非线性交互特征
    data['age_km_log'] = np.log1p(data['vehicle_age_years']) * np.log1p(data['kilometer'])
    data['age_km_sqrt'] = np.sqrt(data['vehicle_age_years']) * np.sqrt(data['kilometer'])
    data['age_km_square'] = (data['vehicle_age_years'] ** 2) * (data['kilometer'] ** 2)
    
    # 创建高强度使用指标（车龄小但里程高）
    data['high_intensity_usage'] = ((data['vehicle_age_years'] < 3) & (data['kilometer'] > 80000)).astype(int)
    
    # 创建低强度使用指标（车龄大但里程低）
    data['low_intensity_usage'] = ((data['vehicle_age_years'] > 8) & (data['kilometer'] < 30000)).astype(int)
    
    return data

# 创建车辆特征
data = create_car_features(data)
data


# In[4]:


def create_statistical_features(data, train_idx):
    """
    创建统计特征
    """
    print("创建统计特征...")
    
    # 仅使用训练集数据创建统计特征
    train_data = data.iloc[train_idx].reset_index(drop=True)
    
    # 品牌级别统计
    brand_stats = train_data.groupby('brand').agg(
        brand_price_mean=('price', 'mean'),
        brand_price_median=('price', 'median'),
        brand_price_std=('price', 'std'),
        brand_price_count=('price', 'count'),
        brand_price_min=('price', 'min'),
        brand_price_max=('price', 'max')
    ).reset_index()
    
    # 合并统计特征
    data = data.merge(brand_stats, on='brand', how='left')
    
    # 相对价格特征（相对于平均价格）
    data['brand_price_ratio'] = data['brand_price_mean'] / data['brand_price_mean'].mean()
    
    # 品牌价格范围
    data['brand_price_range'] = data['brand_price_max'] - data['brand_price_min']
    
    # 车型级别统计
    model_stats = train_data.groupby(['brand', 'model']).agg(
        model_price_mean=('price', 'mean'),
        model_price_median=('price', 'median'),
        model_price_std=('price', 'std'),
        model_price_count=('price', 'count')
    ).reset_index()
    
    # 合并车型统计特征
    data = data.merge(model_stats, on=['brand', 'model'], how='left')
    
    # 车型相对于品牌的价格比率
    data['model_to_brand_price_ratio'] = data['model_price_mean'] / (data['brand_price_mean'] + 1e-5)
    
    # 处理缺失值
    price_cols = [col for col in data.columns if 'price' in col and col != 'price']
    for col in price_cols:
        if data[col].isnull().sum() > 0:
            # 对于缺失的统计值，使用全局平均值填充
            data[col] = data[col].fillna(data[col].mean())
    
    return data

# 找回训练集的索引
train_idx = data[data['source'] == 'train'].index
test_idx = data[data['source'] == 'test'].index

# 创建统计特征
data = create_statistical_features(data, train_idx)
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
    
    # 添加新增的交互组合特征到分类特征列表
    interaction_features = ['brand_bodyType', 'brand_gearbox', 'age_km_segment', 'age_segment_detailed', 'km_segment']
    for feature in interaction_features:
        if feature not in categorical_cols and feature in data.columns:
            categorical_cols.append(feature)
    
    # 转换分类特征
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    return data, categorical_cols

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

# 保存处理后的数据
joblib.dump(X_train, 'processed_data/fe_X_train.joblib')
joblib.dump(X_val, 'processed_data/fe_X_val.joblib')
joblib.dump(y_train, 'processed_data/fe_y_train.joblib')
joblib.dump(y_val, 'processed_data/fe_y_val.joblib')
joblib.dump(X_test, 'processed_data/fe_test_data.joblib')
joblib.dump(test_ids, 'processed_data/fe_sale_ids.joblib')
joblib.dump(cat_features, 'processed_data/fe_cat_features.joblib')

print("预处理后的数据已保存")


# In[9]:


def train_catboost_model(X_train, X_val, y_train, y_val, cat_features):
    """
    训练CatBoost模型
    """
    print("\n开始训练CatBoost模型...")
    
    # 创建数据池
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # 设置模型参数 - 优化后的参数
    params = {
        'iterations': 6000,  # 增加迭代次数以适应更多特征
        'learning_rate': 0.02,  # 降低学习率以提高稳定性
        'depth': 7,  # 增加树深度以捕获更复杂的特征关系
        'l2_leaf_reg': 2.5,  # 调整正则化参数
        'bootstrap_type': 'Bayesian',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 150,  # 增加早停等待轮数
        'verbose': 100,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'task_type': 'CPU',
        'thread_count': -1,
        'feature_border_type': 'GreedyLogSum',  # 更适合处理数值型特征
        # 移除不兼容的leaf_estimation_method参数
        'nan_mode': 'Min'  # 处理缺失值的方式
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
    print("优化后的模型已保存到 processed_data/fe_enhanced_catboost_model_v2.cbm")
    
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
    print("优化后的预测结果已保存到 fe_enhanced_catboost_submit_result.csv")

# 保存优化后的数据
joblib.dump(X_train, 'processed_data/fe_enhanced_X_train.joblib')
joblib.dump(X_val, 'processed_data/fe_enhanced_X_val.joblib')
joblib.dump(y_train, 'processed_data/fe_enhanced_y_train.joblib')
joblib.dump(y_val, 'processed_data/fe_enhanced_y_val.joblib')
joblib.dump(X_test, 'processed_data/fe_enhanced_test_data.joblib')
joblib.dump(test_ids, 'processed_data/fe_enhanced_sale_ids.joblib')
joblib.dump(cat_features, 'processed_data/fe_enhanced_cat_features.joblib')
print("优化后的特征数据已保存")

# 预测测试集
predict_test_data(model, X_test, test_ids, cat_features)

print("\n优化后的模型训练、评估和预测完成！")
print(f"Top 10 重要特征:\n{importance_df.head(10)}")
print("\n品牌和车型特征优化总结:")
print("1. 添加了品牌级别的高级统计特征（车型数量、功率统计、里程统计）")
print("2. 添加了车型级别的统计特征（功率、里程、数量）")
print("3. 添加了车型在品牌中的受欢迎程度特征")
print("4. 添加了车型与品牌价格的比率特征")
print("5. 优化了模型参数以适应新增特征")


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

