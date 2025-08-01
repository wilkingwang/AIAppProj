import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置文件路径
file_path = "香港各区疫情数据_20250322.xlsx"

def analyze_data():
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        print("文件读取成功！")
        print(f"数据总行数: {len(df)}")
        print(f"数据总列数: {len(df.columns)}")
        
        # 显示列名
        print("\n列名:")
        print("=" * 80)
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        
        # 显示前10行数据
        print("\n前10行数据:")
        print("=" * 80)
        print(df.head(10))
        
        return df
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        print("请确保文件在当前目录中")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        print("请确保已安装pandas和openpyxl库")
        print("可以使用: pip install pandas openpyxl matplotlib seaborn 来安装")
        return None

def plot_new_cases(df):
    if df is None:
        return
    
    try:
        # 按地区绘制新增确诊趋势
        plot_district_cases(df)
        
    except Exception as e:
        print(f"绘图时出错: {str(e)}")
        print("尝试使用备用绘图方法...")
        simple_plot(df)

def plot_district_cases(df):
    """按地区绘制新增确诊趋势图"""
    try:
        # 识别关键列
        date_col = '报告日期'
        district_col = '地区名称'
        case_col = '新增确诊'
        
        print(f"\n按地区绘制新增确诊趋势图...")
        print(f"日期列: {date_col}")
        print(f"地区列: {district_col}")
        print(f"确诊列: {case_col}")
        
        # 数据清理
        plot_df = df[[date_col, district_col, case_col]].copy()
        plot_df = plot_df.dropna()
        
        # 转换数据类型
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        plot_df[case_col] = pd.to_numeric(plot_df[case_col], errors='coerce')
        plot_df = plot_df.dropna()
        
        # 获取所有地区
        districts = plot_df[district_col].unique()
        print(f"\n发现 {len(districts)} 个地区: {', '.join(districts[:5])}...")
        
        # 创建大图显示所有地区
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # 为每个地区生成不同颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(districts)))
        
        for i, district in enumerate(districts):
            district_data = plot_df[plot_df[district_col] == district].copy()
            district_data = district_data.sort_values(date_col)
            
            ax.plot(district_data[date_col], district_data[case_col], 
                   marker='o', linewidth=2, markersize=3, 
                   color=colors[i], label=district, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title('香港各区新增确诊人数趋势对比', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('报告日期', fontsize=14)
        ax.set_ylabel('新增确诊人数', fontsize=14)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('香港各区新增确诊趋势对比图.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 '香港各区新增确诊趋势对比图.png'")
        
        # 显示图表
        plt.show()
        
        # 为每个地区创建单独图表
        create_individual_plots(plot_df, date_col, district_col, case_col)
        
    except Exception as e:
        print(f"按地区绘图出错: {str(e)}")
        
def create_individual_plots(df, date_col, district_col, case_col):
    """为每个地区创建单独的图表"""
    try:
        districts = df[district_col].unique()
        
        for district in districts:
            district_data = df[df[district_col] == district].copy()
            district_data = district_data.sort_values(date_col)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(district_data[date_col], district_data[case_col], 
                   marker='o', linewidth=2, markersize=4, color='#e74c3c')
            
            ax.set_title(f'{district} 新增确诊人数趋势', fontsize=16, fontweight='bold')
            ax.set_xlabel('报告日期', fontsize=12)
            ax.set_ylabel('新增确诊人数', fontsize=12)
            
            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(district_data)//10)))
            plt.xticks(rotation=45)
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            if len(district_data) > 0:
                max_cases = district_data[case_col].max()
                min_cases = district_data[case_col].min()
                avg_cases = district_data[case_col].mean()
                
                stats_text = f'最高: {int(max_cases)} | 最低: {int(min_cases)} | 平均: {avg_cases:.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            filename = f'{district}_新增确诊趋势.png'.replace('/', '_')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"已为 {len(districts)} 个地区创建单独图表")
        
    except Exception as e:
        print(f"创建单独图表出错: {str(e)}")

def simple_plot(df):
    """简化版绘图"""
    try:
        # 使用前两列进行绘图
        x_col, y_col = df.columns[0], df.columns[1]
        
        plt.figure(figsize=(12, 6))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.title('新增确诊人数趋势')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('新增确诊人数趋势图_简化版.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"简化绘图也失败: {str(e)}")

if __name__ == "__main__":
    # 分析数据
    df = analyze_data()
    
    # 绘制图表
    if df is not None:
        plot_new_cases(df)