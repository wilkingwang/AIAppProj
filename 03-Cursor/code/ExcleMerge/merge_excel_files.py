import pandas as pd
import os

def merge_employee_data():
    """合并员工基本信息和2024年第4季度绩效数据"""
    
    # 设置工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义文件路径
    basic_info_file = os.path.join(current_dir, "员工基本信息表.xlsx")
    performance_file = os.path.join(current_dir, "员工绩效表.xlsx")
    output_file = os.path.join(current_dir, "员工信息合并表.xlsx")
    
    # 检查文件是否存在
    if not os.path.exists(basic_info_file):
        print(f"文件不存在: {basic_info_file}")
        return
    
    if not os.path.exists(performance_file):
        print(f"文件不存在: {performance_file}")
        return
    
    try:
        # 读取员工基本信息表
        print("正在读取员工基本信息表...")
        basic_df = pd.read_excel(basic_info_file)
        print(f"基本信息表: {len(basic_df)} 条记录")
        
        # 读取绩效表
        print("正在读取员工绩效表...")
        performance_df = pd.read_excel(performance_file)
        print(f"绩效表: {len(performance_df)} 条记录")
        
        # 筛选2024年第4季度的绩效数据
        q4_2024_performance = performance_df[
            (performance_df['年度'] == 2024) & 
            (performance_df['季度'] == 4)
        ].copy()
        print(f"2024年第4季度绩效数据: {len(q4_2024_performance)} 条记录")
        
        # 重命名绩效列以便合并
        q4_2024_performance = q4_2024_performance.rename(columns={'绩效评分': '2024Q4绩效'})
        
        # 只保留需要的列
        q4_2024_performance = q4_2024_performance[['员工ID', '2024Q4绩效']]
        
        # 合并数据（左连接，保留所有员工基本信息）
        print("正在合并数据...")
        merged_df = pd.merge(
            basic_df, 
            q4_2024_performance, 
            on='员工ID', 
            how='left'
        )
        
        # 显示合并结果
        print("\n合并完成！")
        print(f"合并后总记录数: {len(merged_df)}")
        print("\n前5行合并结果:")
        print(merged_df.head())
        
        # 检查是否有员工没有绩效数据
        no_performance = merged_df[merged_df['2024Q4绩效'].isna()]
        if len(no_performance) > 0:
            print(f"\n注意: {len(no_performance)} 名员工没有2024年第4季度绩效数据")
            print("这些员工的员工ID:")
            print(no_performance['员工ID'].tolist())
        
        # 保存合并结果
        print(f"\n正在保存合并结果到: {output_file}")
        merged_df.to_excel(output_file, index=False)
        print("保存完成！")
        
        return merged_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

def main():
    print("开始合并员工信息和绩效数据...")
    print("=" * 50)
    
    result = merge_employee_data()
    
    if result is not None:
        print("\n" + "=" * 50)
        print("合并成功！文件已保存为: 员工信息合并表.xlsx")
    else:
        print("\n合并失败！")

if __name__ == "__main__":
    main()