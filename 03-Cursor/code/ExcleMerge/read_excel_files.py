import pandas as pd
import os

def read_excel_head(file_path, rows=5):
    """读取Excel文件的前几行数据"""
    try:
        df = pd.read_excel(file_path)
        print(f"\n文件: {os.path.basename(file_path)}")
        print(f"总行数: {len(df)}")
        print(f"前{rows}行数据:")
        print("-" * 50)
        print(df.head(rows))
        print("-" * 50)
        return df.head(rows)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def main():
    # 设置工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义文件路径
    performance_file = os.path.join(current_dir, "员工绩效表.xlsx")
    basic_info_file = os.path.join(current_dir, "员工基本信息表.xlsx")
    
    # 检查文件是否存在
    if not os.path.exists(performance_file):
        print(f"文件不存在: {performance_file}")
        return
    
    if not os.path.exists(basic_info_file):
        print(f"文件不存在: {basic_info_file}")
        return
    
    print("开始读取Excel文件...")
    
    # 读取员工绩效表
    performance_data = read_excel_head(performance_file)
    
    # 读取员工基本信息表
    basic_info_data = read_excel_head(basic_info_file)
    
    print("\n文件读取完成！")
    
    # 返回数据用于进一步处理
    return {
        "performance_data": performance_data,
        "basic_info_data": basic_info_data
    }

if __name__ == "__main__":
    result = main()