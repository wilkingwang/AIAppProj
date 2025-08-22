import os
import json
import base64
import asyncio
import time
import dashscope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from sqlalchemy import column, create_engine

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']   # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义资源文件根目录
ROOT_RESOURCE_DIR = os.path.join(os.path.dirname(__file__), 'resources')

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

system_prompt = '''
你是一个门票助手，以下是关于门票订单表相关的字段，我可以编写对应的SQL，对数据进行查询
-- 门票订单表
CREATE TABLE tkt_orders (
    order_time DATETIME,             -- 订单日期
    account_id INT,                  -- 预定用户ID
    gov_id VARCHAR(18),              -- 商品使用人ID（身份证号）
    gender VARCHAR(10),              -- 使用人性别
    age INT,                         -- 年龄
    province VARCHAR(30),           -- 使用人省份
    SKU VARCHAR(100),                -- 商品SKU名
    product_serial_no VARCHAR(30),  -- 商品ID
    eco_main_order_id VARCHAR(20),  -- 订单ID
    sales_channel VARCHAR(20),      -- 销售渠道
    status VARCHAR(30),             -- 商品状态
    order_value DECIMAL(10,2),       -- 订单金额
    quantity INT                     -- 商品数量
);
一日门票，对应多种SKU：
Universal Studios Beijing One-Day Dated Ticket-Standard
Universal Studios Beijing One-Day Dated Ticket-Child
Universal Studios Beijing One-Day Dated Ticket-Senior
二日门票，对应多种SKU：
USB 1.5-Day Dated Ticket Standard
USB 1.5-Day Dated Ticket Discounted
一日门票、二日门票查询
SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END) AS one_day_ticket_sales,
SUM(CASE WHEN SKU LIKE 'USB%' THEN quantity ELSE 0 END) AS two_day_ticket_sales
我将回答用户关于门票相关的问题

每当 exc_sql 工具返回 markdown 表格和图表时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
'''

functions_desc = [
    {
        'name': 'exc_sql',
        'description': '对于生成的SQL，进行SQL查询',
        'parameters': {
            'type': 'object',
            'properties': {
                'sql_input': {
                    'type': 'string',
                    'description': '生成的SQL语句'
                }
            },
            'required': ['sql_input']
        }
    }
]

# 会话隔离 DataFrame 存储
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    
    return None

@register_tool('exc_sql')
class ExcSqlTool(BaseTool):
    name = 'exc_sql'
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [
        {
            'name': 'sql_input',
            'type': 'string',
            'description': '生成的SQL语句',
            'required': True
        }
    ]

    def call(self, params: str,  **kwargs) -> str:
        """执行SQL查询"""
        args = json.loads(params)
        sql_input = args['sql_input']
        database = args.get('database', 'ubr')

        engine = create_engine(
            f'mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/{database}?charset=utf8mb4',
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )

        try:
            df = pd.read_sql(sql_input, engine)
            md = df.head(10).to_markdown(index=False)

            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            # 自动生成文件名
            filename = f'bar_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)

            # 生成图表
            generate_char_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![柱状图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f'SQL执行或可视化失败：{e}'

# 通用可视化函数
def generate_char_png(df_sql, save_path):
    columns = df_sql.columns
    x = np.arange(len(df_sql))
    # 获取object类型
    object_columns = df_sql.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])

    num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()
    if len(num_columns) > 0:
        # 对数据进行透视，以便为每个日期和销售渠道创建堆积柱状图
        pivot_df = df_sql.pivot_table(
            index=columns[0], 
            columns=object_columns,
            values=num_columns,
            fill_value=0)
        
        # 绘制堆积柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        # 为每个销售渠道和票类型创建柱状图
        bottoms = None
        for col in pivot_df.columns:
            ax.bar(pivot_df.index, pivot_df[col], bottom=bottoms, label=str(col))
            if bottoms is None:
                bottoms = pivot_df[col].copy()

            else:
                bottoms += pivot_df[col]
    else:
        bottom = np.zeros(len(df_sql))
        for col in columns[1:]:
            plt.bar(x, df_sql[col], bottom=bottom, label=col)
            bottom += df_sql[col]
        
        plt.xticks(x, df_sql[columns[0]])
    plt.legend()
    plt.title("销售统计")
    plt.xlabel(columns[0])
    plt.ylabel("门票数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 初始化门票助手服务
def init_agent_service():
    """初始化门票助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }

    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票助手',
            description='门票查询与订单分析',
            system_message=system_prompt,
            function_list=['exc_sql']
        )
        print("门票助手初始化成功！")

        return bot
    except Exception as e:
        print(f'初始化门票助手服务失败：{e}')
        raise

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")

        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举2个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                "2023年4、5、6月一日门票，二日门票的销量多少？帮我按照周进行统计",
                '2023年7月的不同省份的入园人数统计',
                '帮我查看2023年10月1-7日销售渠道订单金额排名',
            ]
        }
        print('Web 界面准备就绪，正在启动服务...')
        # 启动 Web 服务
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f'启动 Web 界面失败：{e}')

if __name__ == '__main__':
    app_gui()
