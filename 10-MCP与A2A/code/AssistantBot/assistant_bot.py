import os
import asyncio
import dashscope
from typing import Optional
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

# 定义资源文件跟目录
ROOT_RESOURCE = os.path.join(os.path.abspath(__file__), 'resource')

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
dashscope.timeout = 30

def init_agent_service():
    """
    初始化高德地图助手服务
    """
    llm_cfg = {
        'model': 'qwen-turbo',
        'timeout': 30,
        'retry_count': 3,
    }

    system_prompt = f"""
    你扮演一个地图助手，你具有查询地图、规划线路、推荐景点等能力。
    你可以帮助用户规划旅游线路，查找景点，导航等。
    你应该充分利用高德地图的各种功能来提供专业的建议。
    """
    tools = [
        {
            "type": "mcpServer",
            "mcpServers": {
                "fetch": { 
                    "type": "sse",
                    "url": "https://mcp.api-inference.modelscope.net/11941e2bf9a942/sse"
                },
                # "bing-cn-mcp-server": {
                #     "type": "sse",
                #     "url": "https://mcp.api-inference.modelscope.net/604ac7c21d9043/sse"
                # }
                "amap-maps": {
                    "command": "npx",
                    "args": ["-y", "@amap/amap-maps-mcp-server"],
                    "env": {
                        "AMAP_MAPS_API_KEY": os.getenv('AMAP_API_KEY')
                    }
                }
            }
        }
    ]

    try:
        bot = Assistant(
            llm=llm_cfg,
            name='AI助手',
            description='地图查询/指定网页获取/Bing搜索',
            system_message=system_prompt,
            function_list=tools,
        )
        print(f'初始化地图助手服务成功!')

        return bot
    except Exception as e:
        print(f'初始化地图助手服务失败: {e}')
        raise

def app_gui():
    try:
        print(f'初始化地图助手服务...')
        bot = init_agent_service()

        chatbot_config  = {
            'prompt.suggestions': [
                '将 https://k.sina.com.cn/article_7732457677_1cce3f0cd01901eeeq.html 网页转化为Markdown格式',
                '帮我找一下静安寺附近的停车场',
                '推荐陆家嘴附近的高档餐厅',
                '帮我搜索一下关于AI的最新新闻'
            ]
        }

        print('Web 界面准备就绪，正在启动服务...')
        WebUI(
            bot,
            chatbot_config=chatbot_config ,
        ).run()
    except Exception as e:
        print(f'启动 Web 界面服务失败: {e}')
        raise

if __name__ == '__main__':
    app_gui()
