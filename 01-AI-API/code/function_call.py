import json
import os
import dashscope
from typing import Dict, Any, List
from dashscope.api_entities.dashscope_response import Role

api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

functions = [
    {
        "name": "get_current_weather",
        "description": "Returns real-time weather information.",
        "parameters": {
            "type": "object",
            "preperties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for, e.g., '河北省承德市双桥区'"
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Use your local temerature unit measurement",
                }
            },
            "required": ["location", "format"],
        }
    },
    {
        "name": "schedule_meeting",
        "description": "Schedules a meeting between two people.",
        "parameters": {
            "type": "object",
            "preperties": {
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendees' email addresses or names."
                },
                "date_time": {
                    "type": "string",
                    "description": "Date and time of the meeting in ISO format, e.g., '2023-10-01T14:00Z'"
                },
                "duration_minutes": {
                    "type": "number",
                    "description": "Duration of the meeting in minutes"
                },
                "topic": {
                    "type": "string",
                    "description": "Topic of the meeting"
                }
            },
            "required": ["attendees", "date_time", "duration_minutes", "topic"]
        }
    }
]

initial_message = [
    {
        "role": "system",
        "content": "You are an AI Assistant designed to help users by calling functions based on their requests."
    },
    {
        "role": "user",
        "content": "What's the current weather in Beijing?"
    }
]

def llm_call(message):
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=message,
            functions=functions,
            result_format='message'
        )

        return response.output.choices[0].message
    except Exception as ex:
        print(f"LLM call error: {str(ex)}")
        return{}
    
message = llm_call(initial_message)
print(message)