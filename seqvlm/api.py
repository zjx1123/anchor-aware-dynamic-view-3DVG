'''
VLM调用:API
'''
import io
import os
import yaml
from openai import OpenAI
from mmengine.utils.dl_utils import TimeCounter


with open("../config.yaml", "r") as f:
    configs = yaml.safe_load(f)

_PLACEHOLDERS = {"", "<YOUR_API_KEY>", "<YOUR_API_BASE_URL>", "<YOUR_MODEL_NAME>"}
_ENV_OVERRIDES = {
    "qwen": {
        "base-url": "DASHSCOPE_BASE_URL",
        "model": "DASHSCOPE_MODEL",
        "api-key": "DASHSCOPE_API_KEY",
    },
}


def _resolve_config(model_name):
    cfg = dict(configs[model_name])
    env_map = _ENV_OVERRIDES.get(model_name, {})
    for key, env_name in env_map.items():
        value = str(cfg.get(key, "")).strip()
        if value in _PLACEHOLDERS:
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                cfg[key] = env_value
    api_key = str(cfg.get("api-key", "")).strip()
    if api_key in _PLACEHOLDERS:
        raise ValueError(
            f"Missing API key for '{model_name}'. "
            f"Set config.yaml api-key or export {env_map.get('api-key', 'API key env var')}."
        )
    return cfg


def invoke_api(model, messages):
    cfg = _resolve_config(model)
    base_url, model, api_key = cfg["base-url"], cfg["model"], cfg["api-key"]
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )
    with TimeCounter(tag="Invoke"):
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0.1,
            top_p=0.3,
        )
        result = {
            'answer': response.choices[0].message.content, 
            'prompt_tokens': response.usage.prompt_tokens, 
            'completion_tokens': response.usage.completion_tokens
        }
        return result
    
    
if __name__ == '__main__':
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Who are you?'},
            ],
        }
    ]
    print(invoke_api('gpt-proxy', messages))