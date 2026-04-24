import json
from typing import Any, Iterable

import mlflow
from mlflow.entities import SpanType
from ollama import chat
import os
import re
from openai import OpenAI, OpenAIError
import requests
import pandas as pd
from functools import cache
from loguru import logger

from openai.types.chat import ChatCompletionMessageParam
'''
它告诉代码编辑器："这里可以传入以下几种消息类型"，相当于一个"消息类型集合"。
'''

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from src.config import load_config
from src.data.dtos.agent import AgentModel, ChatMessage
from src.agent.tools.tools import Tool
from src.agent.tools.function_calling_handler import FunctionCallingHandler

OLLAMA_GENERATION_CONFIG = {
    "n": 1,  # number of completions to generate
    "temperature": 0.3,
    "max_completion_tokens": 2048,
}

MAX_CHAT_HISTORY_MESSAGES = 24
MAX_TOOL_CALL_ROUNDS = 4

class ChatAssistant:
    def __init__(
            self,
            *,
            assistant_name: str | None = None,
            model_name: str | None = None,
            system_instruction: str | None = None,
            available_tools: list[Tool] | None = None,
            generation_config: dict[str, Any] | None = None,
        request_timeout: float | None = None,
    ):
        """
        The ChatAssistant class allows sending user messages to the model and receiving responses from
        the model. The assistant can be configured with a specific model, system instruction, and available tools.

        Args:
            assistant_name (str, optional): The name of the assistant just for logging purposes. Defaults to None.
            model_name (str, optional): The name of the Ollama model to use. If not provided, the default model specified in the config file will be used.
            system_instruction (str, optional): The system instruction to provide to the model. If not provided, no system instruction will be used.
            available_tools (list[Tool], optional): The list of available tools that the assistant can use. If not provided, no tools will be available.
            generation_config (dict, optional): The generation configuration to use when sending messages to the model. Defaults to the OLLAMA_GENERATION_CONFIG.
        """
        self._conf = load_config()
        self.model_name = model_name or self._conf.assistant.default_model
        self.assistant_name = assistant_name or "NO NAME"
        if not self.is_model_available(self.model_name):
            raise ValueError(f"Model '{self.model_name}' is not available.")
        self._system_instruction = system_instruction
        self._available_tools = available_tools
        if available_tools is None:
            self._available_tools = []
        self._request_timeout = request_timeout
        self._function_call_handler = FunctionCallingHandler(
            available_tools=self._available_tools,
            use_gemini_format= False, #TODO: 因为gemini格式的工具调用和openai原生格式不太一样，所以暂时先区分开来，如果用得到gemini的话 再说
        )
        self._generation_config = dict(generation_config or OLLAMA_GENERATION_CONFIG)
        self._chat_history: list[ChatCompletionMessageParam] = [] 

    def _trim_chat_history(self) -> None:
        if not self._chat_history:
            return

        system_messages: list[ChatCompletionMessageParam] = []
        rolling_messages = self._chat_history

        first_msg = self._chat_history[0]
        if first_msg.get("role") == "system":
            system_messages = [first_msg]
            rolling_messages = self._chat_history[1:]

        if len(rolling_messages) > MAX_CHAT_HISTORY_MESSAGES:
            rolling_messages = rolling_messages[-MAX_CHAT_HISTORY_MESSAGES:]

        self._chat_history = system_messages + rolling_messages

    @staticmethod
    @cache
    def list_available_models() -> pd.DataFrame:
        base_url = os.getenv("OLLAMA_BASE_URL").rstrip("/")

        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            ollama_models = resp.json().get("models", [])
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            return pd.DataFrame(columns=["name", "display_name"])

        def get_display_name(model_id: str) -> str:
            # 例如: llama3.1:8b -> Llama3.1 8b
            return re.sub(r"[:\-]+", " ", model_id).strip().title()

        models = []
        for m in ollama_models:
            model_name = m.get("name")
            if not model_name:
                continue
            models.append(
                {
                    "name": f"ollama/{model_name}",   # 统一加前缀便于路由
                    "display_name": get_display_name(model_name),
                }
            )

        return pd.DataFrame(models)
    
    @staticmethod
    def get_default_model() -> AgentModel:
        conf = load_config()
        default_model_name = conf.assistant.default_model
        available_models = ChatAssistant.list_available_models()
        model = available_models[available_models["name"] == default_model_name]
        if model.empty:
            model = available_models.iloc[0]

        return AgentModel(
            name = model["name"].values[0],
            display_name = model["display_name"].values[0],
        )


    @staticmethod
    def is_model_available(model_name:str) -> bool:
        available_models = ChatAssistant.list_available_models()
        return model_name in available_models["name"].values

    # _get_ollama_client
    def _get_ollama_client(self) -> OpenAI:
        """Create an OpenAI-compatible client that points to the local Ollama server."""
        base_url = (os.getenv("OLLAMA_BASE_URL")).rstrip("/")

        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        if isinstance(self.model_name, str) and self.model_name.startswith("ollama/"):
            self.model_name = self.model_name.split("/", 1)[1]

        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        return OpenAI(base_url=base_url, api_key=api_key)

    
    def _build_system_instruction(                              #当history为空时，才添加system instruction
        self, system_instruction: str | None = None
    ) -> list[ChatCompletionSystemMessageParam]: 
        if system_instruction:
            return [{"role": "system", "content": system_instruction}]
        return []
    
    def _build_user_message(
        self, prompt: str, base64_image: str | None = None
    ) -> list[ChatCompletionUserMessageParam]: 
        content: list[ChatCompletionContentPartParam] = [
            {"type": "text", "text": prompt}
        ]
        if base64_image:
            if not base64_image.startswith("data:image/"):
                base64_image = f"data:image/png;base64,{base64_image}"

            img_content = ChatCompletionContentPartImageParam(
                image_url={
                    "url": base64_image,
                    "detail":"auto",
                },
                type="image_url",
            )
            content.append(img_content)
        messages = [ChatCompletionUserMessageParam(role="user", content=content)]

        return messages
    
    def _add_assistant_response_to_chat_history(self, response: ChatCompletion) -> None:
        message = response.choices[0].message #选择第一条信息
        if message.role == "assistant":
            self._chat_history.append(ChatCompletionAssistantMessageParam(**message.model_dump()))

    def _create_chat_completion_from_history(self) -> ChatCompletion:
        tools = self._function_call_handler.build_tool_params_for_model() 
        messages = self._chat_history
        client = self._get_ollama_client()
        try:
            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                **self._generation_config,
            }
            if self._request_timeout is not None:
                request_kwargs["timeout"] = self._request_timeout
            if len(tools) > 0:
                logger.debug(
                    f"[{self.assistant_name}] Sending message to model {self.model_name} with tools: {[tool.name for tool in self._available_tools]}"
                )
                response = client.chat.completions.create(
                    tools=tools,
                    tool_choice="auto",
                    **request_kwargs,
                )
            else:
                logger.debug(
                    f"[{self.assistant_name}] Sending message to model {self.model_name} without tools"
                )
                response = client.chat.completions.create(
                    **request_kwargs,
                )
            logger.debug(f"[{self.assistant_name}] Received response from model {self.model_name}: {response}")
            return response
        except OpenAIError as e:
            logger.error(f"[{self.assistant_name}] An OpenAIError occured: {e}") 
            raise e
        except Exception as e:
            logger.error(f"[{self.assistant_name}] Unexpected Error of Type {type(e)}: {e}")
            raise e

    def _get_message_content(self, response_or_message: ChatCompletion | ChatCompletionMessageParam) -> str:
        text = ""
        if isinstance(response_or_message, ChatCompletion):
            message = response_or_message.choices[0].message
            if message.content:
                text = message.content
        else:
            content = response_or_message.get("content")
            texts = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, Iterable):
                for part in content:
                    if isinstance(part, str):
                        texts.append(part)
                    else:
                        t = part.get("text", part.get("refusal", None))
                        if t:
                            texts.append(t)

            if texts:
                text = "\n".join(texts)

        return text
    

    def _is_tool_call_response(self, response: ChatCompletion) -> bool:
        '''
        从response中找到choices[0].message.tool_calls，如果存在且长度大于0，则说明需要调用工具
        choices 是模型生成的候选回复列表，包含了AI对你输入的所有可能回答 
        因为生成式AI每次对同一个问题可能给出不同的回答，这个列表让你可以从多个候选回答中选择。
        默认情况下 n == 1
        '''
        try:
            tool_calls = response.choices[0].message.tool_calls
            return tool_calls is not None and len(tool_calls) > 0
        except AttributeError:
            return False
        

    def _execute_tool_calls(self, response: ChatCompletion) -> list[ChatCompletionToolMessageParam]:
        tool_calls = tool_calls = response.choices[0].message.tool_calls
        if tool_calls is None:
            return []

        tool_messages = []
        for tc in tool_calls:
            try:
                tool_name = tc.function.name
                tool_args_str = tc.function.arguments or "{}"
                tool_args = json.loads(tool_args_str)

                result_json_str = self._function_call_handler.execute_function(
                    name=tool_name,
                    **tool_args,
                )

                tool_message = ChatCompletionToolMessageParam(
                    role="tool",
                    content=result_json_str,
                    tool_call_id = tc.id,
                )
                tool_messages.append(tool_message)
            except Exception as e:
                logger.error(f"[{self.assistant_name}] Error executing tool call: {e}")
                raise e

        return tool_messages
                




    def send_user_message(
            self,
            text_message: str,
            base64_image: str | None = None,
    ) -> str:
        logger.info(f"[{self.assistant_name}] Sending user message to model {self.model_name}: {text_message}")
        if len(self._chat_history) ==0:
            self._chat_history.extend(self._build_system_instruction(self._system_instruction))
        user_message = self._build_user_message(text_message, base64_image)
        self._chat_history.extend(user_message)
        self._trim_chat_history()
        response = self._run_agentic_loop()
        return response
    
    def get_conversation_history(self)-> list[ChatMessage]:
        pass
    

    @mlflow.trace(span_type = SpanType.AGENT)
    def _run_agentic_loop(self) -> str:
        #Send the messages in the chat history to the model
        response = self._create_chat_completion_from_history()  #return response from ollama client
        self._add_assistant_response_to_chat_history(response) 

        #  Run the agentic loop until no tool calls are present in the response
        tool_call_round = 0
        while self._is_tool_call_response(response):
            tool_call_round += 1
            if tool_call_round > MAX_TOOL_CALL_ROUNDS:
                logger.warning(
                    f"[{self.assistant_name}] Tool-call rounds exceeded {MAX_TOOL_CALL_ROUNDS}; stopping loop early"
                )
                break
            logger.debug(f"[{self.assistant_name}] Tool Calls detected in response!")
            tool_messages = self._execute_tool_calls(response)
            self._chat_history.extend(tool_messages)
            self._trim_chat_history()

            response = self._create_chat_completion_from_history()  #return response from ollama client
            self._add_assistant_response_to_chat_history(response)
            self._trim_chat_history()

        message = self._get_message_content(response)
        return message







# if __name__ == "__main__":
#     # 测试列出可用模型
#     models_df = ChatAssistant.list_available_models()
#     print("Available Ollama Models:")
#     print(models_df)
#     '''
#                          name        display_name
# 0          ollama/qwen3.5:27b         Qwen3.5 27B
# 1          ollama/qwen3-vl:8b         Qwen3 Vl 8B
#     '''