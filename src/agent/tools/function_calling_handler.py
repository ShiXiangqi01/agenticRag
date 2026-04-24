import srsly
from fastapi.encoders import jsonable_encoder
from loguru import logger

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from src.agent.tools.tools import Tool
from src.agent.tools.function_schema import generate_openai_function_schema

class FunctionCallingHandler:
    def __init__(
        self,
        available_tools: list[Tool] | None = None,
        use_gemini_format: bool = False,
    ):
        self.use_gemini_format = use_gemini_format
        self._available_tools = available_tools or []
        self._name_to_function = {}
        self.__register_tool_functions() 

    def __register_tool_functions(self):
        for tool in self._available_tools:
            for name, func in tool.functions.items():
                if not callable(func):
                    raise ValueError(f"Function {name} in tool {tool.name} is not callable.")
                self._name_to_function[name] = func
                logger.debug(f"Registered function '{name}'")

    def _get_registered_function(self) -> list[str]:
        return list(self._name_to_function.keys())
    
    def execute_function(
            self,
            *,
            name: str,
            convert_result_to_json: bool = True,
            **kwargs,
    ):
        logger.debug(f"Attempting to execute function '{name}' with arguments: {kwargs}")
        if name not in self._name_to_function:
            raise ValueError(f"Function '{name}' is not registered.")
        try:
            res = self._name_to_function[name](**kwargs)
        except Exception as e:
            res = str(e)
        
        if convert_result_to_json:
            res = srsly.json_dumps(jsonable_encoder(res))
        logger.debug(f"Function '{name}' executed successfully with result: {res}")
        return res

    def build_tool_params_for_model(self) -> list[ChatCompletionToolParam]:
        tool_params = []
        for _, func in self._name_to_function.items():
            function = generate_openai_function_schema(func, use_gemini_format=False)
            tool_params.append(ChatCompletionToolParam(function= function, type="function"))
        return tool_params
    


    

    
