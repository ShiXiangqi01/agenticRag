import inspect
import logging
import contextlib
from dataclasses import dataclass
from typing import Callable, Any, Literal, get_args, get_origin, get_type_hints
from openai import NOT_GIVEN
from typing_extensions import TypeGuard
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel, Field, create_model
from griffe import Docstring, DocstringSectionKind


'''
name,
description(the description of what the function does, used by the model to choose when andhow to call the function), 
parameters, 
strict
'''
_EMPTY_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}

def __is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


def __is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def __has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False

def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert __is_dict(value), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved



def _ensure_strict_json_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the OpenAI API expects.
    """
    if schema == {}:
        return _EMPTY_SCHEMA
    return __ensure_strict_json_schema(schema, path=(), root=schema)


# Adapted from https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
def __ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not __is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")
    # Convert "integer" types to "number" for strict JSON schema consistency
    if "type" in json_schema:
        t = json_schema["type"]
        if t == "integer":
            json_schema["type"] = "number"
        elif isinstance(t, list):
            json_schema["type"] = ["number" if x == "integer" else x for x in t]

    defs = json_schema.get("$defs")
    if __is_dict(defs):
        for def_name, def_schema in defs.items():
            __ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if __is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            __ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif typ == "object" and "additionalProperties" in json_schema and json_schema["additionalProperties"] is True:
        raise ValueError(
            "additionalProperties should not be set for object types. This could be because "
            "you're using an older version of Pydantic, or because you configured additional "
            "properties to be allowed. If you really need this, update the function or output tool "
            "to not use a strict schema."
        )

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if __is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: __ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if __is_dict(items):
        json_schema["items"] = __ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = json_schema.get("anyOf")
    if __is_list(any_of):
        json_schema["anyOf"] = [
            __ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if __is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(__ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                __ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", NOT_GIVEN) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and __has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not __is_dict(resolved):
            raise ValueError(f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}")

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid
        return __ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def __purge_titles(data):
    # helper function to remove parameter "title"s in the JSON schema
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key == "title" and "type" in data.keys():
                continue
            else:
                new_data[key] = __purge_titles(value)
        return new_data
    elif isinstance(data, list):
        return [__purge_titles(item) for item in data]
    else:
        return data


def __purge_defaults(data):
    # helper function to remove parameter "default"s in the JSON schema
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key == "default" and "type" in data.keys():
                continue
            else:
                new_data[key] = __purge_defaults(value)
        return new_data
    elif isinstance(data, list):
        return [__purge_defaults(item) for item in data]
    else:
        return data

__DocstringStyle__ = Literal["google", "numpy", "sphinx"]

@dataclass
class __FuncSchema__:
    '''
    Captures the schema for a python function, in preparation for sending it to an LLM as a tool.
    '''
    name: str
    """The name of the function."""
    description: str | None
    """The description of the function."""
    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""
    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""
    signature: inspect.Signature
    """The signature of the function."""
    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""


    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
        the original function.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False

        # Use enumerate() so we can skip the first parameter if it's context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            # If the function takes a RunContextWrapper and this is the first parameter, skip it.
            if self.takes_context and idx == 0:
                continue

            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
        return positional_args, keyword_args

@dataclass
class __FuncDocumentation__:
    name: str
    """The name of the function."""
    description: str | None
    """The description of the function, typically derived from the docstring."""
    param_descriptions: dict[str, str] | None
    """A mapping of parameter names to their descriptions, typically derived from the docstring."""


@contextlib.contextmanager
def _suppress_griffe_logging():
    # Supresses warnings about missing annotations for params
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)

'''
用以下函数举例
def send_email(to: str, subject: str, body: str, is_html: bool = False):
    """
    发送一封电子邮件。
    
    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
        is_html: 正文是否为 HTML 格式
    """
    pass
'''

def _generate_func_documentation(
    func: Callable[..., Any],
    style: __DocstringStyle__ | None = None,
) ->  __FuncDocumentation__:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, 
        # TODO: we will attempt to auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        return __FuncDocumentation__(name=name, description=None, param_descriptions=None)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or "google")
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text),
        None,
    )

    param_descriptions: dict[str, str] = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return __FuncDocumentation__(
        name=func.__name__,
        description=description,
        param_descriptions=param_descriptions or None,
    )



def _function_schema(
        func: Callable[..., Any],
        docstring_style: __DocstringStyle__ | None = None, 
        name_override: str | None = None,
        description_override: str | None = None,
        use_docstring_info: bool = True,
        strict_json_schema: bool = True,
) -> __FuncSchema__:
    """
    Given a python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided, we will default as google.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one derived from the
            docstring.
        use_docstring_info: If True, uses the docstring to generate the description and parameter
            descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll ensure that
            the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**
            recommend setting this to True, as it increases the likelihood of the LLM providing
            correct JSON input.

    Returns:
        A `FuncSchema` object containing the function's name, description, parameter descriptions,
        and other metadata.
    """
    if use_docstring_info:
        doc_info = _generate_func_documentation(func, docstring_style)
        param_descs = doc_info.param_descriptions or {}
    else:
        doc_info = None
        param_descs = {}
    
    func_name = name_override or doc_info.name if doc_info else func.__name__

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = list(sig.parameters.values())
    takes_context = False
    filtered_params = []

    if params:
        first_param = params[0]
        first_name = first_param.name
        # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect._empty:
            # origin = get_origin(ann) or ann
            # if origin is RunContextWrapper:
            #     takes_context = True  # Mark that the function takes context
            # else:
            filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))

    # For parameters other than the first, raise error if any use RunContextWrapper.
    for param in params[1:]:
        name = param.name
        ann = type_hints.get(name, param.annotation)
        # if ann != inspect._empty:
        #     origin = get_origin(ann) or ann
        #     if origin is RunContextWrapper:
        #         raise ValueError(
        #             f"RunContextWrapper param found at non-first position in function"
        #             f" {func.__name__}"
        #        )
        filtered_params.append((name, param))

    # We will collect field definitions for create_model as a dict:
    #   field_name -> (type_annotation, default_value_or_Field(...))
    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default

        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any

        # If a docstring param description exists, use it
        field_description = param_descs.get(name, None)

        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = list[ann]  # type: ignore

            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),  # type: ignore
            )

        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = dict[str, ann]  # type: ignore

            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),  # type: ignore
            )

        else:
            # Normal parameter
            if default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )

    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)

    # 4. Build JSON schema from that model
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = _ensure_strict_json_schema(json_schema)

    # 5. Return as a FuncSchema dataclass
    return __FuncSchema__(
        name=func_name,
        description=description_override or doc_info.description if doc_info else None,
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
    )

def generate_openai_function_schema(func: Callable, use_gemini_format: bool = False) ->  FunctionDefinition:
    func_schema = _function_schema(func)
    func_name = func_schema.name
    func_desc = func_schema.description

    # if use_gemini_format:
    #     params_schema = Tool(function_declarations=[FunctionDeclaration.from_func(func)]).to_dict()[ #TODO: geminig格式，用到再说
    #         "function_declarations"
    #     ][0]["parameters"]
    #     params_schema = __replace_type__with_type(params_schema)    #TODO: gemini格式，用到再说
    #     params_schema = __purge_titles(params_schema)
    #     params_schema = __purge_defaults(params_schema)
    # else:
    params_schema = dict(func_schema.params_json_schema)
    params_schema = __purge_titles(params_schema)
    params_schema = __purge_defaults(params_schema)

    func_def = FunctionDefinition(
        name=func_name,
        description=func_desc or "No description provided.",
        parameters=params_schema,  # type: ignore
        strict=True,
    )
    return func_def

# if __name__ == "__main__":
#     def send_email(to: str, subject: str, body: str, is_html: bool = False):
#         """
#         发送一封电子邮件。
        
#         Args:
#             to: 收件人邮箱地址
#             subject: 邮件主题
#             body: 邮件正文内容
#             is_html: 正文是否为 HTML 格式
#         """
#         pass

#     schema = generate_openai_function_schema(send_email)
#     print(json.dumps(schema.model_dump(), indent=2))