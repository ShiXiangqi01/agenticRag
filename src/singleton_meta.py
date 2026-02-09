from abc import ABCMeta
from typing import Type, TypeVar
from loguru import logger

'''
了一个单例元类，确保使用该元类的类在程序运行期间只有一个实例。
当你调用类（例如 MyClass()）时，不会直接创建新实例，而是先检查 __singleton：
已存在：直接返回已有实例
不存在：创建新实例（调用 __new__ 和 __init__），保存后返回
这样无论调用多少次 MyClass()，都返回同一个对象。

'''


SingletonInstance = TypeVar("SingletonInstance")


class SingletonMeta(ABCMeta):
    def __init__(cls, class_name, bases, attrs):
        cls.__singleton = None
        logger.info(f"Instantiating {class_name} Singleton...")
        super().__init__(class_name, bases, attrs)

    def __call__(cls: Type[SingletonInstance], *args, **kwargs) -> SingletonInstance:
        # Not sure if there's a way to typehint the __singleton
        # attribute somehow.
        if cls.__singleton:  # type: ignore
            return cls.__singleton  # type: ignore
        singleton = cls.__new__(cls, *args, **kwargs)
        singleton.__init__(*args, **kwargs)
        cls.__singleton = singleton  # type: ignore
        return singleton
