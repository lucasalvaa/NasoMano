from threading import Lock, Semaphore
import language_tool_python

"""
An implementation of the Thread-Safe Singleton design pattern around a single shared
instance of LanguageTool. It closely follows the design pattern implementation described 
in: https://refactoring.guru/design-patterns/singleton/python/example#example-1

A semaphore has also been added to protect the LanguageTool Docker container
from an excessive number of simultaneous calls when the NasoMano API receives
hundreds of requests at the same time.
"""

class SingletonMeta(type):
    _instances: dict[type, object] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LanguageToolSingleton(metaclass=SingletonMeta):
    def __init__(self, language: str = "en-US", concurrency_limit: int = 20) -> None:
        self.tool = language_tool_python.LanguageTool(
            language,
            remote_server='http://127.0.0.1:8081/'
        )
        self.semaphore = Semaphore(concurrency_limit)

    def check(self, prompt: str):
        with self.semaphore:
            return self.tool.check(prompt)


def get_languagetool_instance(
    language: str = "en-US",
    concurrency_limit: int = 20,
) -> LanguageToolSingleton:
    return LanguageToolSingleton(language, concurrency_limit)