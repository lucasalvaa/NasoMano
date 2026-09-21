import language_tool_python
import threading

_lt_instance = None
_lock = threading.Lock()

def get_languagetool_instance():
    global _lt_instance

    with _lock:
        if _lt_instance is None:
            _lt_instance = language_tool_python.LanguageTool(
            'en-US',
            remote_server='http://127.0.0.1:8081/'
        )
    return _lt_instance