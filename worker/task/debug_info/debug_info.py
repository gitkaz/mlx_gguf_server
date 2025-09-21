import json
from pydantic import BaseModel

from ...task_response import TaskResponse

class DebugInfo:
    def __init__(self):
        self.results:dict = {}

    def set(self, **kwargs):
        task_category = kwargs.get("task_category")
        if task_category != "debug_latest":
            for k,v in kwargs.items():
                if k in ["task_id", "task_category", "params", "kv_cache", "stream_generate"]:
                    self.results[k] = v
            self._check_params_type()

    def _check_params_type(self):
        params = self.results.get("params")
        if params:
            try:
                json.dumps(params)
            except:
                if isinstance(params, BaseModel):
                    self.results["params"] = params.dict()
                else:
                    self.results["params"] = f"{type(params)} is not json serializable"        

    def get(self) -> TaskResponse:
        return (TaskResponse(200, self.results))