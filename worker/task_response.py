import json
from typing import Union

class TaskResponse:
    def __init__(self, status: int, message: Union[int, float, str, dict] = None):
        self.status = status
        self.message = message
        self.validate()

    def validate(self):
        if not isinstance(self.status, int):
            self.message = f"Internal Error: Response validation failed. {self.status=} is not Int."
            self.status = 500
        if not (200 <= self.status < 600):
            self.message = f"Internal Error: Response validation failed. {self.status=} is not in range."
            self.status = 500
        if self.message is not None and not isinstance(self.message, (int, float, str, dict)):
            self.message = f"Internal Error: Response validation failed. {type(self.message)} is not valid type."
            self.status = 500

    def to_json(self):
        return json.dumps({"status": self.status, "message": self.message})

    def to_dict(self):
        return {"status": self.status, "message": self.message}

    @classmethod
    def create(cls, status, message):
        return cls(status, message)
