from schemas import AgentEvent
import time
import json
from pathlib import Path
import dataclasses

class Observer:
    def __init__(self):
        self.output_file = Path(f"traces/trace_{time.time()}")
        self.output_file.parent.mkdir(exist_ok=True, parents=True)

    def write_event(self, event: AgentEvent):
        with open(self.output_file, 'a') as file:
            file.write(json.dumps(dataclasses.asdict(event), indent=2))