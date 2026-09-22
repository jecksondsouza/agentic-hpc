from dataclasses import dataclass, field
from pydantic import BaseModel

@dataclass
class AgentState():
    messages: list = field(default_factory=list)
    iteration: int = 0
    tool_calls: int = 0
    errors: int = 0

@dataclass
class AgentEvent():
    timestamp: int
    iteration: int
    event: str
    message: str
    tool_calls: int
    errors: int
    tool: str | None = None
    arguments: str | None = None
    current_best_run: RawExperimentResult | None = None

@dataclass
class Tool:
    name: str
    function: Callable
    args_schema: Type[BaseModel] | None = None

class RunExperimentArgs(BaseModel):
    cpus: list[int] = field(default_factory=list)
    num_threads: int | None = 0

@dataclass
class RawExperimentResult():
    output: str
    error: str
    errorcode: str
    
    args: RunExperimentArgs
    execution_time_s: float # In seconds with ms precision

class Config(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "my_llm"
    max_iterations: int = 10
    max_tool_calls: int = 20
# when adding new fields to the config, make sure to also add it in CONFIG_FIELDS, in main.py so that the configuration command can work correctly    
