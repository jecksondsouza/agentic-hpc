from dataclasses import dataclass, field

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

