import platform
import subprocess

from dataclasses import dataclass
from typing import Callable, Type
from pydantic import BaseModel


@dataclass
class Tool:
    name: str
    function: Callable
    args_schema: Type[BaseModel] | None = None

class RunCommandArgs(BaseModel):
    command: str

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Return information about the current compute system.",
            "parameters": {
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an approved diagnostic system command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "An approved diagnostic command."
                    }
                },
                "required": ["command"],
            },
        },
    }
]

def get_system_info():
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

def run_command(command: str):

    allowed = {
        "uname -a",
        "nproc",
        "free -h",
        "lscpu",
    }

    if command not in allowed:
        raise ValueError(f"Command not allowed. Allowed commands are {', '.join(allowed)}")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode,
    }

tools_registry = {
    "get_system_info": Tool(
        name="get_system_info",
        function=get_system_info,
    ),

    "run_command": Tool(
        name="run_command",
        function=run_command,
        args_schema=RunCommandArgs,
    ),
}