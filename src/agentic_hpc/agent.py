import json
from pydantic import BaseModel
from dataclasses import dataclass, field
from openai import OpenAI
import time


# Internal modules
from tools import tools, tools_registry
from schemas import AgentState, AgentEvent
from observer import Observer

# Agent limitations - As of now, only one tool call per iteration, which means that MAX_TOOL_CALLS would not be reached
MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 20

class Agent:

    def __init__(self, client: OpenAI):
        self.state = AgentState()
        self.client = client
        self.observer = Observer()
    
    def run_agent(self, user_prompt):

        self.state.messages = [
            {
                # System prompt
                "role": "system",
                "content": (
                    "You are a helpful systems engineering assistant. "
                    "Be concise and technically precise. "
                    "Use tools when necessary. "
                ),
            },
            {
                # User request
                "role": "user",
                "content": user_prompt,
            }
        ]

        while True:

            self._check_limits()
            self.state.iteration += 1

            response = self.client.chat.completions.create(
                model="empero-ai/Qwen3.5-9B-Claude-Code-GGUF:Q8_0",
                messages=self.state.messages,
                tools=tools,
            )

            message = response.choices[0].message

            self.observer.write_event(AgentEvent(
                timestamp = time.time_ns(),
                iteration=self.state.iteration,
                event="LLM response",
                message=message.content,
                tool_calls=self.state.tool_calls,
                errors=self.state.errors,
            ))

            # No tool call need anymore -> final answer reached
            if not message.tool_calls:
                self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="Final answer",
                    message=message.content,
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
                return message.content

            # Add the tool request message to the context
            self.state.messages.append(message)

            # Execute every requested tool
            for tool_call in message.tool_calls:
                
                tool_name = tool_call.function.name
                print (f"Calling tool {tool_name} with arguments {tool_call.function.arguments}")

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    result = self._execute_tool(
                        tool_call.function.name,
                        arguments,
                    )
                    # Deliberately considers a tool call only if the call goes through
                    self.state.tool_calls += 1

                except json.JSONDecodeError as e:

                    result = {
                        "error": "invalid_tool_arguments",
                        "message": str(e),
                    }
                    self.state.errors += 1

                except ValueError as e:

                    result = {
                        "error": "invalid_tool_request",
                        "message": str(e),
                    }
                    self.state.errors += 1

                except Exception as e:

                    result = {
                        "error": "tool_execution_failed",
                        "message": str(e),
                    }
                    self.state.errors += 1

                # Add the output of the tool call to the context
                self.state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
                self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="tool call",
                    message=json.dumps(result),
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                    tool=tool_name,
                    arguments=tool_call.function.arguments
                ))

                

    def _execute_tool(self, name: str, arguments: dict):

        tool = tools_registry[name]

        if tool.args_schema is None:
            if arguments:
                raise ValueError(
                    f"Tool '{name}' takes no arguments"
                )

            return tool.function()

        validated = tool.args_schema.model_validate(arguments)

        return tool.function(
            **validated.model_dump()
        )

    def _check_limits(self):
        if self.state.tool_calls >= MAX_TOOL_CALLS:
            self.state.errors += 1
            self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="RuntimeError",
                    message="Agent exceeded tool call limit",
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
            raise RuntimeError("Agent exceeded tool call limit")

        if self.state.iteration >= MAX_ITERATIONS:
            self.state.errors += 1
            self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="RuntimeError",
                    message="Agent exceeded iteration limit",
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
            raise RuntimeError("Agent exceeded iteration limit")

