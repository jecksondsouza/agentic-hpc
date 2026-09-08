import json
from pydantic import BaseModel
import dataclasses
from openai import OpenAI
import time
import sys

# Internal modules
from tools import tools_list, tools_registry, run_experiment
from schemas import AgentState, AgentEvent, RawExperimentResult
from observer import Observer
from hardware_profile import HardwareProfile

MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 20

class Agent:

    def __init__(self, client: OpenAI, benchmark_path: str, docker_image: str):
        self.state = AgentState()
        self.client = client
        self.observer = Observer()
        self.hardware_profile = HardwareProfile()

        self.benchmark_path = benchmark_path
        self.docker_image = docker_image
    
    def run_agent(self, user_prompt):

        self.observer.write_hardware_profile(self.hardware_profile)

        baseline_result = run_experiment(self.benchmark_path, self.docker_image)

        best_run = baseline_result

        self.state.messages = [
            {
                # System prompt
                "role": "developer",
                "content": (
                    "You are an expert HPC researcher and systems engineer. "
                    "Your goal is to find the combination of parameter that give the best performance for the running benchmark."
                    "To do that, you will use reasoning based on the platform information you are currently running in."
                    "Be concise and technically precise."
                    "Use tools when necessary."
                    "Never repeat an experiment with same arguments that you have already run before, even if the arguments caused an error in the tool execution." #This is not working. Will have to reject it deterministically
                    "Keep running new experiments until you are told that you must stop and conclude your thoughts."                                        
                ),
            },

        ]

        while True:

            if self._reached_limits():
                self.state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tries for this run and you must stop now. Conclude your thoughts now and do not call any more tools"
                            "Your thoughts should be output in a markdown format and they should explain the reasoning behind the chosen best experiment."
                            f"the current best run is: \n {json.dumps(dataclasses.asdict(best_run))}"
                        )
                    }
                )
            else:
                self.state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You are not done yet, keep trying new experiments."
                            f"the current best run is: \n {json.dumps(dataclasses.asdict(best_run))}"
                        )
                    }
                )
            self.state.iteration += 1

            response = self.client.chat.completions.create(
                model="empero-ai/Qwen3.5-9B-Claude-Code-GGUF:Q8_0",
                messages=self.state.messages,
                tools=tools_list,
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

                    if result.execution_time_s < best_run.execution_time_s:
                        best_run = result

                except json.JSONDecodeError as e:

                    result = RawExperimentResult(
                        output= "",
                        error= str(e),
                        errorcode= "invalid_tool_arguments",
                        args= arguments,
                        execution_time_s= sys.float_info.max
                    )
                    self.state.errors += 1

                except ValueError as e:

                    result = RawExperimentResult(
                        output= "",
                        error= str(e),
                        errorcode= "invalid_tool_request",
                        args= arguments,
                        execution_time_s= sys.float_info.max
                    )
                    self.state.errors += 1

                except Exception as e:

                    result = RawExperimentResult(
                        output= "",
                        error= str(e),
                        errorcode= "tool_execution_failed",
                        args= arguments,
                        execution_time_s= sys.float_info.max
                    )
                    self.state.errors += 1

                # Add the output of the tool call to the context
                self.state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(dataclasses.asdict(result)),
                    }
                )
                self.state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"the current best run is: \n {json.dumps(dataclasses.asdict(best_run))}"
                    }
                )
                self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="tool call",
                    message=json.dumps(dataclasses.asdict(result)),
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                    tool=tool_name,
                    arguments=tool_call.function.arguments,
                    current_best_run=dataclasses.asdict(best_run),
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

        arguments = validated.model_dump()

        if name == "run_experiment":
            arguments["benchmark_path"] = self.benchmark_path
            arguments["docker_image"] = self.docker_image

        return tool.function(
            **arguments
        )

    def _reached_limits(self) -> bool:
        if self.state.tool_calls >= MAX_TOOL_CALLS:
            self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="Tool call limit",
                    message="Agent exceeded tool call limit",
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
            return True

        if self.state.iteration >= MAX_ITERATIONS:
            self.state.errors += 1
            self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="Iteration limit",
                    message="Agent exceeded iteration limit",
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
            return True
        return False

