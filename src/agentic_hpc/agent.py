import json
from pydantic import BaseModel
import dataclasses
from openai import OpenAI
import time
import sys

# Internal modules
from tools import tools_list, tools_registry
from schemas import AgentState, AgentEvent, RawExperimentResult
from observer import Observer
from hardware_profile import HardwareProfile
from validation import RepeatedExperimentError, ToolCallValidator

MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 20

class Agent:

    def __init__(self, client: OpenAI, benchmark_path: str, docker_image: str,
                 model: str = "my_llm",
                 max_iterations: int = MAX_ITERATIONS,
                 max_tool_calls: int = MAX_TOOL_CALLS):
        self.state = AgentState()
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.observer = Observer()
        self.hardware_profile = HardwareProfile()
        self.validator = ToolCallValidator(self.hardware_profile)

        self.benchmark_path = benchmark_path
        self.docker_image = docker_image
    
    def run_agent(self, user_prompt):

        self.observer.write_hardware_profile(self.hardware_profile)

        baseline_result = self._execute_tool("run_experiment", {"cpus": [0], "num_threads": 1})

        best_run = baseline_result

        system_prompt = ("You are an expert HPC researcher and systems engineer. "
                    "Your goal is to find the combination of parameter that give the best performance for the running benchmark."
                    "To do that, you will use reasoning based on the platform information you are currently running in. You will not try to run experiments on incompatible hardware."
                    f"The hardware profile of the platform you are running in is given in the json bellow: \n {json.dumps(self.hardware_profile)}"
                    "Be concise and technically precise."
                    "Always provide your reasoning at each response."
                    "Use tools when necessary."
                    "Never repeat an experiment with same arguments that you have already run before, even if the arguments caused an error in the tool execution. Repeated experiments are rejected automatically and do not count against your tool call budget."
                    "Keep running new experiments until you are told that you must stop and conclude your thoughts."                               
                    )
        
        self.observer.write_event(AgentEvent(
            timestamp = time.time_ns(),
            iteration=self.state.iteration,
            event="System Prompt",
            message=system_prompt,
            tool_calls=self.state.tool_calls,
            errors=self.state.errors,
        ))

        while True:

            if self._reached_limits():
                self.state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tries for this run and you must stop now. Conclude your thoughts now and do not call any more tools"
                            "Your thoughts should be output in a markdown format and they should explain the reasoning behind the chosen best experiment."
                            "Add all previous reasoning steps you have used for each experiment you run, showing step by step how you reached the final conclusion."
                            f"the current best run is: \n {json.dumps(dataclasses.asdict(best_run))}"
                        )
                    }
                )
            else:
                self.state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You can still try to run more experiments. You still have {self.max_iterations-self.state.iteration} iterations and {self.max_tool_calls-self.state.tool_calls} tool calls."
                            ##### The line bellow is useful if you want the agent to decide to stop early. If a smart LLM is used, this can be useful to reduce the amount of tries and it could possibly lead to quickly finding the best solution for well known applications.
                            # "However, if you are certain you have reached the final solution, you can stop now by not calling any tools."
                            ##### The line bellow is useful if you want the agent to try something closer to a heuristic search. It might be necessary if running applications that the agent could not know about (e.g., not well-known benchmarks)
                            # "Do not stop until you have tried to run a few experiments around the optimal solution (for instance, a few threads less or more, or pinning to different cores using the best thread count so far)"
                            ##### The line bellow forces the agent to use all tries, but doesn't guide it towards the best solution, rather leaves it to decide on the strategy.
                            "You must use all your tries in a best effort to reason the best solution."
                            f"the current best run is: \n {json.dumps(dataclasses.asdict(best_run))}"
                        )
                    }
                )
            self.state.iteration += 1

            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=self.state.messages,
                tools=tools_list,
                # Not adding previous response id to control context internally
            )

            message = response.output_text

            self.observer.write_event(AgentEvent(
                timestamp = time.time_ns(),
                iteration=self.state.iteration,
                event="LLM response",
                message=response.model_dump_json(),
                tool_calls=self.state.tool_calls,
                errors=self.state.errors,
            ))

            tool_called = False

            # Execute every requested tool
            for item in response.output:
                if item.type == "function_call":
                    tool_called = True
                    tool_name = item.name
                    print (f"Calling tool {tool_name} with arguments {item.arguments}")

                    # Add the tool request message to the context as a proper
                    # function_call item (a raw JSON string is not a valid input item)
                    self.state.messages.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })

                    try:
                        arguments = json.loads(
                            item.arguments or "{}"
                        )

                        result = self._execute_tool(
                            item.name,
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

                    except RepeatedExperimentError as e:

                        result = RawExperimentResult(
                            output= "",
                            error= str(e),
                            errorcode= "repeated_experiment",
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
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(dataclasses.asdict(result)),
                        }
                    )
                    self.state.messages.append(
                        {
                            "role": "user",
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
                        arguments=item.arguments,
                        current_best_run=dataclasses.asdict(best_run),
                    ))
            if not tool_called:
                # No tool call needed anymore -> final answer reached
                if not response.tools:
                    self.observer.write_event(AgentEvent(
                        timestamp = time.time_ns(),
                        iteration=self.state.iteration,
                        event="Final answer",
                        message=message,
                        tool_calls=self.state.tool_calls,
                        errors=self.state.errors,
                    ))
                    return message
                

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
        self.validator.validate(name, arguments, self.state.iteration)

        if name == "run_experiment":
            arguments["benchmark_path"] = self.benchmark_path
            arguments["docker_image"] = self.docker_image

        return tool.function(
            **arguments
        )

    def _reached_limits(self) -> bool:
        if self.state.tool_calls >= self.max_tool_calls:
            self.observer.write_event(AgentEvent(
                    timestamp = time.time_ns(),
                    iteration=self.state.iteration,
                    event="Tool call limit",
                    message="Agent exceeded tool call limit",
                    tool_calls=self.state.tool_calls,
                    errors=self.state.errors,
                ))
            return True

        if self.state.iteration >= self.max_iterations:
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

