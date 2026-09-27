"""Aggregates the validation of tool calls. This validates that arguments
have the expected format, that they use valid data and that they are not repeated

Experiment signature
--------------------
Before a tool is executed, a tool call is rejected (with no execution)
when its arguments duplicate an experiment that was already run, or when
a hardware validator rejects one of its fields. The rejection is reported
to the LLM as a tool error so it can reason about it and fix its
arguments. Rejected calls are still recorded, so the same arguments
cannot be retried.

Hardware validators
-------------------
To support a new parameter (e.g., numa nodes, gpus, memory policies),
write a validator with the signature

    validator(value, profile: HardwareProfile) -> list[str]

returning one human-readable problem string per issue found (an empty
list means the value is valid), and register it in ``VALIDATORS`` under
the name of the argument field it checks. No changes to the agent or the
tools are needed: any tool whose arguments contain a registered field is
checked automatically.
"""

import json

from hardware_profile import HardwareProfile


class RepeatedExperimentError(Exception):
    """Raised when a tool call duplicates an experiment that was already run."""


def experiment_signature(name: str, arguments: dict) -> str:
    """Order-independent signature of a tool call's arguments."""
    if name == "run_experiment":
        cpus = ",".join(str(c) for c in sorted(set(arguments.get("cpus") or [])))
        threads = arguments.get("num_threads") or 0
        return f"run_experiment|cpus={cpus}|threads={threads}"
    return f"{name}|{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"


def _format_cpu_set(cpus: set[int]) -> str:
    """Render a set of cpu ids as a compact range string, e.g. '0-3,7'."""
    if not cpus:
        return ""
    ordered = sorted(cpus)
    parts = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _check_cpus(cpus: list[int], profile: HardwareProfile) -> list[str]:
    """Reject cpu ids that do not exist on this platform.

    An empty list is valid: it means no cpu set restriction.
    """
    if not cpus:
        return []
    available = {cpu["id"] for cpu in profile["cpus"]}
    invalid = [cpu for cpu in cpus if cpu not in available]
    if not invalid:
        return []
    return [
        f"cpus {invalid} do not exist on this platform; "
        f"available cpus: {_format_cpu_set(available)}"
    ]


VALIDATORS: dict = {
    "cpus": _check_cpus,
}


def validate_arguments(arguments: dict, profile: HardwareProfile) -> None:
    """Check the argument fields of a tool call against the hardware.

    Runs every validator registered for a field present in
    ``arguments``. Raises ``ValueError`` with all problems found (not
    just the first one) so the LLM sees every issue in a single tool
    error.
    """
    errors: list[str] = []
    for field, validator in VALIDATORS.items():
        if field in arguments:
            errors.extend(validator(arguments[field], profile))
    if errors:
        raise ValueError("; ".join(errors))


class ToolCallValidator:
    """Tracks executed experiments and validates tool-call arguments.

    One instance per agent run: the ``executed_experiments`` state must
    not be shared between runs. ``validate`` rejects duplicated or
    hardware-incompatible calls (recording them, so they cannot be
    retried).
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile
        self.executed_experiments: dict[str, int] = {}

    def validate(self, name: str, arguments: dict, iteration: int) -> None:
        """Check a tool call before execution; raise to reject it.

        Raises ``RepeatedExperimentError`` for duplicated arguments and
        ``ValueError`` for hardware-incompatible arguments. A rejected
        call is recorded as executed.
        """
        signature = experiment_signature(name, arguments)

        if signature in self.executed_experiments:
            first = self.executed_experiments[signature]
            raise RepeatedExperimentError(
                f"Experiment with identical arguments was already run at iteration {first} ({signature}). "
                "Choose different arguments."
            )
        self.executed_experiments[signature] = iteration

        validate_arguments(arguments, self.profile)
