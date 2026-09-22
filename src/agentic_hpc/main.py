# External modules
import argparse
from pathlib import Path

import yaml
from openai import OpenAI
from agent import Agent
from schemas import Config


USER_CONFIG_PATH = Path.home() / ".config" / "agentic-hpc" / "config.yaml"

CONFIG_FIELDS = [
    ("base_url", str, "OpenAI-compatible server URL"),
    ("api_key", str, "API key"),
    ("model", str, "Model name"),
    ("max_iterations", int, "Maximum number of LLM iterations"),
    ("max_tool_calls", int, "Maximum number of experiment runs"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HPC benchmark optimization agent."
    )
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser(
        "configure",
        help="Interactively set the user configuration "
             f"({USER_CONFIG_PATH})",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help="Path to the benchmark to execute",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Name of the docker image to run the experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file "
        "(by default: ./config.yaml, then ~/.config/agentic-hpc/config.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> Config:
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.append(Path("config.yaml"))
        candidates.append(Path.home() / ".config" / "agentic-hpc" / "config.yaml")

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open() as f:
                data = yaml.safe_load(f) or {}
            config = Config.model_validate(data)
            print(f"Loaded configuration from {candidate}")
            return config

    if config_path:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print("No config file found, using default configuration "
          f"({Config().base_url} / {Config().model})")
    return Config()


def read_user_config() -> Config:
    if USER_CONFIG_PATH.is_file():
        with USER_CONFIG_PATH.open() as f:
            return Config.model_validate(yaml.safe_load(f) or {})
    return Config()


def prompt_value(field_type: type, label: str, default: object) -> object:
    while True:
        try:
            raw = input(f"{label} [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        if field_type is int:
            try:
                return int(raw)
            except ValueError:
                print("  Please enter a whole number.")
                continue
        return raw


def run_configure() -> None:
    current = read_user_config()
    print(f"Configuring user config file: {USER_CONFIG_PATH}")
    print("Press ENTER to accept the suggested value, or type a new one.\n")

    values = {}
    for field, field_type, label in CONFIG_FIELDS:
        values[field] = prompt_value(
            field_type, label, getattr(current, field)
        )

    config = Config.model_validate(values)

    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONFIG_PATH.open("w") as f:
        yaml.safe_dump(
            config.model_dump(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"\nConfiguration written to {USER_CONFIG_PATH}")
    print(config.model_dump_json(indent=2))


def main():
    args = parse_args()

    if args.command == "configure":
        run_configure()
        return

    missing = [
        name
        for name, value in (("--benchmark-path", args.benchmark_path),
                             ("--docker-image", args.docker_image))
        if not value
    ]
    if missing:
        raise SystemExit(
            f"error: the following arguments are required: {' '.join(missing)}"
        )

    config = load_config(args.config)

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )

    my_agent = Agent(
        client,
        args.benchmark_path,
        args.docker_image,
        model=config.model,
        max_iterations=config.max_iterations,
        max_tool_calls=config.max_tool_calls,
    )

    answer = my_agent.run_agent(
        "Run the experiment and optimize my benchmark."
    )

    print(answer)


if __name__ == "__main__":
    main()
