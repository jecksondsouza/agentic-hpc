# External modules
import argparse

from openai import OpenAI
from agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HPC benchmark optimization agent."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the benchmark to execute",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        required=True,
        help="Name of the docker image to run the experiment",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
    )

    my_agent = Agent(client, args.benchmark_path, args.docker_image)

    answer = my_agent.run_agent(
        "Run the experiment and optimize my benchmark."
    )

    print(answer)


if __name__ == "__main__":
    main()
