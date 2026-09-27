import subprocess

from schemas import RawExperimentResult, RunExperimentArgs, Tool

tools_list = [
    {
        "type": "function",
        "name": "run_experiment",
        "description": "Run the benchmark (path and docker image fixed by the agent) along with several environment execution parameters.",
        "strict": "true",
        "parameters": {
            "type": "object",
            "properties": {
                "cpus": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of cpus that should be used to run the benchmark. "
                                    "A subset of cpus can be send by sending a subset list of the cpu range. "
                                    "For instance, sending [0,1,2,3] will use only cpus 0, 1, 2 and 3 to run the experiment. "
                                    "Only cpus that exist on the platform (see the hardware profile) are accepted; "
                                    "a call requesting non-existent cpus is rejected without running."
                },
                "num_threads": {
                    "type": "integer",
                    "description": "The number of threads the benchmark should spawn"
                },
                # We can explore other properties in the future, such as OMP_SCHEDULE, numa policy, disabling HT...
            },
            "required": [],
        },
    }
]



def run_experiment(benchmark_path: str, docker_image: str, cpus: list[int] = [], num_threads: int = 0) -> RawExperimentResult:

    cpu_list = f"--cpuset-cpus='{','.join(map(str,cpus))}' " if cpus else ""
    omp_threads = f"-e OMP_NUM_THREADS={num_threads} " if num_threads > 0 else ""

    command = f"docker run -it --rm  --privileged \
        {cpu_list}{omp_threads}{docker_image} bash -c 'TIMEFORMAT=%3R && time ./{benchmark_path}'"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )

    elapsed_seconds = next(
        float(line)
        for line in reversed(result.stdout.strip().splitlines())
        if line.strip().replace(".", "", 1).isdigit()
    )

    structured_result = RawExperimentResult(
        output=result.stdout, 
        error=result.stderr, 
        errorcode=result.returncode,
        args = (cpus, num_threads),
        execution_time_s=elapsed_seconds
        )

    return structured_result


tools_registry = {
    "run_experiment": Tool(
        name="run_experiment",
        function=run_experiment,
        args_schema=RunExperimentArgs,
    ),
}