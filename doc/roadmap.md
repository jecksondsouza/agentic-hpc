# Roadmap

- ~~Create a config file for setting up LLM and API~~ - DONE
- New parameters for optimization (scheduling, deployment, numa nodes) 
- New non-functional requirements for optimization (energy, cost)
- Use perf instead of time for capturing more detailed information (cache misses, branch misses...)
- ~~Move from chat completion to responses API~~ - DONE
- ~~Refuse to run repeated experiments deterministically (sometimes the LLM will insist on repeating)~~ - DONE
- Further validate tool usage (e.g., check that resources exist - num_cpus <= actual number of cpus)