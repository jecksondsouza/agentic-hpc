from pathlib import Path
from typing import List
import re

SYS_CPU_PATH = Path("/sys/devices/system/cpu")


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _read_str(path: Path) -> str:
    return path.read_text().strip()


def _parse_cpu_list(text: str) -> List[int]:
    '''
        Split the input in a list of CPUs separated by commas or
        in a range of CPUs delimited by a dash
    '''
    cpus: List[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _build_cache(index_dir: Path) -> Cache:
    shared_cpu = index_dir / "shared_cpu_list"
    shared_cpus = _parse_cpu_list(_read_str(shared_cpu)) if shared_cpu.exists() else []
    return Cache(
        id=_read_int(index_dir / "id"),
        level=_read_int(index_dir / "level"),
        type=_read_str(index_dir / "type"),
        coherency_line_size=_read_int(index_dir / "coherency_line_size"),
        number_of_sets=_read_int(index_dir / "number_of_sets"),
        ways_of_associativity=_read_int(index_dir / "ways_of_associativity"),
        size=_read_str(index_dir / "size"),
        shared_cpu_list=shared_cpus,
    )


def _build_cpu(cpu_dir: Path) -> CPU:
    cache_dir = cpu_dir / "cache"
    match = re.search(r'(\d+)$', cpu_dir.as_posix())
    id = int(match.group(1)) if match else -1
    caches = [_build_cache(d) for d in sorted(cache_dir.iterdir()) if d.is_dir()]
    return CPU(id, caches=caches)


class Cache(dict):
    def __init__(
        self,
        id: int, # Not a unique id really, but a representation of the socket the cache is located in
        level: int,
        type: str,  # Instruction, Data, Unified
        coherency_line_size: int,
        number_of_sets: int,
        ways_of_associativity: int,
        size: str,  # In KB
        shared_cpu_list: List[int],  # CPU ids
    ) -> None:
        dict.__init__(self,
            id = id,
            level = level,
            type = type,
            coherency_line_size = coherency_line_size,
            number_of_sets = number_of_sets,
            ways_of_associativity = ways_of_associativity,
            size = size,
            shared_cpu_list = shared_cpu_list)


class CPU(dict):
    def __init__(self, id, caches: List[Cache] = None) -> None:
        # Each cache in the CPU
        dict.__init__(self,
            id = id,
            caches = caches or [])


class HardwareProfile(dict):
    def __init__(self) -> None:
        # Each cpu in the system
        self.cpus: List[CPU] = []
        for cpu_dir in sorted(SYS_CPU_PATH.iterdir()):
            if cpu_dir.is_dir() and cpu_dir.name.startswith("cpu") and (cpu_dir / "cache").is_dir():
                # Assuming that a CPU must have cache memory
                self.cpus.append(_build_cpu(cpu_dir))
        dict.__init__(self, cpus=self.cpus)
