#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/mpi/test_mpi_correctness.py

文件作用：
    该文件用于批量测试 MPI 版 All-Reduce 算法的正确性。

当前测试对象：
    1. src/mpi/ibing_mpi.py
    2. src/mpi/ring_mpi.py

测试方式：
    本文件不会直接 import 并调用 ibing_mpi.py / ring_mpi.py，
    而是通过 subprocess 启动真实 MPI 多进程命令。

    例如：
        mpiexec -n 5 python src/mpi/ibing_mpi.py --chunk_size 1024 --check

    这样可以更接近真实运行方式，确保每个 rank 都是独立 MPI 进程。

使用示例：
    默认同时测试 IBing 和 Ring：
        python src/mpi/test_mpi_correctness.py

    只测试 IBing：
        python src/mpi/test_mpi_correctness.py --algo ibing

    只测试 Ring：
        python src/mpi/test_mpi_correctness.py --algo ring

    指定进程数：
        python src/mpi/test_mpi_correctness.py --world_sizes 3 4 5 8

    指定 chunk_size：
        python src/mpi/test_mpi_correctness.py --chunk_sizes 4 1024 4096

    Windows / MS-MPI：
        python src/mpi/test_mpi_correctness.py --launcher mpiexec --np_flag -n

    Linux / OpenMPI：
        python src/mpi/test_mpi_correctness.py --launcher mpirun --np_flag -np

注意：
    1. Windows 上通常使用 mpiexec -n。
    2. Linux / WSL / OpenMPI 上通常使用 mpirun -np。
    3. 如果你的 mpiexec 已经可以正常运行，默认配置通常不用改。
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional


AlgorithmName = Literal["ibing", "ring", "both"]

DEFAULT_WORLD_SIZES = [3, 4, 5, 8]
DEFAULT_CHUNK_SIZES = [4, 1024]


@dataclass(frozen=True)
class MpiCorrectnessCase:
    """
    表示一个 MPI 正确性测试用例。

    Attributes:
        algo:
            算法名称，取值为 "ibing" 或 "ring"。

        world_size:
            MPI 进程数。

        chunk_size:
            每个 chunk 中的元素数量。

        dtype:
            通信数组使用的数据类型。

        value_scale:
            初始化数据时使用的 rank 间隔。
    """

    algo: str
    world_size: int
    chunk_size: int
    dtype: str
    value_scale: int


@dataclass(frozen=True)
class MpiCorrectnessResult:
    """
    保存一个 MPI 正确性测试用例的运行结果。

    Attributes:
        case:
            当前测试用例。

        command:
            实际执行的命令。

        return_code:
            子进程返回码。0 表示成功，非 0 表示失败。

        passed:
            当前测试是否通过。

        stdout:
            子进程标准输出。

        stderr:
            子进程标准错误。

        elapsed_seconds:
            subprocess 运行耗时，单位为秒。
    """

    case: MpiCorrectnessCase
    command: List[str]
    return_code: int
    passed: bool
    stdout: str
    stderr: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, str | int | float | bool]:
        """
        将结果转换为字典，便于保存为 CSV。

        Returns:
            当前测试结果对应的字典。
        """

        return {
            "algo": self.case.algo,
            "world_size": self.case.world_size,
            "chunk_size": self.case.chunk_size,
            "dtype": self.case.dtype,
            "value_scale": self.case.value_scale,
            "return_code": self.return_code,
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "command": " ".join(self.command),
        }


def validate_world_size(world_size: int) -> None:
    """
    检查 MPI 进程数是否合法。

    Args:
        world_size:
            MPI 进程数。

    Raises:
        ValueError:
            当 world_size 小于 2 时抛出异常。
    """

    if world_size < 2:
        raise ValueError(f"world_size must be >= 2, but got {world_size}.")


def validate_chunk_size(chunk_size: int) -> None:
    """
    检查 chunk_size 是否合法。

    Args:
        chunk_size:
            每个 chunk 的元素数量。

    Raises:
        ValueError:
            当 chunk_size 小于 1 时抛出异常。
    """

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, but got {chunk_size}.")


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件路径为：
        src/mpi/test_mpi_correctness.py

    因此：
        当前文件 parent      = src/mpi
        parent.parent        = src
        parent.parent.parent = 项目根目录

    Returns:
        项目根目录路径。
    """

    current_file = Path(__file__).resolve()
    mpi_dir = current_file.parent
    src_dir = mpi_dir.parent
    project_root = src_dir.parent
    return project_root


def get_script_path(algo: str) -> Path:
    """
    根据算法名称获取对应 MPI 脚本路径。

    Args:
        algo:
            算法名称，取值为 "ibing" 或 "ring"。

    Returns:
        对应脚本的绝对路径。

    Raises:
        ValueError:
            当 algo 不合法时抛出异常。
    """

    project_root = get_project_root()

    if algo == "ibing":
        return project_root / "src" / "mpi" / "ibing_mpi.py"

    if algo == "ring":
        return project_root / "src" / "mpi" / "ring_mpi.py"

    raise ValueError(f"Unsupported algo: {algo}")


def infer_default_launcher() -> tuple[str, str]:
    """
    根据当前操作系统推断默认 MPI 启动器和进程数参数。

    Returns:
        launcher:
            MPI 启动器名称。

        np_flag:
            指定进程数的参数。

    说明：
        Windows / MS-MPI 常用：
            mpiexec -n 5 ...

        Linux / OpenMPI 常用：
            mpirun -np 5 ...

        这里为了兼容 Windows，默认 Windows 返回 mpiexec -n；
        非 Windows 返回 mpirun -np。
    """

    system_name = platform.system().lower()

    if "windows" in system_name:
        return "mpiexec", "-n"

    return "mpirun", "-np"


def resolve_algorithms(algo: AlgorithmName) -> List[str]:
    """
    根据命令行算法选项解析实际要测试的算法列表。

    Args:
        algo:
            "ibing"、"ring" 或 "both"。

    Returns:
        需要测试的算法列表。
    """

    if algo == "both":
        return ["ibing", "ring"]

    return [algo]


def build_test_cases(
    algorithms: Iterable[str],
    world_sizes: Iterable[int],
    chunk_sizes: Iterable[int],
    dtype: str,
    value_scale: int,
) -> List[MpiCorrectnessCase]:
    """
    构造所有 MPI 正确性测试用例。

    Args:
        algorithms:
            算法列表。

        world_sizes:
            MPI 进程数列表。

        chunk_sizes:
            chunk_size 列表。

        dtype:
            数据类型。

        value_scale:
            初始化数据间隔。

    Returns:
        测试用例列表。
    """

    cases: List[MpiCorrectnessCase] = []

    for world_size in world_sizes:
        validate_world_size(world_size)

    for chunk_size in chunk_sizes:
        validate_chunk_size(chunk_size)

    for algo in algorithms:
        for world_size in world_sizes:
            for chunk_size in chunk_sizes:
                case = MpiCorrectnessCase(
                    algo=algo,
                    world_size=world_size,
                    chunk_size=chunk_size,
                    dtype=dtype,
                    value_scale=value_scale,
                )
                cases.append(case)

    return cases


def build_command(
    case: MpiCorrectnessCase,
    launcher: str,
    np_flag: str,
    python_executable: str,
) -> List[str]:
    """
    为某个测试用例构造实际执行的 MPI 命令。

    Args:
        case:
            测试用例。

        launcher:
            MPI 启动器，例如 mpiexec 或 mpirun。

        np_flag:
            指定进程数的参数，例如 -n 或 -np。

        python_executable:
            Python 解释器路径。默认建议使用 sys.executable，
            这样可以确保使用当前虚拟环境中的 Python。

    Returns:
        命令列表，可直接传给 subprocess.run。
    """

    script_path = get_script_path(case.algo)

    command = [
        launcher,
        np_flag,
        str(case.world_size),
        python_executable,
        str(script_path),
        "--chunk_size",
        str(case.chunk_size),
        "--dtype",
        case.dtype,
        "--value_scale",
        str(case.value_scale),
        "--check",
    ]

    return command


def run_one_case(
    case: MpiCorrectnessCase,
    launcher: str,
    np_flag: str,
    python_executable: str,
    timeout: Optional[int],
    verbose: bool,
) -> MpiCorrectnessResult:
    """
    运行单个 MPI 正确性测试用例。

    Args:
        case:
            测试用例。

        launcher:
            MPI 启动器。

        np_flag:
            指定进程数的参数。

        python_executable:
            Python 解释器路径。

        timeout:
            子进程超时时间，单位秒。
            None 表示不设置超时。

        verbose:
            是否打印子进程完整 stdout / stderr。

    Returns:
        当前测试用例的运行结果。
    """

    import time

    command = build_command(
        case=case,
        launcher=launcher,
        np_flag=np_flag,
        python_executable=python_executable,
    )

    start_time = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=str(get_project_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        elapsed_seconds = time.perf_counter() - start_time

        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode

    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time

        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

        return MpiCorrectnessResult(
            case=case,
            command=command,
            return_code=-1,
            passed=False,
            stdout=stdout,
            stderr=stderr + f"\n[TIMEOUT] Exceeded timeout={timeout} seconds.",
            elapsed_seconds=elapsed_seconds,
        )

    # 正确性判定：
    #   1. 子进程返回码为 0；
    #   2. 输出中包含 [PASS]；
    #   3. 输出中包含 global_correct。
    #
    # 注意：
    #   ibing_mpi.py 和 ring_mpi.py 自身已经会在 global_correct=False 时返回非 0。
    #   这里额外检查输出关键字，是为了避免误判。
    passed = (
        return_code == 0
        and "[PASS]" in stdout
        and "global_correct" in stdout
    )

    result = MpiCorrectnessResult(
        case=case,
        command=command,
        return_code=return_code,
        passed=passed,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=elapsed_seconds,
    )

    if verbose:
        print_case_output(result)

    return result


def print_case_output(result: MpiCorrectnessResult) -> None:
    """
    打印某个测试用例的完整输出。

    Args:
        result:
            测试结果。
    """

    print()
    print("-" * 80)
    print("Command:")
    print(" ".join(result.command))
    print("-" * 80)
    print("STDOUT:")
    print(result.stdout.strip() if result.stdout.strip() else "<empty>")
    print("-" * 80)
    print("STDERR:")
    print(result.stderr.strip() if result.stderr.strip() else "<empty>")
    print("-" * 80)


def run_all_cases(
    cases: Iterable[MpiCorrectnessCase],
    launcher: str,
    np_flag: str,
    python_executable: str,
    timeout: Optional[int],
    verbose: bool,
) -> List[MpiCorrectnessResult]:
    """
    批量运行所有 MPI 正确性测试用例。

    Args:
        cases:
            测试用例列表。

        launcher:
            MPI 启动器。

        np_flag:
            进程数参数。

        python_executable:
            Python 解释器路径。

        timeout:
            每个用例的超时时间。

        verbose:
            是否打印详细输出。

    Returns:
        测试结果列表。
    """

    results: List[MpiCorrectnessResult] = []

    cases = list(cases)
    total_cases = len(cases)

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{total_cases}] "
            f"algo={case.algo}, "
            f"world_size={case.world_size}, "
            f"chunk_size={case.chunk_size}, "
            f"dtype={case.dtype} ... ",
            end="",
            flush=True,
        )

        result = run_one_case(
            case=case,
            launcher=launcher,
            np_flag=np_flag,
            python_executable=python_executable,
            timeout=timeout,
            verbose=verbose,
        )

        results.append(result)

        if result.passed:
            print(f"PASS ({result.elapsed_seconds:.3f}s)")
        else:
            print(f"FAIL ({result.elapsed_seconds:.3f}s)")
            print_failure_detail(result)

    return results


def print_failure_detail(result: MpiCorrectnessResult) -> None:
    """
    打印失败测试用例的关键信息。

    Args:
        result:
            失败的测试结果。
    """

    print()
    print("=" * 80)
    print("[FAIL DETAIL]")
    print("=" * 80)
    print(f"algo       : {result.case.algo}")
    print(f"world_size : {result.case.world_size}")
    print(f"chunk_size : {result.case.chunk_size}")
    print(f"dtype      : {result.case.dtype}")
    print(f"return_code: {result.return_code}")
    print("command    :")
    print(" ".join(result.command))
    print("-" * 80)

    print("STDOUT:")
    print(result.stdout.strip() if result.stdout.strip() else "<empty>")
    print("-" * 80)

    print("STDERR:")
    print(result.stderr.strip() if result.stderr.strip() else "<empty>")
    print("=" * 80)
    print()


def save_results_to_csv(
    results: List[MpiCorrectnessResult],
    output_path: str | Path,
) -> None:
    """
    保存测试结果到 CSV 文件。

    Args:
        results:
            测试结果列表。

        output_path:
            CSV 输出路径。
    """

    output_path = Path(output_path)

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algo",
        "world_size",
        "chunk_size",
        "dtype",
        "value_scale",
        "return_code",
        "passed",
        "elapsed_seconds",
        "command",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.as_dict())

    print(f"[OK] MPI correctness results saved to: {output_path}")


def print_summary(results: List[MpiCorrectnessResult]) -> None:
    """
    打印所有 MPI 正确性测试的摘要。

    Args:
        results:
            测试结果列表。
    """

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed

    print()
    print("=" * 80)
    print("MPI Correctness Test Summary")
    print("=" * 80)
    print(f"total cases : {total}")
    print(f"passed      : {passed}")
    print(f"failed      : {failed}")
    print("-" * 80)

    by_algo: dict[str, list[MpiCorrectnessResult]] = {}

    for result in results:
        by_algo.setdefault(result.case.algo, []).append(result)

    for algo, algo_results in sorted(by_algo.items()):
        algo_total = len(algo_results)
        algo_passed = sum(1 for result in algo_results if result.passed)
        print(f"{algo:<8}: {algo_passed}/{algo_total} passed")

    print("-" * 80)

    if failed == 0:
        print("[PASS] All MPI correctness tests passed.")
    else:
        print("[FAIL] Some MPI correctness tests failed.")

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace:
            命令行参数对象。
    """

    default_launcher, default_np_flag = infer_default_launcher()

    parser = argparse.ArgumentParser(
        description="Batch correctness tests for MPI Ring and IBing All-Reduce."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ibing", "ring", "both"],
        default="both",
        help="Algorithm to test. Default: both.",
    )

    parser.add_argument(
        "--world_sizes",
        type=int,
        nargs="*",
        default=DEFAULT_WORLD_SIZES,
        help=(
            "MPI process counts to test. "
            "Default: 3 4 5 8."
        ),
    )

    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="*",
        default=DEFAULT_CHUNK_SIZES,
        help=(
            "Chunk sizes to test. "
            "Default: 4 1024."
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Data type used by MPI scripts. Default: float32.",
    )

    parser.add_argument(
        "--value_scale",
        type=int,
        default=10,
        help="Scale used to initialize worker data. Default: 10.",
    )

    parser.add_argument(
        "--launcher",
        type=str,
        default=default_launcher,
        help=(
            "MPI launcher. "
            "Windows usually uses mpiexec. "
            "Linux/OpenMPI usually uses mpirun."
        ),
    )

    parser.add_argument(
        "--np_flag",
        type=str,
        default=default_np_flag,
        help=(
            "Flag for number of MPI processes. "
            "Windows/MS-MPI usually uses -n. "
            "OpenMPI often uses -np."
        ),
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=(
            "Python executable used inside MPI processes. "
            "Default: current Python executable."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help=(
            "Timeout seconds for each MPI case. "
            "Default: 60."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full stdout/stderr for each test case.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional CSV output path. "
            "Example: --output results/tables/mpi_correctness.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析命令行参数；
        2. 构造 MPI 正确性测试用例；
        3. 逐个通过 subprocess 启动真实 MPI 多进程命令；
        4. 收集返回码、stdout、stderr；
        5. 打印测试摘要；
        6. 如果指定 --output，则保存 CSV；
        7. 如果任意用例失败，则返回非 0 退出码。
    """

    args = parse_args()

    algorithms = resolve_algorithms(args.algo)

    cases = build_test_cases(
        algorithms=algorithms,
        world_sizes=args.world_sizes,
        chunk_sizes=args.chunk_sizes,
        dtype=args.dtype,
        value_scale=args.value_scale,
    )

    print("=" * 80)
    print("MPI All-Reduce Correctness Batch Test")
    print("=" * 80)
    print(f"algorithms       : {algorithms}")
    print(f"world_sizes      : {args.world_sizes}")
    print(f"chunk_sizes      : {args.chunk_sizes}")
    print(f"dtype            : {args.dtype}")
    print(f"value_scale      : {args.value_scale}")
    print(f"launcher         : {args.launcher}")
    print(f"np_flag          : {args.np_flag}")
    print(f"python           : {args.python}")
    print(f"timeout          : {args.timeout}")
    print(f"total test cases : {len(cases)}")
    print("=" * 80)
    print()

    results = run_all_cases(
        cases=cases,
        launcher=args.launcher,
        np_flag=args.np_flag,
        python_executable=args.python,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    print_summary(results)

    if args.output is not None:
        save_results_to_csv(
            results=results,
            output_path=args.output,
        )

    failed_count = sum(1 for result in results if not result.passed)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()