#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/run_benchmark.py

文件作用：
    该文件用于自动批量运行 MPI benchmark。

    前面你已经实现了：

        src/mpi/benchmark.py

    它需要通过 mpiexec 或 mpirun 启动，例如：

        mpiexec -n 5 python src/mpi/benchmark.py --algo all --data_sizes_mb 1 10 50

    但是如果要测试多个进程数，例如：

        N = 3, 4, 5, 8

    手动运行会比较繁琐。因此本文件负责自动构造并执行这些命令。

当前功能：
    1. 支持批量测试多个 world_size；
    2. 支持批量测试多个 data_size_mb；
    3. 支持 ring / ibing / mpi_allreduce / all；
    4. 支持 warmup 和 repeat 设置；
    5. 自动为每个 world_size 保存独立 CSV；
    6. 自动合并所有 CSV 为总结果文件；
    7. 支持 Windows MS-MPI 和 Linux OpenMPI；
    8. 支持失败时打印 stdout / stderr，方便排错。

使用示例：
    Windows / MS-MPI:
        python scripts/run_benchmark.py --world_sizes 3 4 5 8 --data_sizes_mb 1 10 50

    Linux / OpenMPI:
        python scripts/run_benchmark.py --launcher mpirun --np_flag -np --world_sizes 3 4 5 8

    只测试 Ring 和 IBing，不测试 MPI_Allreduce:
        python scripts/run_benchmark.py --algo ring
        python scripts/run_benchmark.py --algo ibing

    保存到指定目录:
        python scripts/run_benchmark.py --output_dir results/raw --merged_output results/tables/mpi_benchmark_all.csv

说明：
    本文件本身不是 MPI 程序，不需要用 mpiexec 启动。

    正确运行方式是：

        python scripts/run_benchmark.py

    而不是：

        mpiexec -n 5 python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional


AlgorithmName = Literal["ring", "ibing", "mpi_allreduce", "all"]

DEFAULT_WORLD_SIZES = [3, 4, 5, 8]
DEFAULT_DATA_SIZES_MB = [1.0, 10.0, 50.0]


@dataclass(frozen=True)
class BenchmarkRunCase:
    """
    表示一次 benchmark 子进程运行任务。

    Attributes:
        world_size:
            MPI 进程数。

        algo:
            测试算法。

        data_sizes_mb:
            每个 rank 的数据大小列表，单位 MiB。

        dtype:
            数据类型。

        warmup:
            预热轮数。

        repeat:
            正式计时轮数。

        output_path:
            当前 world_size 对应的原始 CSV 输出路径。
    """

    world_size: int
    algo: str
    data_sizes_mb: List[float]
    dtype: str
    warmup: int
    repeat: int
    output_path: Path


@dataclass(frozen=True)
class BenchmarkRunResult:
    """
    保存一次 benchmark 子进程运行结果。

    Attributes:
        case:
            对应的 benchmark 运行任务。

        command:
            实际执行的命令。

        return_code:
            子进程返回码。

        passed:
            是否运行成功。

        elapsed_seconds:
            当前子进程总耗时。

        stdout:
            标准输出。

        stderr:
            标准错误。
    """

    case: BenchmarkRunCase
    command: List[str]
    return_code: int
    passed: bool
    elapsed_seconds: float
    stdout: str
    stderr: str


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件位于：

        scripts/run_benchmark.py

    因此：

        当前文件 parent        = scripts
        当前文件 parent.parent = 项目根目录

    Returns:
        项目根目录路径。
    """

    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent
    project_root = scripts_dir.parent
    return project_root


def get_benchmark_script_path() -> Path:
    """
    获取 src/mpi/benchmark.py 的绝对路径。

    Returns:
        benchmark.py 的路径。
    """

    return get_project_root() / "src" / "mpi" / "benchmark.py"


def infer_default_launcher() -> tuple[str, str]:
    """
    根据操作系统推断默认 MPI 启动器。

    Windows / MS-MPI:
        mpiexec -n

    Linux / OpenMPI:
        mpirun -np

    Returns:
        launcher:
            MPI 启动器。

        np_flag:
            指定进程数的参数。
    """

    system_name = platform.system().lower()

    if "windows" in system_name:
        return "mpiexec", "-n"

    return "mpirun", "-np"


def validate_world_size(world_size: int) -> None:
    """
    检查 world_size 是否合法。

    Args:
        world_size:
            MPI 进程数。
    """

    if world_size < 2:
        raise ValueError(f"world_size must be >= 2, but got {world_size}.")


def validate_data_size(data_size_mb: float) -> None:
    """
    检查 data_size_mb 是否合法。

    Args:
        data_size_mb:
            每个 rank 的数据大小，单位 MiB。
    """

    if data_size_mb <= 0:
        raise ValueError(f"data_size_mb must be > 0, but got {data_size_mb}.")


def validate_repeat_and_warmup(repeat: int, warmup: int) -> None:
    """
    检查 repeat 和 warmup 是否合法。

    Args:
        repeat:
            正式测试轮数。

        warmup:
            预热轮数。
    """

    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, but got {repeat}.")

    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, but got {warmup}.")


def ensure_output_dirs(output_dir: Path, merged_output: Path) -> None:
    """
    创建输出目录。

    Args:
        output_dir:
            每个 world_size 原始 CSV 的输出目录。

        merged_output:
            合并 CSV 的输出路径。
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if merged_output.parent:
        merged_output.parent.mkdir(parents=True, exist_ok=True)


def build_output_path(
    output_dir: Path,
    world_size: int,
    algo: str,
    dtype: str,
) -> Path:
    """
    构造某个 world_size 下的原始 CSV 输出路径。

    Args:
        output_dir:
            原始 CSV 输出目录。

        world_size:
            MPI 进程数。

        algo:
            算法名称。

        dtype:
            数据类型。

    Returns:
        CSV 输出路径。
    """

    filename = f"mpi_benchmark_n{world_size}_{algo}_{dtype}.csv"
    return output_dir / filename


def build_run_cases(
    world_sizes: Iterable[int],
    algo: str,
    data_sizes_mb: Iterable[float],
    dtype: str,
    warmup: int,
    repeat: int,
    output_dir: Path,
) -> List[BenchmarkRunCase]:
    """
    构造所有 benchmark 运行任务。

    Args:
        world_sizes:
            MPI 进程数列表。

        algo:
            测试算法。

        data_sizes_mb:
            数据大小列表。

        dtype:
            数据类型。

        warmup:
            预热轮数。

        repeat:
            正式计时轮数。

        output_dir:
            原始 CSV 输出目录。

    Returns:
        BenchmarkRunCase 列表。
    """

    validate_repeat_and_warmup(repeat=repeat, warmup=warmup)

    world_sizes = list(world_sizes)
    data_sizes_mb = list(data_sizes_mb)

    for world_size in world_sizes:
        validate_world_size(world_size)

    for data_size in data_sizes_mb:
        validate_data_size(data_size)

    cases: List[BenchmarkRunCase] = []

    for world_size in world_sizes:
        output_path = build_output_path(
            output_dir=output_dir,
            world_size=world_size,
            algo=algo,
            dtype=dtype,
        )

        case = BenchmarkRunCase(
            world_size=world_size,
            algo=algo,
            data_sizes_mb=data_sizes_mb,
            dtype=dtype,
            warmup=warmup,
            repeat=repeat,
            output_path=output_path,
        )
        cases.append(case)

    return cases


def build_command(
    case: BenchmarkRunCase,
    launcher: str,
    np_flag: str,
    python_executable: str,
    skip_check: bool,
) -> List[str]:
    """
    构造实际执行的 MPI benchmark 命令。

    Args:
        case:
            benchmark 运行任务。

        launcher:
            MPI 启动器，例如 mpiexec 或 mpirun。

        np_flag:
            进程数参数，例如 -n 或 -np。

        python_executable:
            Python 解释器路径。

        skip_check:
            是否跳过正确性检查。

    Returns:
        可传给 subprocess.run 的命令列表。
    """

    benchmark_script = get_benchmark_script_path()

    command = [
        launcher,
        np_flag,
        str(case.world_size),
        python_executable,
        str(benchmark_script),
        "--algo",
        case.algo,
        "--data_sizes_mb",
    ]

    command.extend(str(size) for size in case.data_sizes_mb)

    command.extend(
        [
            "--dtype",
            case.dtype,
            "--warmup",
            str(case.warmup),
            "--repeat",
            str(case.repeat),
            "--output",
            str(case.output_path),
        ]
    )

    if skip_check:
        command.append("--skip_check")

    return command


def run_one_case(
    case: BenchmarkRunCase,
    launcher: str,
    np_flag: str,
    python_executable: str,
    timeout: Optional[int],
    skip_check: bool,
    verbose: bool,
) -> BenchmarkRunResult:
    """
    运行一个 benchmark 子进程任务。

    Args:
        case:
            benchmark 运行任务。

        launcher:
            MPI 启动器。

        np_flag:
            进程数参数。

        python_executable:
            Python 解释器路径。

        timeout:
            超时时间，单位秒。

        skip_check:
            是否跳过正确性检查。

        verbose:
            是否打印完整 stdout/stderr。

    Returns:
        BenchmarkRunResult。
    """

    command = build_command(
        case=case,
        launcher=launcher,
        np_flag=np_flag,
        python_executable=python_executable,
        skip_check=skip_check,
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

        result = BenchmarkRunResult(
            case=case,
            command=command,
            return_code=completed.returncode,
            passed=completed.returncode == 0,
            elapsed_seconds=elapsed_seconds,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time

        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

        result = BenchmarkRunResult(
            case=case,
            command=command,
            return_code=-1,
            passed=False,
            elapsed_seconds=elapsed_seconds,
            stdout=stdout,
            stderr=stderr + f"\n[TIMEOUT] Exceeded timeout={timeout} seconds.",
        )

    if verbose:
        print_run_output(result)

    return result


def print_run_output(result: BenchmarkRunResult) -> None:
    """
    打印一次 benchmark 子进程的完整输出。

    Args:
        result:
            子进程运行结果。
    """

    print()
    print("-" * 100)
    print("Command:")
    print(" ".join(result.command))
    print("-" * 100)
    print("STDOUT:")
    print(result.stdout.strip() if result.stdout.strip() else "<empty>")
    print("-" * 100)
    print("STDERR:")
    print(result.stderr.strip() if result.stderr.strip() else "<empty>")
    print("-" * 100)


def print_failure_detail(result: BenchmarkRunResult) -> None:
    """
    打印失败任务的详细信息。

    Args:
        result:
            失败任务结果。
    """

    print()
    print("=" * 100)
    print("[FAIL DETAIL]")
    print("=" * 100)
    print(f"world_size : {result.case.world_size}")
    print(f"algo       : {result.case.algo}")
    print(f"dtype      : {result.case.dtype}")
    print(f"return_code: {result.return_code}")
    print(f"elapsed    : {result.elapsed_seconds:.3f}s")
    print("command    :")
    print(" ".join(result.command))
    print("-" * 100)
    print("STDOUT:")
    print(result.stdout.strip() if result.stdout.strip() else "<empty>")
    print("-" * 100)
    print("STDERR:")
    print(result.stderr.strip() if result.stderr.strip() else "<empty>")
    print("=" * 100)
    print()


def run_all_cases(
    cases: Iterable[BenchmarkRunCase],
    launcher: str,
    np_flag: str,
    python_executable: str,
    timeout: Optional[int],
    skip_check: bool,
    verbose: bool,
) -> List[BenchmarkRunResult]:
    """
    批量运行所有 benchmark 任务。

    Args:
        cases:
            benchmark 任务列表。

        launcher:
            MPI 启动器。

        np_flag:
            进程数参数。

        python_executable:
            Python 解释器路径。

        timeout:
            每个任务的超时时间。

        skip_check:
            是否跳过正确性检查。

        verbose:
            是否打印详细输出。

    Returns:
        BenchmarkRunResult 列表。
    """

    cases = list(cases)
    total = len(cases)
    results: List[BenchmarkRunResult] = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{total}] "
            f"world_size={case.world_size}, "
            f"algo={case.algo}, "
            f"data_sizes={case.data_sizes_mb}, "
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
            skip_check=skip_check,
            verbose=verbose,
        )

        results.append(result)

        if result.passed:
            print(f"PASS ({result.elapsed_seconds:.3f}s)")
        else:
            print(f"FAIL ({result.elapsed_seconds:.3f}s)")
            print_failure_detail(result)

    return results


def read_csv_rows(csv_path: Path) -> List[dict[str, str]]:
    """
    读取一个 benchmark CSV 文件。

    Args:
        csv_path:
            CSV 文件路径。

    Returns:
        CSV 行列表。
    """

    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return rows


def merge_result_csvs(
    run_results: List[BenchmarkRunResult],
    merged_output: Path,
) -> None:
    """
    合并所有成功运行产生的 benchmark CSV。

    Args:
        run_results:
            子进程运行结果列表。

        merged_output:
            合并后的总 CSV 输出路径。
    """

    all_rows: List[dict[str, str]] = []

    for run_result in run_results:
        if not run_result.passed:
            continue

        csv_path = run_result.case.output_path
        rows = read_csv_rows(csv_path)

        for row in rows:
            row["run_world_size"] = str(run_result.case.world_size)
            row["run_command"] = " ".join(run_result.command)
            all_rows.append(row)

    if not all_rows:
        print("[WARN] No CSV rows found to merge.")
        return

    # 保证字段顺序稳定。
    preferred_fields = [
        "algo",
        "world_size",
        "run_world_size",
        "data_size_mb_target",
        "data_size_mb_actual",
        "chunk_size",
        "dtype",
        "warmup",
        "repeat",
        "value_scale",
        "step_count",
        "correct",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "run_command",
    ]

    existing_fields = list(all_rows[0].keys())
    fieldnames = []

    for field in preferred_fields:
        if field in existing_fields and field not in fieldnames:
            fieldnames.append(field)

    for field in existing_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    if merged_output.parent:
        merged_output.parent.mkdir(parents=True, exist_ok=True)

    with merged_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            writer.writerow(row)

    print(f"[OK] Merged benchmark CSV saved to: {merged_output}")


def save_run_log(
    run_results: List[BenchmarkRunResult],
    log_output: Path,
) -> None:
    """
    保存每个子进程运行任务的日志摘要。

    Args:
        run_results:
            子进程运行结果列表。

        log_output:
            运行日志 CSV 输出路径。
    """

    if log_output.parent:
        log_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "world_size",
        "algo",
        "data_sizes_mb",
        "dtype",
        "warmup",
        "repeat",
        "output_path",
        "return_code",
        "passed",
        "elapsed_seconds",
        "command",
    ]

    with log_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in run_results:
            writer.writerow(
                {
                    "world_size": result.case.world_size,
                    "algo": result.case.algo,
                    "data_sizes_mb": " ".join(
                        str(size) for size in result.case.data_sizes_mb
                    ),
                    "dtype": result.case.dtype,
                    "warmup": result.case.warmup,
                    "repeat": result.case.repeat,
                    "output_path": str(result.case.output_path),
                    "return_code": result.return_code,
                    "passed": result.passed,
                    "elapsed_seconds": result.elapsed_seconds,
                    "command": " ".join(result.command),
                }
            )

    print(f"[OK] Benchmark run log saved to: {log_output}")


def print_summary(results: List[BenchmarkRunResult]) -> None:
    """
    打印批量 benchmark 的运行摘要。

    Args:
        results:
            子进程运行结果列表。
    """

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed

    print()
    print("=" * 100)
    print("Batch Benchmark Summary")
    print("=" * 100)
    print(f"total runs : {total}")
    print(f"passed     : {passed}")
    print(f"failed     : {failed}")
    print("-" * 100)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] "
            f"N={result.case.world_size}, "
            f"algo={result.case.algo}, "
            f"dtype={result.case.dtype}, "
            f"elapsed={result.elapsed_seconds:.3f}s, "
            f"csv={result.case.output_path}"
        )

    print("-" * 100)

    if failed == 0:
        print("[PASS] All benchmark runs completed successfully.")
    else:
        print("[FAIL] Some benchmark runs failed.")

    print("=" * 100)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace。
    """

    default_launcher, default_np_flag = infer_default_launcher()

    parser = argparse.ArgumentParser(
        description="Batch runner for MPI All-Reduce benchmark."
    )

    parser.add_argument(
        "--world_sizes",
        type=int,
        nargs="*",
        default=DEFAULT_WORLD_SIZES,
        help="MPI process counts to test. Default: 3 4 5 8.",
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["ring", "ibing", "mpi_allreduce", "all"],
        default="all",
        help="Algorithm to benchmark. Default: all.",
    )

    parser.add_argument(
        "--data_sizes_mb",
        type=float,
        nargs="*",
        default=DEFAULT_DATA_SIZES_MB,
        help="Data sizes per rank in MiB. Default: 1 10 50.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Data type. Default: float32.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations. Default: 5.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=30,
        help="Timed iterations. Default: 30.",
    )

    parser.add_argument(
        "--launcher",
        type=str,
        default=default_launcher,
        help="MPI launcher. Windows usually uses mpiexec; Linux often uses mpirun.",
    )

    parser.add_argument(
        "--np_flag",
        type=str,
        default=default_np_flag,
        help="Flag for process count. Windows/MS-MPI usually uses -n; OpenMPI uses -np.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used inside MPI processes. Default: current Python.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout seconds for each world_size run. Default: 600.",
    )

    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Pass --skip_check to src/mpi/benchmark.py.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/raw",
        help="Directory for per-world-size raw CSV files. Default: results/raw.",
    )

    parser.add_argument(
        "--merged_output",
        type=str,
        default="results/tables/mpi_benchmark_all.csv",
        help="Merged CSV output path. Default: results/tables/mpi_benchmark_all.csv.",
    )

    parser.add_argument(
        "--log_output",
        type=str,
        default="results/tables/mpi_benchmark_run_log.csv",
        help="Run log CSV output path. Default: results/tables/mpi_benchmark_run_log.csv.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full stdout/stderr for each benchmark subprocess.",
    )

    return parser.parse_args()


def main() -> None:
    """
    程序入口。

    执行流程：
        1. 解析参数；
        2. 构造多个 world_size 的 benchmark 命令；
        3. 逐个调用 mpiexec / mpirun；
        4. 每个 world_size 输出一个原始 CSV；
        5. 合并所有原始 CSV 为总 CSV；
        6. 保存运行日志；
        7. 如果任意任务失败，则返回非 0。
    """

    args = parse_args()

    output_dir = Path(args.output_dir)
    merged_output = Path(args.merged_output)
    log_output = Path(args.log_output)

    ensure_output_dirs(
        output_dir=output_dir,
        merged_output=merged_output,
    )

    cases = build_run_cases(
        world_sizes=args.world_sizes,
        algo=args.algo,
        data_sizes_mb=args.data_sizes_mb,
        dtype=args.dtype,
        warmup=args.warmup,
        repeat=args.repeat,
        output_dir=output_dir,
    )

    print("=" * 100)
    print("MPI Benchmark Batch Runner")
    print("=" * 100)
    print(f"project_root  : {get_project_root()}")
    print(f"benchmark.py  : {get_benchmark_script_path()}")
    print(f"world_sizes   : {args.world_sizes}")
    print(f"algo          : {args.algo}")
    print(f"data_sizes_mb : {args.data_sizes_mb}")
    print(f"dtype         : {args.dtype}")
    print(f"warmup        : {args.warmup}")
    print(f"repeat        : {args.repeat}")
    print(f"launcher      : {args.launcher}")
    print(f"np_flag       : {args.np_flag}")
    print(f"python        : {args.python}")
    print(f"timeout       : {args.timeout}")
    print(f"skip_check    : {args.skip_check}")
    print(f"output_dir    : {output_dir}")
    print(f"merged_output : {merged_output}")
    print(f"log_output    : {log_output}")
    print(f"total runs    : {len(cases)}")
    print("=" * 100)
    print()

    results = run_all_cases(
        cases=cases,
        launcher=args.launcher,
        np_flag=args.np_flag,
        python_executable=args.python,
        timeout=args.timeout,
        skip_check=args.skip_check,
        verbose=args.verbose,
    )

    print_summary(results)

    merge_result_csvs(
        run_results=results,
        merged_output=merged_output,
    )

    save_run_log(
        run_results=results,
        log_output=log_output,
    )

    failed_count = sum(1 for result in results if not result.passed)

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()