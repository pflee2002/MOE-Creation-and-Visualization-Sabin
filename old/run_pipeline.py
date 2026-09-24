import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import paths


STAGES = [
    "01_prepare_parent_links.py",
    "02_create_moe_segments.py",
    "03_calculate_segment_moes.py",
    "04_build_corridors.py",
    "05_visualize_corridors.py",
]


# Run one script and copy its live output to the pipeline log.
def run_script(script_name, log, arguments=None):
    arguments = arguments or []
    heading = f"\n[{script_name}]\n"
    print(heading, end="")
    log.write(heading)
    log.flush()
    process = subprocess.Popen(
        [sys.executable, "-u", str(paths.CODE_DIR / script_name), *arguments],
        cwd=paths.CODE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{script_name} failed with exit code {return_code}.")


# Compile, validate, and run the complete workflow in sequence.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--force-txdot", action="store_true")
    parser.add_argument("--start-stage", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rebuild-partitions", action="store_true")
    args = parser.parse_args()
    paths.check_inputs()
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(paths.CODE_DIR)]
    )
    if compile_result.returncode:
        raise RuntimeError("Compilation failed.")

    log_file = paths.OUTPUT_DIR / "pipeline_run.log"
    started = datetime.now()
    with log_file.open("w", encoding="utf-8") as log:
        print(f"Pipeline started: {started:%Y-%m-%d %H:%M:%S}")
        if not args.skip_checks:
            run_script("check_pipeline.py", log)
        if args.check_only:
            return
        for stage_number, stage in enumerate(STAGES, start=1):
            if stage_number < args.start_stage:
                continue
            stage_arguments = []
            if stage.startswith("01_") and args.force_txdot:
                stage_arguments.append("--force-txdot")
            if stage_number >= 3 and args.workers:
                stage_arguments.extend(["--workers", str(args.workers)])
            if stage_number in (3, 4) and args.rebuild_partitions:
                stage_arguments.append("--rebuild-partitions")
            run_script(stage, log, stage_arguments)
        finished = datetime.now()
        message = (
            f"\nPipeline completed: {finished:%Y-%m-%d %H:%M:%S}\n"
            f"Total runtime: {finished - started}\n"
            f"Log: {log_file}\n"
        )
        print(message, end="")
        log.write(message)


if __name__ == "__main__":
    main()
