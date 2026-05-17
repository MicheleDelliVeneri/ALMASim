import json
import subprocess
import sys


def get_changed_lines(file_path, base_ref="origin/main"):
    try:
        # Get the diff of the file compared to base_ref
        diff_output = subprocess.check_output(
            ["git", "diff", f"{base_ref}...HEAD", "--unified=0", file_path],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error getting diff for {file_path}: {e.output}")
        return set()

    changed_lines = set()
    for line in diff_output.splitlines():
        if line.startswith("@@"):
            # Format: @@ -line,count +line,count @@
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith("+"):
                spec = parts[2][1:].split(",")
                start_line = int(spec[0])
                count = int(spec[1]) if len(spec) > 1 else 1
                for i in range(start_line, start_line + count):
                    changed_lines.add(i)
    return changed_lines


def main():
    try:
        with open("cov_target2.json") as f:
            coverage_data = json.load(f)
    except FileNotFoundError:
        print("cov_target2.json not found")
        return

    files_to_check = [
        "src/almasim/cli_products.py",
        "src/almasim/cli_simulation.py",
        "src/almasim/services/compute/slurm.py",
        "src/almasim/services/archive/calibrate_ms.py",
        "tests/unit/test_scheduling_cluster.py",
        "src/almasim/cli_image.py",
        "src/almasim/cli_metadata.py",
        "src/almasim/cli.py",
    ]

    for file_path in files_to_check:
        print(f"\nChecking: {file_path}")
        changed_lines = get_changed_lines(file_path)
        if not changed_lines:
            print("  No changed lines found in diff.")
            continue

        # coverage-report json format: files[file_path].missing_lines
        file_cov = coverage_data.get("files", {}).get(file_path, {})
        missing_lines = set(file_cov.get("missing_lines", []))

        missing_changed = sorted(list(changed_lines.intersection(missing_lines)))

        if missing_changed:
            print(f"  Missing coverage on changed lines: {missing_changed}")
        else:
            print("  All changed lines are covered.")


if __name__ == "__main__":
    main()
