import json
import subprocess


def get_changed_lines():
    cmd = ["git", "diff", "origin/main...HEAD", "--unified=0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    changed = {}
    current_file = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            changed[current_file] = set()
        elif line.startswith("@@"):
            parts = line.split()
            # Format: @@ -start,len +start,len @@
            new_part = parts[2]
            if "," in new_part:
                start, length = map(int, new_part[1:].split(","))
            else:
                start, length = int(new_part[1:]), 1
            for i in range(start, start + length):
                changed[current_file].add(i)
    return changed


def main():
    try:
        with open("cov_target2.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Coverage report cov_target2.json not found.")
        return

    changed_lines = get_changed_lines()
    files_to_check = [
        "src/almasim/cli_products.py",
        "src/almasim/cli_simulation.py",
        "src/almasim/services/compute/slurm.py",
        "src/almasim/services/archive/calibrate_ms.py",
        "src/almasim/cli_image.py",
        "src/almasim/cli_metadata.py",
        "src/almasim/cli.py",
        "tests/unit/test_scheduling_cluster.py",
    ]

    for file_path in files_to_check:
        if file_path not in changed_lines:
            continue

        # Coverage JSON uses absolute paths or relative paths.
        # We need to find the matching entry.
        cov_info = None
        for key, value in data["files"].items():
            if key.endswith(file_path):
                cov_info = value
                break

        if not cov_info:
            print(f"No coverage data for {file_path}")
            continue

        missing_lines = set(cov_info["missing_lines"])
        missing_changed = sorted(list(missing_lines.intersection(changed_lines[file_path])))

        if missing_changed:
            print(f"Missing changed lines in {file_path}: {missing_changed}")
        else:
            print(f"All changed lines in {file_path} are covered.")


if __name__ == "__main__":
    main()
