#!/usr/bin/env bash
# Adopt the candidate baselines of a golden:supersede PR (MADIV-28 T0.18).
#
#   scripts/regression/adopt_candidate_baseline.sh RUN_ID VERSION "REASON"
#
# RUN_ID is the golden-ci workflow run of the PR's head commit (it uploaded the
# golden-candidate-baseline and golden-candidate-simulation artifacts);
# VERSION is the next baseline, e.g. v2. The script downloads both
# candidates, checks their SHA-256, publishes them as release
# golden-baseline-VERSION and points ci/golden/baseline.json at them. Commit
# that change to the PR: its next CI run must then pass against the new
# baseline. Record the baseline in the plan's Baselines register.
set -euo pipefail
RUN_ID="${1:?usage: adopt_candidate_baseline.sh RUN_ID VERSION REASON}"
VERSION="${2:?baseline version, e.g. v2}"
REASON="${3:?reason, e.g. 'SIM-3: integration time in seconds (#300)'}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
TAG="golden-baseline-$VERSION"
gh release view "$TAG" -R "$REPO" > /dev/null 2>&1 && { echo "release $TAG already exists" >&2; exit 1; }

WORK="$(mktemp -d)"
gh run download "$RUN_ID" -R "$REPO" -n golden-candidate-baseline -D "$WORK/tier1"
gh run download "$RUN_ID" -R "$REPO" -n golden-candidate-simulation -D "$WORK/sim"
(cd "$WORK/tier1" && sha256sum -c golden-baseline-candidate.tar.gz.sha256)
(cd "$WORK/sim" && sha256sum -c golden-simulation-candidate.tar.gz.sha256)
cp "$WORK/tier1/golden-baseline-candidate.tar.gz" "$WORK/$TAG.tar.gz"
cp "$WORK/sim/golden-simulation-candidate.tar.gz" "$WORK/golden-simulation-$VERSION.tar.gz"
(cd "$WORK" && sha256sum "$TAG.tar.gz" "golden-simulation-$VERSION.tar.gz" > SHA256SUMS)
commit="$(gh run view "$RUN_ID" -R "$REPO" --json headSha -q .headSha)"

gh release create "$TAG" -R "$REPO" --target main --prerelease --latest=false \
    --title "CI golden baselines $VERSION" \
    --notes "Golden baselines $VERSION for the ALMASim CI regression gate (MADIV-28), from workflow run $RUN_ID at commit $commit. Reason: $REASON. Test data, not a software release." \
    "$WORK/$TAG.tar.gz" "$WORK/golden-simulation-$VERSION.tar.gz" "$WORK/SHA256SUMS"

python3 - "$REPO_ROOT/ci/golden/baseline.json" "$REPO" "$TAG" "$VERSION" "$WORK/SHA256SUMS" "$RUN_ID" "$commit" "$REASON" <<'PY'
import json, sys
path, repo, tag, version, sums, run_id, commit, reason = sys.argv[1:]
digests = dict(reversed(line.split()) for line in open(sums))
base = f"https://github.com/{repo}/releases/download/{tag}/"
pointers = json.load(open(path))
source = f"workflow run {run_id}, commit {commit[:8]}; {reason}"
pointers["baseline"] = {"url": base + f"{tag}.tar.gz", "sha256": digests[f"{tag}.tar.gz"],
                        "version": version, "source": source}
name = f"golden-simulation-{version}.tar.gz"
pointers["simulation_baseline"] = {"url": base + name, "sha256": digests[name],
                                   "version": version, "source": source}
open(path, "w").write(json.dumps(pointers, indent=2) + "\n")
PY
echo "Published $TAG and updated ci/golden/baseline.json; commit it to the PR."
