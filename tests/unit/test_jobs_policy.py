"""The command policy (web console plan, J1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from almasim.jobs import Resources
from almasim.jobs.policy import KNOWN_COMMANDS, PolicyError, has_option, plan_job
from almasim.jobs.profile import CommandPolicy, Profile

pytestmark = pytest.mark.unit

ZWICKY = Profile(
    partition="zwickyA-CPU",
    account="ska",
    wsclean="/opt/wsclean",
    casa_data=Path("/data/casa"),
    commands={"image image-from-ms": CommandPolicy(cpus=4, memory="16G", time="2-00:00:00")},
)


def test_network_commands_run_on_the_server_and_the_rest_on_slurm():
    assert plan_job(["metadata", "--band", "3"], ZWICKY).executor == "server"
    assert plan_job(["products", "resolve", "--member-ous-uid", "x"], ZWICKY).executor == "server"
    assert plan_job(["image", "clean-quality", "/out"], ZWICKY).executor == "slurm"


def test_a_profile_can_move_a_command():
    profile = Profile(commands={"metadata": CommandPolicy(executor="slurm")})
    assert plan_job(["metadata"], profile).executor == "slurm"


def test_imaging_becomes_a_slurm_driver_with_the_profiles_tools():
    plan = plan_job(["image", "image-from-ms", "p.csv", "out", "--cube"], ZWICKY)
    assert plan.command == "image image-from-ms"
    assert plan.argv == [
        "image", "image-from-ms", "p.csv", "out", "--cube",
        "--backend", "slurm",
        "--slurm-queue", "zwickyA-CPU",
        "--slurm-project", "ska",
        "--casa-data-root", "/data/casa",
        "--wsclean-bin", "/opt/wsclean",
    ]  # fmt: skip
    assert plan.resources == Resources(
        partition="zwickyA-CPU", account="ska", cpus=4, memory="16G", time="2-00:00:00"
    )


def test_options_the_caller_gave_are_kept():
    argv = [
        "image", "image-from-ms", "p.csv", "out",
        "--backend=sync", "--wsclean-bin", "/my/wsclean", "--casa-data-root", "/mine",
    ]  # fmt: skip
    assert plan_job(argv, ZWICKY).argv == argv


def test_calibration_on_slurm_does_not_download_casa_data():
    plan = plan_job(["visibilities", "calibrate", "--input-root", "/in"], ZWICKY)
    assert "--skip-casa-data-update" in plan.argv
    assert plan.argv[plan.argv.index("--backend") + 1] == "slurm"


def test_a_download_fans_out_only_for_its_post_processing():
    plain = plan_job(["products", "download", "--products-csv", "p.csv"], ZWICKY)
    assert plain.executor == "server"
    assert "--backend" not in plain.argv
    assert "--skip-casa-data-update" not in plain.argv  # the server node has the internet

    unpack = plan_job(["products", "download", "--products-csv", "p.csv", "--unpack-ms"], ZWICKY)
    assert unpack.argv[unpack.argv.index("--backend") + 1] == "slurm"
    assert "--skip-casa-data-update" not in unpack.argv


def test_predict_gets_the_partition_and_clean_the_wsclean_binary():
    assert plan_job(["predict", "ms.ms"], ZWICKY).argv[-2:] == ["--slurm-queue", "zwickyA-CPU"]
    assert plan_job(["clean", "-size", "512", "512", "ms.ms"], ZWICKY).argv[-2:] == [
        "--wsclean-bin",
        "/opt/wsclean",
    ]


def test_extras_go_before_a_double_dash():
    plan = plan_job(["clean", "--", "-name", "x", "ms.ms"], ZWICKY)
    assert plan.argv == ["clean", "--wsclean-bin", "/opt/wsclean", "--", "-name", "x", "ms.ms"]


def test_the_profiles_extra_args_come_last_and_overrides_win():
    profile = Profile(commands={"simulation": CommandPolicy(args=("--verbose",), cpus=16)})
    plan = plan_job(["simulation", "--metadata-path", "m.csv"], profile, {"cpus": 32})
    assert plan.argv[-1] == "--verbose"
    assert plan.resources.cpus == 32


def test_unknown_commands_are_refused():
    with pytest.raises(PolicyError, match="unknown command"):
        plan_job(["rm", "-rf"], ZWICKY)
    with pytest.raises(PolicyError):
        plan_job([], ZWICKY)
    with pytest.raises(PolicyError):
        plan_job(["--help"], ZWICKY)


def test_has_option():
    assert has_option(["--backend=slurm"], "--backend")
    assert has_option(["x", "--postprocess-backend", "sync"], "--backend", "--postprocess-backend")
    assert not has_option(["--backends"], "--backend")


def test_every_known_command_exists_in_the_cli():
    """The policy's command list follows the CLI (docs/cli.md is generated from it)."""
    reference = (Path(__file__).resolve().parents[2] / "docs" / "cli.md").read_text()
    for command in KNOWN_COMMANDS:
        assert f"`almasim {command}`" in reference, command


def test_option_value():
    from almasim.jobs.policy import option_value

    assert option_value(["--backend=sync"], "--backend") == "sync"
    assert (
        option_value(["x", "--postprocess-backend", "slurm"], "--backend", "--postprocess-backend")
        == "slurm"
    )
    assert option_value(["--backend"], "--backend") is None
    assert option_value([], "--backend") is None
