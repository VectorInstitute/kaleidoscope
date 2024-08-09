"""Tests for shell command executors."""

from web.auto_scaling.executors.remote_ssh_executor import _SSHRemoteSpecs


def test_ssh_get_prefix_args():
    """Test generating ssh command prefix using remote specs util."""
    remote_specs = _SSHRemoteSpecs("user", "hostname", 30383)
    remote_cli_args = remote_specs.get_args_prefix()
    assert remote_cli_args == "ssh -p 30383 user@hostname".split(" ")
