"""
GR00T inference server with an optional decimate+resample(+satfix) wrapper inserted between
Gr00tPolicy and Gr00tSimPolicyWrapper, so the RoboCasa client sees the exact same interface
as gr00t/eval/run_gr00t_server.py but the server internally decimates the action chunk to
every k-th step and reconstructs it via zoh / tac_fold / tac_fold_satfix before returning it.

Usage (mirrors run_gr00t_server.py, adds --resampler/--decimation_k):
  python run_gr00t_satfix_server.py --model-path nvidia/GR00T-N1.6-3B \
      --embodiment-tag ROBOCASA_PANDA_OMRON --use-sim-policy-wrapper \
      --resampler tac_fold_satfix --decimation_k 2

On SIGTERM/SIGINT, dumps DecimationSatfixWrapper.unfixed_saturation (per-key fraction of
elements the unfixed tac_fold reconstruction would have put outside [-1,1], only populated
when resampler=="tac_fold_satfix") before exiting, so the diagnostic survives a `terminate()`
from a driver script instead of being lost with the process.
"""
import signal
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad")

from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import tyro

from gr00t_decimation_wrapper import DecimationSatfixWrapper


@dataclass
class ServerConfig:
    model_path: str
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 5555
    strict: bool = True
    resampler: str = "zoh"  # zoh | tac_fold | tac_fold_satfix
    decimation_k: int = 2


def _dump_diagnostic(inner_policy):
    print("[satfix-server] unfixed_saturation diagnostic (fraction of elements the unfixed "
          "tac_fold reconstruction put outside [-1,1], per action key):")
    if not inner_policy.unfixed_saturation:
        print("  (empty -- resampler wasn't tac_fold_satfix, or no requests were served)")
    for key, fracs in inner_policy.unfixed_saturation.items():
        avg = sum(fracs) / len(fracs)
        print(f"  {key}: n_chunks={len(fracs)} mean_frac={avg:.4f} max_frac={max(fracs):.4f}")


def main(config: ServerConfig):
    print(f"[satfix-server] embodiment={config.embodiment_tag} resampler={config.resampler} k={config.decimation_k}")
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    inner_policy = DecimationSatfixWrapper(
        policy,
        k=config.decimation_k,
        resampler=config.resampler,
        embodiment_tag=config.embodiment_tag.value,
    )
    policy = Gr00tSimPolicyWrapper(inner_policy)

    def _handle_term(signum, frame):
        _dump_diagnostic(inner_policy)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_term)

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        _dump_diagnostic(inner_policy)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
