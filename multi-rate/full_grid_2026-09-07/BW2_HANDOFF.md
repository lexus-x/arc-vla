# ARC-TAC BW2 handoff

The campaign archive contains the code, preregistration, cached demonstrations, and the
declared Push-T, RoboMimic, and RoboCasa inputs. On `blackwell2-0000`:

```bash
mkdir -p ~/arc_tacfold_campaign
cd ~/arc_tacfold_campaign
tailscale file get .
tar -xzf arc_tacfold_bw2_bundle_20260918_v5.tar.gz
bash run_arc_tacfold_bw2.sh
```

Do not edit the preregistration after launching. The runner validates the environments and
inputs, executes the fixed campaign, applies the locked claim gates, and creates
`arc_tacfold_bw2_results.tgz`. Preserve the full log even if a gate fails.
