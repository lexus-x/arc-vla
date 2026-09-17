"""Phase-1 smoke test for the Flow-SDE port (flow_sde.py).

Checks (minimum bar for "the RL loss is computable"):
1. Loads the existing flow8300_s0 SFT checkpoint via stock lerobot SmolVLAPolicy.
2. Builds a small batch of dummy LIBERO-Spatial-shaped observations via the
   checkpoint's own pre-processor pipeline.
3. Runs sample_actions_flow_sde in stochastic mode -> checks action shape,
   finite log-prob.
4. Calls .backward() on the log-prob sum -> checks gradients reach model params.
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline

from flow_sde import sample_actions_flow_sde

CKPT = "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300"
DEVICE = "cuda"


def main():
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(DEVICE)
    policy.train()  # need grads for the backward() check

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json"
    )

    bsize = 2
    dummy_obs = {
        "observation.images.image": torch.rand(bsize, 3, 256, 256),
        "observation.images.wrist_image": torch.rand(bsize, 3, 256, 256),
        "observation.state": torch.randn(bsize, 8),
        "task": ["pick up the black bowl and place it on the plate"] * bsize,
    }

    batch = preprocessor(dummy_obs)
    print("Preprocessed batch keys:", list(batch.keys()))

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]

    print("images[0] shape:", images[0].shape, "state shape:", state.shape)
    print("lang_tokens shape:", lang_tokens.shape, "lang_masks shape:", lang_masks.shape)

    result = sample_actions_flow_sde(
        policy, images, img_masks, lang_tokens, lang_masks, state,
        noise_level=0.1, stochastic=True,
    )

    actions = result["actions"]
    log_prob = result["log_prob"]
    chosen_idx = result["chosen_idx"]

    print("CHECK actions.shape:", tuple(actions.shape))
    print("CHECK chosen_idx:", chosen_idx, "/", result["num_steps"])
    print("CHECK log_prob finite:", torch.isfinite(log_prob).all().item())
    print("CHECK log_prob shape:", tuple(log_prob.shape))
    print("CHECK log_prob requires_grad:", log_prob.requires_grad)

    loss = -log_prob.sum()
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in policy.parameters()
        if p.grad is not None and p.requires_grad
    ]
    n_params_with_grad = sum(1 for p in policy.parameters() if p.grad is not None)
    n_params_total = sum(1 for _ in policy.parameters())

    print(f"CHECK params_with_grad: {n_params_with_grad} / {n_params_total}")
    print(f"CHECK max grad norm: {max(grad_norms) if grad_norms else 0.0}")
    print(f"CHECK any nonzero grad: {any(g > 0 for g in grad_norms)}")

    assert torch.isfinite(log_prob).all(), "log_prob has NaN/Inf"
    assert n_params_with_grad > 0, "no gradients reached any parameter"
    assert any(g > 0 for g in grad_norms), "all gradients are zero"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
