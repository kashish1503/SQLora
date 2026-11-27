import copy

import pytest
import torch
import torch.nn as nn

from lora_sql.lora import (
    LoRALinear,
    count_parameters,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
)


class TinyBlock(nn.Module):
    """Mimics the naming of a transformer block so target matching is realistic."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, 2 * d)
        self.down_proj = nn.Linear(2 * d, d)

    def forward(self, x):
        h = self.q_proj(x) + self.v_proj(x)
        return self.down_proj(torch.relu(self.up_proj(h)))


class TinyModel(nn.Module):
    def __init__(self, d: int = 32, n: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyBlock(d) for _ in range(n))
        self.lm_head = nn.Linear(d, 10)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.lm_head(x)


def _randomize_B(model: nn.Module) -> None:
    """B starts at zero; give it values so the adapter actually changes outputs."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.02)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_zero_init_is_identity():
    model = TinyModel()
    x = torch.randn(4, 32)
    expected = model(x)
    inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8)
    torch.testing.assert_close(model(x), expected)


def test_injection_targets_and_param_count():
    model = TinyModel(d=32, n=2)
    replaced = inject_lora(model, ["q_proj", "v_proj"], r=4)
    assert sorted(replaced) == sorted(
        [f"layers.{i}.{name}" for i in range(2) for name in ("q_proj", "v_proj")]
    )
    assert not isinstance(model.lm_head, LoRALinear)
    mark_only_lora_trainable(model)
    trainable, _ = count_parameters(model)
    # Each adapted layer adds r * (in + out) params.
    assert trainable == 4 * (4 * (32 + 32))


def test_no_match_raises():
    with pytest.raises(ValueError):
        inject_lora(TinyModel(), ["does_not_exist"])


def test_only_lora_gets_gradients():
    model = TinyModel()
    inject_lora(model, ["q_proj", "up_proj"], r=4)
    mark_only_lora_trainable(model)
    _randomize_B(model)
    model(torch.randn(8, 32)).sum().backward()
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name


def test_merge_unmerge_roundtrip():
    model = TinyModel()
    inject_lora(model, ["q_proj", "v_proj", "down_proj"], r=8, alpha=16)
    _randomize_B(model)
    model.eval()
    x = torch.randn(4, 32)
    adapted = model(x)

    layer = model.layers[0].q_proj
    original_w = layer.base.weight.clone()
    layer.merge()
    layer.unmerge()
    torch.testing.assert_close(layer.base.weight, original_w)

    merged = merge_lora(copy.deepcopy(model), unwrap=True)
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    torch.testing.assert_close(merged(x), adapted, rtol=1e-5, atol=1e-5)


def test_rslora_scaling():
    layer = LoRALinear(nn.Linear(16, 16), r=16, alpha=32, use_rslora=True)
    assert layer.scaling == pytest.approx(32 / 4)


def test_state_dict_roundtrip():
    model = TinyModel()
    inject_lora(model, ["q_proj", "v_proj"], r=4)
    _randomize_B(model)
    sd = lora_state_dict(model)
    assert sd and all("lora_" in k for k in sd)

    fresh = copy.deepcopy(model)
    with torch.no_grad():
        for m in fresh.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.zero_()
                m.lora_B.zero_()
    load_lora_state_dict(fresh, sd)
    x = torch.randn(4, 32)
    torch.testing.assert_close(fresh(x), model(x))


def test_matches_hf_peft():
    """Same A/B weights => same outputs as the reference implementation."""
    peft = pytest.importorskip("peft")

    base = TinyModel().eval()
    ours = copy.deepcopy(base)
    inject_lora(ours, ["q_proj", "v_proj"], r=8, alpha=16)
    _randomize_B(ours)

    ref = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    with torch.no_grad():
        for name, module in ours.named_modules():
            if isinstance(module, LoRALinear):
                peft_layer = ref.get_submodule(f"base_model.model.{name}")
                peft_layer.lora_A["default"].weight.copy_(module.lora_A)
                peft_layer.lora_B["default"].weight.copy_(module.lora_B)

    x = torch.randn(4, 32)
    torch.testing.assert_close(ours(x), ref(x))
    mark_only_lora_trainable(ours)
    trainable_ours, _ = count_parameters(ours)
    trainable_ref, _ = ref.get_nb_trainable_parameters()
    assert trainable_ours == trainable_ref
