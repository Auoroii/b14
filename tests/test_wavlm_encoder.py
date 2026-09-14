"""Offline CPU tests for the dependency-injected WavLM encoder."""

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor
from transformers import WavLMConfig, WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from emotion_model.common import apply_query_mask
from emotion_model.speech import WavLMEncoder

pytestmark = pytest.mark.filterwarnings(
    "ignore:Support for mismatched key_padding_mask and attn_mask is deprecated.*:UserWarning"
)


def _make_tiny_wavlm() -> WavLMModel:
    """Create a random local 12-layer WavLM without pretrained weights."""
    torch.manual_seed(7)
    config = WavLMConfig(
        hidden_size=4,
        num_hidden_layers=12,
        num_attention_heads=1,
        intermediate_size=8,
        hidden_dropout=0.2,
        activation_dropout=0.2,
        attention_dropout=0.2,
        feat_proj_dropout=0.2,
        layerdrop=0.0,
        apply_spec_augment=False,
        feat_extract_norm="layer",
        conv_dim=(2,),
        conv_stride=(2,),
        conv_kernel=(3,),
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=1,
        num_buckets=4,
        max_bucket_distance=8,
    )
    return WavLMModel(config)


def _right_padded_batch(*, padding_value: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return waveform [2, 7] and valid-prefix mask [2, 7]."""
    waveform = torch.tensor(
        [
            [0.1, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3],
            [0.5, -0.4, 0.25, padding_value, padding_value, padding_value, padding_value],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, False, False, False, False],
        ]
    )
    return waveform, valid_mask


def test_tiny_wavlm_and_wrapper_extract_exact_transformer_layers() -> None:
    """Exclude embedding H0 and return 12 finite, consistently shaped layers."""
    model = _make_tiny_wavlm()
    assert model.config.num_hidden_layers == 12
    encoder = WavLMEncoder(model, freeze_wavlm=True)
    waveform, valid_mask = _right_padded_batch()
    original_waveform = waveform.clone()
    original_mask = valid_mask.clone()
    sanitized_waveform = torch.where(valid_mask, waveform, torch.zeros_like(waveform))

    with torch.no_grad():
        raw_output = model(
            input_values=sanitized_waveform,
            attention_mask=valid_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    output = encoder(waveform, valid_mask)

    assert raw_output.hidden_states is not None
    assert len(raw_output.hidden_states) == 13
    assert len(output.hidden_states) == 12
    assert output.last_hidden_state is output.hidden_states[-1]
    assert output.feature_attention_mask.dtype == torch.bool
    assert output.feature_attention_mask.shape == (2, 3)
    assert torch.equal(
        output.feature_attention_mask,
        torch.tensor([[True, True, True], [True, False, False]]),
    )
    assert torch.equal(output.feature_attention_mask.sum(dim=1), torch.tensor([3, 1]))

    expected_first = apply_query_mask(
        raw_output.hidden_states[1],
        output.feature_attention_mask,
    )
    expected_last = apply_query_mask(
        raw_output.hidden_states[-1],
        output.feature_attention_mask,
    )
    masked_embedding = apply_query_mask(
        raw_output.hidden_states[0],
        output.feature_attention_mask,
    )
    assert output.hidden_states[0] is not raw_output.hidden_states[0]
    assert not torch.equal(output.hidden_states[0], masked_embedding)
    torch.testing.assert_close(output.hidden_states[0], expected_first)
    torch.testing.assert_close(output.hidden_states[-1], expected_last)
    for layer_index, wrapper_layer in enumerate(output.hidden_states, start=1):
        expected_layer = apply_query_mask(
            raw_output.hidden_states[layer_index],
            output.feature_attention_mask,
        )
        torch.testing.assert_close(wrapper_layer, expected_layer)

    for hidden_state in output.hidden_states:
        assert hidden_state.shape == (2, 3, 4)
        assert hidden_state.device == waveform.device
        assert torch.isfinite(hidden_state).all()
        assert torch.equal(hidden_state[1, 1:], torch.zeros((2, 4)))
    assert torch.equal(waveform, original_waveform)
    assert torch.equal(valid_mask, original_mask)


def test_padding_values_are_sanitized_before_wavlm() -> None:
    """Make arbitrary non-finite padding unable to affect valid WavLM frames."""
    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(model, freeze_wavlm=True)
    baseline_waveform, valid_mask = _right_padded_batch()
    changed_waveform, _ = _right_padded_batch()
    changed_waveform[1, 3:] = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 1.0e30]
    )
    original_changed = changed_waveform.clone()

    baseline = encoder(baseline_waveform, valid_mask)
    changed = encoder(changed_waveform, valid_mask)

    assert torch.equal(changed.feature_attention_mask, baseline.feature_attention_mask)
    for baseline_layer, changed_layer in zip(
        baseline.hidden_states,
        changed.hidden_states,
        strict=True,
    ):
        torch.testing.assert_close(changed_layer, baseline_layer)
    torch.testing.assert_close(changed_waveform, original_changed, equal_nan=True)


@pytest.mark.parametrize(
    ("waveform", "valid_mask", "expected_exception", "message"),
    [
        (
            torch.ones((1, 7)),
            torch.tensor([[True, False, True, False, False, False, False]]),
            ValueError,
            "internal",
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[False, False]]),
            ValueError,
            "fully padded",
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[False, True]]),
            ValueError,
            "internal",
        ),
        (
            torch.ones((1, 2)),
            torch.ones((1, 2), dtype=torch.bool),
            ValueError,
            "too short",
        ),
        (
            torch.ones((1, 7)),
            torch.ones((1, 7)),
            TypeError,
            "torch.bool",
        ),
        (
            torch.ones((1, 7)),
            torch.ones((1, 6), dtype=torch.bool),
            ValueError,
            "shape",
        ),
        (
            torch.ones((1, 7), dtype=torch.long),
            torch.ones((1, 7), dtype=torch.bool),
            TypeError,
            "floating point",
        ),
    ],
)
def test_forward_rejects_invalid_waveform_and_mask_inputs(
    waveform: Tensor,
    valid_mask: Tensor,
    expected_exception: type[Exception],
    message: str,
) -> None:
    """Reject mask holes, full padding, short inputs, dtype, and shape errors."""
    encoder = WavLMEncoder(_make_tiny_wavlm())

    with pytest.raises(expected_exception, match=message):
        encoder(waveform, valid_mask)


def test_forward_rejects_nonfinite_valid_waveform_values() -> None:
    """Reject NaN or Inf when the corresponding waveform mask is valid."""
    encoder = WavLMEncoder(_make_tiny_wavlm())
    waveform = torch.tensor([[0.0, 0.1, float("nan"), 0.2, 0.3, 0.4, 0.5]])
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    with pytest.raises(ValueError, match="must be finite"):
        encoder(waveform, valid_mask)


@pytest.mark.parametrize(
    ("sample_rate", "expected_exception"),
    [
        (0, ValueError),
        (-16000, ValueError),
        (8000, ValueError),
        (16000.0, TypeError),
    ],
)
def test_forward_rejects_invalid_or_mismatched_sample_rate(
    sample_rate: object,
    expected_exception: type[Exception],
) -> None:
    """Reject non-positive, mismatched, and non-integer sample rates."""
    encoder = WavLMEncoder(_make_tiny_wavlm())
    waveform = torch.ones((1, 7))
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    with pytest.raises(expected_exception, match="sample_rate"):
        encoder(waveform, valid_mask, sample_rate=cast(int, sample_rate))


def test_constructor_rejects_layer_count_mismatch() -> None:
    """Fail early when the injected 12-layer model is expected to have 11 layers."""
    model = _make_tiny_wavlm()

    with pytest.raises(ValueError, match="layer count"):
        WavLMEncoder(model, expected_num_hidden_layers=11)


def test_constructor_never_calls_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use dependency injection without pretrained lookup or network access."""
    model = _make_tiny_wavlm()

    def fail_from_pretrained(*args: object, **kwargs: object) -> None:
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(WavLMModel, "from_pretrained", fail_from_pretrained)
    encoder = WavLMEncoder(model)

    assert encoder.model is model


def test_forward_uses_explicit_hidden_state_flags_and_valid_mask_polarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request structured hidden states and pass the True-valid mask unchanged."""
    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(model)
    original_forward: Callable[..., Wav2Vec2BaseModelOutput] = model.forward
    captured_kwargs: dict[str, object] = {}

    def recording_forward(*args: object, **kwargs: object) -> Wav2Vec2BaseModelOutput:
        captured_kwargs.update(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    waveform, valid_mask = _right_padded_batch()
    encoder(waveform, valid_mask)

    assert captured_kwargs["output_hidden_states"] is True
    assert captured_kwargs["return_dict"] is True
    assert captured_kwargs["attention_mask"] is valid_mask
    assert torch.equal(cast(Tensor, captured_kwargs["attention_mask"]), valid_mask)


def test_forward_rejects_abnormal_hidden_state_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a backend response missing one of its 13 expected states."""
    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(model)
    original_forward: Callable[..., Wav2Vec2BaseModelOutput] = model.forward

    def truncated_forward(*args: object, **kwargs: object) -> Wav2Vec2BaseModelOutput:
        output = original_forward(*args, **kwargs)
        assert output.hidden_states is not None
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            extract_features=output.extract_features,
            hidden_states=output.hidden_states[:-1],
            attentions=output.attentions,
        )

    monkeypatch.setattr(model, "forward", truncated_forward)
    waveform = torch.ones((1, 7))
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    with pytest.raises(RuntimeError, match="13 total"):
        encoder(waveform, valid_mask)


def test_forward_rejects_abnormal_hidden_state_feature_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject backend states whose feature dimension differs from the config."""
    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(model)
    original_forward: Callable[..., Wav2Vec2BaseModelOutput] = model.forward

    def wrong_width_forward(*args: object, **kwargs: object) -> Wav2Vec2BaseModelOutput:
        output = original_forward(*args, **kwargs)
        assert output.hidden_states is not None
        wrong_width_states = tuple(state[..., :3] for state in output.hidden_states)
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=wrong_width_states[-1],
            extract_features=output.extract_features,
            hidden_states=wrong_width_states,
            attentions=output.attentions,
        )

    monkeypatch.setattr(model, "forward", wrong_width_forward)
    waveform = torch.ones((1, 7))
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    with pytest.raises(RuntimeError, match=r"expected 4, received 3"):
        encoder(waveform, valid_mask)


def test_exact_minimum_convolution_length_and_one_less_are_handled_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept the kernel-sized minimum and reject one fewer point before forward."""
    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(model)
    original_forward: Callable[..., Wav2Vec2BaseModelOutput] = model.forward
    forward_called = False

    def recording_forward(*args: object, **kwargs: object) -> Wav2Vec2BaseModelOutput:
        nonlocal forward_called
        forward_called = True
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    minimum_waveform = torch.ones((1, 3))
    minimum_mask = torch.ones((1, 3), dtype=torch.bool)

    accepted = encoder(minimum_waveform, minimum_mask)

    assert forward_called
    assert accepted.feature_attention_mask.shape == (1, 1)
    assert torch.equal(accepted.feature_attention_mask, torch.tensor([[True]]))

    forward_called = False
    too_short_waveform = torch.ones((1, 2))
    too_short_mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"too short.*\[2\]"):
        encoder(too_short_waveform, too_short_mask)
    assert not forward_called


def test_frozen_wavlm_stays_eval_and_is_deterministic_in_wrapper_train_mode() -> None:
    """Keep frozen parameters/eval behavior and return deterministic features."""
    encoder = WavLMEncoder(_make_tiny_wavlm(), freeze_wavlm=True)
    waveform = torch.linspace(-0.5, 0.5, 7).unsqueeze(0)
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    returned = encoder.train()
    first = encoder(waveform, valid_mask)
    second = encoder(waveform, valid_mask)

    assert returned is encoder
    assert encoder.training
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert all(not hidden_state.requires_grad for hidden_state in first.hidden_states)
    for first_layer, second_layer in zip(
        first.hidden_states,
        second.hidden_states,
        strict=True,
    ):
        assert torch.equal(first_layer, second_layer)


@pytest.mark.parametrize(
    ("unfreeze_last_n_layers", "expected_indices"),
    [
        (0, ()),
        (2, (10, 11)),
        (4, (8, 9, 10, 11)),
    ],
)
def test_only_requested_final_transformer_layers_are_trainable(
    unfreeze_last_n_layers: int,
    expected_indices: tuple[int, ...],
) -> None:
    """Freeze all non-layer modules and select exactly the requested final layers."""

    model = _make_tiny_wavlm()
    encoder = WavLMEncoder(
        model,
        freeze_wavlm=unfreeze_last_n_layers == 0,
        unfreeze_last_n_layers=unfreeze_last_n_layers,
    )

    assert encoder.fully_frozen is (unfreeze_last_n_layers == 0)
    assert encoder.trainable_transformer_layer_indices == expected_indices
    assert all(
        not parameter.requires_grad
        for parameter in model.feature_extractor.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.feature_projection.parameters()
    )
    for layer_index, layer in enumerate(model.encoder.layers):
        assert all(
            parameter.requires_grad is (layer_index in expected_indices)
            for parameter in layer.parameters()
        )
    allowed_prefixes = tuple(
        f"encoder.layers.{layer_index}." for layer_index in expected_indices
    )
    assert all(
        name.startswith(allowed_prefixes)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_partial_unfreezing_keeps_frozen_checkpoint_state_compatible() -> None:
    """Strictly load a fully frozen encoder state into a partially trainable one."""

    frozen = WavLMEncoder(
        _make_tiny_wavlm(),
        freeze_wavlm=True,
        unfreeze_last_n_layers=0,
    )
    partial = WavLMEncoder(
        _make_tiny_wavlm(),
        freeze_wavlm=False,
        unfreeze_last_n_layers=2,
    )

    incompatible = partial.load_state_dict(frozen.state_dict(), strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert partial.trainable_transformer_layer_indices == (10, 11)


@pytest.mark.parametrize(
    ("freeze_wavlm", "unfreeze_last_n_layers", "message"),
    [
        (True, 2, "true exactly"),
        (False, 0, "true exactly"),
        (False, -1, r"\[0, 12\]"),
        (False, 13, r"\[0, 12\]"),
    ],
)
def test_constructor_rejects_inconsistent_or_out_of_range_unfreezing(
    freeze_wavlm: bool,
    unfreeze_last_n_layers: int,
    message: str,
) -> None:
    """Reject contradictory freeze flags and layer counts outside H1--H12."""

    with pytest.raises(ValueError, match=message):
        WavLMEncoder(
            _make_tiny_wavlm(),
            freeze_wavlm=freeze_wavlm,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )


@pytest.mark.parametrize("freeze_wavlm", [True, False])
def test_wrapper_eval_mode_propagates_to_wavlm(freeze_wavlm: bool) -> None:
    """Make eval mode disable training behavior for either freeze setting."""
    encoder = WavLMEncoder(
        _make_tiny_wavlm(),
        freeze_wavlm=freeze_wavlm,
        unfreeze_last_n_layers=0 if freeze_wavlm else 2,
    )
    encoder.train()

    returned = encoder.eval()

    assert returned is encoder
    assert not encoder.training
    assert not encoder.model.training


def test_partial_wavlm_propagates_train_mode_and_final_layer_gradients() -> None:
    """Train only the final two Transformer layers with finite gradients."""
    model = _make_tiny_wavlm()
    model.config.layerdrop = 1.0
    encoder = WavLMEncoder(
        model,
        freeze_wavlm=False,
        unfreeze_last_n_layers=2,
    )
    waveform = torch.linspace(-0.5, 0.5, 7).unsqueeze(0)
    valid_mask = torch.ones((1, 7), dtype=torch.bool)

    returned = encoder.train()
    output = encoder(waveform, valid_mask)
    output.last_hidden_state.square().mean().backward()

    assert returned is encoder
    assert encoder.training
    assert encoder.model.training
    assert encoder.model.config.layerdrop == 0.0
    assert encoder.model.encoder.config.layerdrop == 0.0
    assert len(output.hidden_states) == 12
    assert output.last_hidden_state.requires_grad
    for layer_index, layer in enumerate(encoder.model.encoder.layers):
        gradients = [
            parameter.grad
            for parameter in layer.parameters()
            if parameter.grad is not None
        ]
        if layer_index in (10, 11):
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
        else:
            assert not gradients
    assert all(
        parameter.grad is None
        for parameter in encoder.model.feature_extractor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in encoder.model.feature_projection.parameters()
    )
