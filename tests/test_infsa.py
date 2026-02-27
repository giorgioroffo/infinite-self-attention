"""
Tests for the InfSA library.

Run with:  pytest InfSA_release/tests/test_infsa.py -v
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infsa.core import pure_infsa_scores, linear_infsa_scores, infsa_attention
from infsa.attention import InfSAAttention
from infsa.convert import convert, replace_attention


# ---------------------------------------------------------------------------
# Core functional tests
# ---------------------------------------------------------------------------

class TestPureInfSAScores:
    def test_output_shape_4d(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        scores = pure_infsa_scores(q, k, rho=0.9)
        assert scores.shape == (2, 8, 16, 16)

    def test_output_shape_3d(self):
        q = torch.randn(16, 10, 64)  # (B*H, N, D)
        k = torch.randn(16, 10, 64)
        scores = pure_infsa_scores(q, k)
        assert scores.shape == (16, 10, 10)

    def test_non_negative(self):
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        scores = pure_infsa_scores(q, k)
        assert (scores >= 0).all()

    def test_rho_scaling(self):
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        s1 = pure_infsa_scores(q, k, rho=0.5)
        s2 = pure_infsa_scores(q, k, rho=1.0)
        # s1 should be approximately half of s2
        ratio = s1.sum() / (s2.sum() + 1e-12)
        assert abs(ratio.item() - 0.5) < 0.01

    def test_gradient_flow(self):
        q = torch.randn(1, 2, 4, 8, requires_grad=True)
        k = torch.randn(1, 2, 4, 8, requires_grad=True)
        scores = pure_infsa_scores(q, k)
        loss = scores.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None


class TestLinearInfSAScores:
    def test_output_shape_4d(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        scores = linear_infsa_scores(q, k)
        assert scores.shape == (2, 8, 16, 1)

    def test_output_shape_3d(self):
        q = torch.randn(16, 10, 64)
        k = torch.randn(16, 10, 64)
        scores = linear_infsa_scores(q, k)
        assert scores.shape == (16, 10, 1)

    def test_non_negative(self):
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        scores = linear_infsa_scores(q, k)
        assert (scores >= 0).all()

    def test_gradient_flow(self):
        q = torch.randn(1, 2, 4, 8, requires_grad=True)
        k = torch.randn(1, 2, 4, 8, requires_grad=True)
        scores = linear_infsa_scores(q, k)
        loss = scores.sum()
        loss.backward()
        assert q.grad is not None


class TestInfSAAttentionFunction:
    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_output_shape(self, variant):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        out = infsa_attention(q, k, v, variant=variant)
        assert out.shape == v.shape

    def test_invalid_variant(self):
        q = k = v = torch.randn(1, 1, 4, 8)
        with pytest.raises(ValueError, match="Unknown InfSA variant"):
            infsa_attention(q, k, v, variant="invalid")

    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_gradient_flow(self, variant):
        q = torch.randn(1, 2, 4, 8, requires_grad=True)
        k = torch.randn(1, 2, 4, 8, requires_grad=True)
        v = torch.randn(1, 2, 4, 8, requires_grad=True)
        out = infsa_attention(q, k, v, variant=variant)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert v.grad is not None


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

class TestInfSAAttentionModule:
    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_forward_batch_first(self, variant):
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant=variant, batch_first=True)
        x = torch.randn(2, 10, 64)
        out, weights = attn(x, x, x)
        assert out.shape == (2, 10, 64)

    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_forward_seq_first(self, variant):
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant=variant, batch_first=False)
        x = torch.randn(10, 2, 64)  # (N, B, E)
        out, weights = attn(x, x, x)
        assert out.shape == (10, 2, 64)

    def test_need_weights(self):
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant="pure_infsa")
        x = torch.randn(2, 10, 64)
        out, weights = attn(x, x, x, need_weights=True)
        assert weights is not None

    def test_rho_property(self):
        attn = InfSAAttention(embed_dim=64, num_heads=4, rho_init=0.8)
        assert abs(attn.rho - 0.8) < 0.01

    def test_rho_trainable(self):
        attn = InfSAAttention(embed_dim=64, num_heads=4, rho_trainable=True)
        trainable_params = [n for n, p in attn.named_parameters() if p.requires_grad]
        assert "rho_logit" in trainable_params

    def test_training_step(self):
        """Test that a full training step works (forward + backward + step)."""
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant="pure_infsa")
        optimizer = torch.optim.Adam(attn.parameters(), lr=1e-3)
        x = torch.randn(2, 10, 64)
        out, _ = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            InfSAAttention(embed_dim=64, num_heads=4, variant="invalid")


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------

class TestConvert:
    def _make_simple_transformer(self):
        """Create a minimal Transformer for testing."""
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        return model

    def test_convert_transformer_encoder(self):
        model = self._make_simple_transformer()
        converted = convert(model, variant="pure_infsa", verbose=False)
        # Check that MHA modules were replaced
        for name, module in converted.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                pytest.fail(f"Found unconverted MHA at {name}")

    def test_convert_preserves_structure(self):
        model = self._make_simple_transformer()
        x = torch.randn(2, 10, 64)
        converted = convert(model, variant="pure_infsa", verbose=False)
        out = converted(x)
        assert out.shape == (2, 10, 64)

    def test_convert_linear_infsa(self):
        model = self._make_simple_transformer()
        x = torch.randn(2, 10, 64)
        converted = convert(model, variant="linear_infsa", verbose=False)
        out = converted(x)
        assert out.shape == (2, 10, 64)

    def test_convert_inplace(self):
        model = self._make_simple_transformer()
        model_id = id(model)
        convert(model, variant="pure_infsa", inplace=True, verbose=False)
        assert id(model) == model_id

    def test_convert_not_inplace(self):
        model = self._make_simple_transformer()
        converted = convert(model, variant="pure_infsa", inplace=False, verbose=False)
        assert id(converted) != id(model)

    def test_replace_single_mha(self):
        mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        new_attn = replace_attention(mha, variant="pure_infsa", copy_weights=True)
        assert isinstance(new_attn, InfSAAttention)
        assert new_attn.embed_dim == 128
        assert new_attn.num_heads == 8

    def test_weight_copy(self):
        mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        new_attn = replace_attention(mha, variant="pure_infsa", copy_weights=True)
        # Weights should have been copied
        E = 64
        q_w_orig = mha.in_proj_weight[:E]
        assert torch.allclose(new_attn.q_proj.weight, q_w_orig)

    def test_include_patterns(self):
        model = self._make_simple_transformer()
        # Only convert layer 0
        converted = convert(
            model, variant="pure_infsa",
            include_patterns=[r"layers\.0\."],
            verbose=False,
        )
        # Layer 0 should be converted, layer 1 should not
        has_infsa_0 = False
        has_mha_1 = False
        for name, module in converted.named_modules():
            if "layers.0" in name and isinstance(module, InfSAAttention):
                has_infsa_0 = True
            if "layers.1" in name and isinstance(module, nn.MultiheadAttention):
                has_mha_1 = True
        assert has_infsa_0
        assert has_mha_1

    def test_exclude_patterns(self):
        model = self._make_simple_transformer()
        converted = convert(
            model, variant="pure_infsa",
            exclude_patterns=[r"layers\.1\."],
            verbose=False,
        )
        has_mha_1 = False
        for name, module in converted.named_modules():
            if "layers.1" in name and isinstance(module, nn.MultiheadAttention):
                has_mha_1 = True
        assert has_mha_1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_end_to_end_training(self, variant):
        """Full training loop with a converted model."""
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        model = convert(model, variant=variant, verbose=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 10, 64)

        for _ in range(3):
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_deterministic(self):
        """Same input should produce same output (eval mode)."""
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant="pure_infsa")
        attn.eval()
        x = torch.randn(2, 10, 64)
        out1, _ = attn(x, x, x)
        out2, _ = attn(x, x, x)
        assert torch.allclose(out1, out2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_cuda(self, variant):
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant=variant).cuda()
        x = torch.randn(2, 10, 64).cuda()
        out, _ = attn(x, x, x)
        assert out.device.type == "cuda"
        assert out.shape == (2, 10, 64)

    @pytest.mark.parametrize("variant", ["pure_infsa", "linear_infsa"])
    def test_mixed_precision(self, variant):
        attn = InfSAAttention(embed_dim=64, num_heads=4, variant=variant)
        x = torch.randn(2, 10, 64)
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out, _ = attn(x, x, x)
        # Output should be produced without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
