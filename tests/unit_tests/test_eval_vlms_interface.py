"""
Unit tests for the GRAID VLM evaluation interface.

Tests the interface functions and configuration validation without requiring
actual VLM models or databases.
"""

import pytest

from graid.evaluator.eval_vlms import (
    METRIC_CONFIGS,
    PROMPT_CONFIGS,
    VLM_CONFIGS,
    create_metric,
    create_prompt,
    create_vlm,
    list_available_metrics,
    list_available_prompts,
    list_available_vlms,
    validate_configuration,
)


class TestVLMInterface:
    """Test VLM interface utility functions."""

    def test_list_available_vlms(self):
        """Test listing available VLM types."""
        vlms = list_available_vlms()

        # Should return a dictionary
        assert isinstance(vlms, dict)

        # Should contain expected VLM types
        expected_vlms = ["GPT", "Gemini", "Llama", "Llama_CD", "GPT_CD", "Gemini_CD"]
        for vlm_type in expected_vlms:
            assert vlm_type in vlms

        # GPT should have multiple models
        assert len(vlms["GPT"]) > 1
        assert "gpt-4o" in vlms["GPT"]

        # Llama should have default model
        assert vlms["Llama"] == ["default"]

    def test_list_available_metrics(self):
        """Test listing available evaluation metrics."""
        metrics = list_available_metrics()

        # Should return a list
        assert isinstance(metrics, list)

        # Should contain expected metrics
        expected_metrics = ["ExactMatch", "Contains", "LLMJudge"]
        for metric in expected_metrics:
            assert metric in metrics

    def test_list_available_prompts(self):
        """Test listing available prompt types."""
        prompts = list_available_prompts()

        # Should return a list
        assert isinstance(prompts, list)

        # Should contain expected prompts
        expected_prompts = ["ZeroShotPrompt", "SetOfMarkPrompt", "CoT"]
        for prompt in expected_prompts:
            assert prompt in prompts

    def test_create_metric(self):
        """Test metric creation."""
        # Valid metrics should work
        for metric_type in METRIC_CONFIGS.keys():
            metric = create_metric(metric_type)
            assert metric is not None

        # Invalid metric should raise error
        with pytest.raises(ValueError, match="Unknown metric type"):
            create_metric("InvalidMetric")

    def test_vlm_config_validation(self):
        """Test VLM configuration validation."""
        # GPT requires model name
        config = VLM_CONFIGS["GPT"]
        assert config["requires_model_selection"] is True
        assert len(config["models"]) > 0

        # Llama doesn't require model name
        config = VLM_CONFIGS["Llama"]
        assert config["requires_model_selection"] is False
        assert len(config["models"]) == 0

    def test_prompt_config_validation(self):
        """Test prompt configuration validation."""
        # CoT should be incompatible with ExactMatch
        config = PROMPT_CONFIGS["CoT"]
        assert "ExactMatch" in config.get("incompatible_metrics", [])

        # ZeroShotPrompt should support CD
        config = PROMPT_CONFIGS["ZeroShotPrompt"]
        assert config.get("supports_cd", False) is True

        # SetOfMarkPrompt should require GPU
        config = PROMPT_CONFIGS["SetOfMarkPrompt"]
        assert config.get("requires_gpu", False) is True


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_valid_configurations(self):
        """Test valid configuration combinations."""
        # Standard combinations should be valid
        valid_configs = [
            ("Llama", "LLMJudge", "ZeroShotPrompt"),
            ("GPT", "Contains", "CoT"),
            ("Gemini", "ExactMatch", "ZeroShotPrompt"),
        ]

        for vlm, metric, prompt in valid_configs:
            is_valid, error = validate_configuration(vlm, metric, prompt)
            assert is_valid is True
            assert error is None

    def test_invalid_configurations(self):
        """Test invalid configuration combinations."""
        # CoT with ExactMatch should be invalid
        is_valid, error = validate_configuration("Llama", "ExactMatch", "CoT")
        assert is_valid is False
        assert "incompatible" in error.lower()
        assert "Contains" in error  # Should suggest alternative


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_create_vlm_invalid_type(self):
        """Test VLM creation with invalid type."""
        with pytest.raises(ValueError, match="Unknown VLM type"):
            create_vlm("InvalidVLM")

    def test_create_vlm_missing_model_name(self):
        """Test VLM creation missing required model name."""
        with pytest.raises(ValueError, match="Model name required"):
            create_vlm("GPT")  # GPT requires model name

    def test_create_prompt_invalid_type(self):
        """Test prompt creation with invalid type."""
        with pytest.raises(ValueError, match="Unknown prompt type"):
            create_prompt("InvalidPrompt", "Llama")

    def test_create_prompt_cot_batch_conflict(self):
        """Test CoT prompt with batch processing should fail."""
        with pytest.raises(ValueError, match="CoT does not support batch processing"):
            create_prompt("CoT", "Llama", use_batch=True)


class TestIntegrationReadiness:
    """Test that the interface is ready for integration."""

    def test_all_vlm_configs_complete(self):
        """Test that all VLM configs have required fields."""
        required_fields = ["class", "requires_model_selection", "models", "description"]

        for vlm_type, config in VLM_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"VLM {vlm_type} missing field {field}"

    def test_all_metric_configs_complete(self):
        """Test that all metric configs have required fields."""
        required_fields = ["class", "description"]

        for metric_type, config in METRIC_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Metric {metric_type} missing field {field}"

    def test_all_prompt_configs_complete(self):
        """Test that all prompt configs have required fields."""
        required_fields = ["class", "description"]

        for prompt_type, config in PROMPT_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Prompt {prompt_type} missing field {field}"


if __name__ == "__main__":
    pytest.main([__file__])
