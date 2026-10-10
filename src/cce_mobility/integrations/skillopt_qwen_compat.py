from __future__ import annotations


def install_qwen38_explicit_no_thinking(qwen_backend) -> bool:
    """Make SkillOpt's disabled-thinking setting explicit for Qwen3.8-Max."""
    post = qwen_backend._post_chat_completion
    if getattr(post, "_cce_qwen38_no_thinking", False):
        return False

    def post_with_explicit_no_thinking(payload, timeout, config):
        if (
            payload.get("model") == "qwen3.8-max"
            and config.enable_thinking is False
            and "enable_thinking" not in payload
            and "chat_template_kwargs" not in payload
        ):
            payload = {**payload, "enable_thinking": False}
        return post(payload, timeout, config)

    post_with_explicit_no_thinking._cce_qwen38_no_thinking = True
    qwen_backend._post_chat_completion = post_with_explicit_no_thinking
    return True
