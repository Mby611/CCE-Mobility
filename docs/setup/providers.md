# Model Services and Local Keys

Use official accounts and your own credentials. Copy `src/cce_mobility/local_secrets.example.py` to `local_secrets.py` in the same package and fill the named key fields. The file is gitignored. Environment variables remain supported and take precedence. Do not put literal credentials in experiment JSON or YAML.

## Which Keys Are Required?

The default template contains two role-based names, not one required key per model vendor:

| Local field | Used by | Value |
| --- | --- | --- |
| `CCE_MODEL_API_KEY` | Agent and world model | Token issued by their configured model service |
| `CCE_JUDGE_API_KEY` | Judge | Token issued by the configured judging service |

If the Agent/WM use Qwen and the Judge uses DeepSeek, put a Qwen token in the first field and a DeepSeek token in the second. If all roles use the same service/account, the two fields may contain the same token. A new model on the same service does not necessarily require a new key; verify that the account can access that model.

`api_key_env` is a **key-name lookup**, despite the historical option name: it reads the corresponding environment variable first, then the attribute in `local_secrets.py`. You do not need to export either key. Names such as `OPENAI_API_KEY`, `QWEN_API_KEY`, `DEEPSEEK_API_KEY`, `GLM_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` are optional custom aliases, not extra required credentials. To use one, explicitly add that attribute to your local file and point the role's `api_key_env` to it. The default examples consistently use the two role-based fields.

Existing local configuration should use the current `CCE_MODEL_API_KEY` and `CCE_JUDGE_API_KEY` names. Local files and saved outputs are deliberately not automatically rewritten. Create a new output directory when adopting a renamed protocol; do not bypass an earlier run's hash locks.

Each model file has exactly three objects: `agent`, `world_model`, `judge`. Agent and WM must share provider, endpoint, and model; temperatures may differ. Save customized files in `configs/local/`. Do not edit source files after a formal run has begun.

| Service | Key documentation | Protocol | Official endpoint |
| --- | --- | --- | --- |
| OpenAI | [Quickstart](https://developers.openai.com/api/docs/quickstart) | Responses, or Chat Completions for compatible models | https://api.openai.com/v1 |
| Qwen | [Model Studio keys](https://help.aliyun.com/zh/model-studio/get-api-key) | OpenAI-compatible Chat Completions | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| DeepSeek | [API documentation](https://api-docs.deepseek.com/en/) | OpenAI-compatible Chat Completions | https://api.deepseek.com |
| GLM | [Quickstart](https://docs.bigmodel.cn/cn/guide/start/quick-start) | OpenAI-compatible Chat Completions | https://open.bigmodel.cn/api/paas/v4 |
| Claude | [Authentication](https://platform.claude.com/docs/en/manage-claude/authentication) | Anthropic Messages | https://api.anthropic.com/v1/messages |
| Gemini | [API key](https://ai.google.dev/gemini-api/docs/api-key) | Official OpenAI-compatible interface in this implementation | https://generativelanguage.googleapis.com/v1beta/openai |

Select the endpoint for your account region. Qwen's table entry is the mainland China endpoint; other regions use different domains. Endpoint and model availability are not established by a config example. Names printed in the paper may not be publicly available now; choose an explicitly available model, document the change, and do not claim the original model snapshot.

## Chat Completions

An individual role configuration:
```json
{
  "model": "your-available-model",
  "provider": "openai_compatible",
  "base_url": "https://api.deepseek.com",
  "api_key_env": "CCE_MODEL_API_KEY",
  "temperature": 0.0,
  "max_tokens": 8192,
  "extra": {"api_timeout_seconds": 180, "api_max_retries": 3}
}
```
Replace the placeholder before running; preparation rejects it. Use the same acting model for the agent and world-model roles. Configure Judge separately with its own key-name field.

## OpenAI Responses

```json
{
  "model": "your-available-responses-model",
  "provider": "responses",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "CCE_MODEL_API_KEY",
  "max_tokens": 8192,
  "extra": {"omit_temperature": true, "api_timeout_seconds": 180}
}
```
Responses reasoning settings belong in `extra.reasoning`, for example `{"effort":"low"}`, only when the selected model supports it. The backend uses `max_output_tokens`; it is not the Chat Completions payload.

## Anthropic Messages

```json
{
  "model": "your-available-claude-model",
  "provider": "anthropic",
  "chat_url": "https://api.anthropic.com/v1/messages",
  "api_key_env": "CCE_MODEL_API_KEY",
  "temperature": 0.0,
  "max_tokens": 8192,
  "extra": {"anthropic_auth_style": "x-api-key", "anthropic_version": "2023-06-01"}
}
```
Messages uses the `x-api-key` and `anthropic-version` headers. Do not send a Chat Completions payload to this endpoint. Add thinking parameters only if supported by the chosen model.

## Qwen and Gemini Thinking

For a Qwen model supporting the switch, set `extra.enable_thinking=false` explicitly. No universal "reasoning off" switch is applied to all providers. Gemini's [OpenAI compatibility documentation](https://ai.google.dev/gemini-api/docs/openai) describes supported `reasoning_effort` values and model-specific limitations; some models cannot disable reasoning. Do not combine contradictory reasoning options.

`cce check-models configs/local/models.json` makes one real request per role and reports model/transport metadata. Generated experiment configs preserve all request settings. An HTTP error, empty output, or unavailable model is a failed check, not permission to substitute another model.

MB's official Planner–Worker–Reporter runner currently requires an OpenAI-compatible **acting** endpoint. Its Judge can use the other supported protocols. Native Anthropic/Responses acting endpoints work in TP; using them for MB requires a separately implemented upstream provider adapter.

中文要点：默认只填 Agent/WM 与 Judge 两个角色 Key，不需要开通所有服务商。字段名由 `api_key_env` 指定，可以直接读取本地文件，不要求导出环境变量。协议、URL、模型和推理设置在本地模型 JSON 中显式指定。配置模板不是服务验收结果；先运行连通性检查，再进行付费 smoke。
