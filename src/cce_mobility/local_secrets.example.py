# Copy to local_secrets.py, which is ignored by Git. Never commit actual keys.
# Agent and world model use the first key; the independent judge uses the second.
# Put the key issued by each role's configured service here. Fill only these two.
# If all roles use the same account/service, both values may be the same.
CCE_MODEL_API_KEY = "replace-with-your-model-token"
CCE_JUDGE_API_KEY = "replace-with-your-judge-token"
