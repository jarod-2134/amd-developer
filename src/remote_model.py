def generate_remote(prompt, client, selected_model, category="unknown", stop_sequences=None):
    # Category-specific, terse system prompts. Keeping completion text short is
    # the primary lever for token efficiency (output tokens dominate cost).
    system_prompts = {
        "Mathematical reasoning": "Provide ONLY the final numerical answer. No working.",
        "Code debugging": "Provide ONLY the corrected code. No explanations, no markdown fences.",
        "Code generation": "Provide ONLY the generated code. No explanations, no markdown fences.",
        "Factual knowledge": "Answer the factual question as concisely as possible.",
        "Sentiment classification": "Respond as 'LABEL: one-sentence justification'. LABEL is Positive, Negative, or Neutral.",
        "Named entity recognition": "List entities as 'TYPE: value', one per line. Types: PERSON, ORG, LOCATION. No prose.",
        "Text summarisation": "Provide ONLY the summary, following any length/format constraint in the task.",
        "Logical / deductive reasoning": "Provide ONLY the final answer. No reasoning trace.",
    }
    system_instruction = system_prompts.get(category, "Provide extremely concise answers.")

    # Per-category completion-token ceiling. Tight caps save output tokens while
    # leaving enough room for code answers (which can be long). Reasoning models
    # consume thinking tokens from the same budget, so reasoning-heavy categories
    # keep a wider ceiling.
    max_tokens_by_category = {
        "Sentiment classification": 60,
        "Factual knowledge": 150,
        "Named entity recognition": 200,
        "Text summarisation": 200,
        "Mathematical reasoning": 120,
        "Logical / deductive reasoning": 300,
        "Code debugging": 1024,
        "Code generation": 1024,
    }
    max_tokens = max_tokens_by_category.get(category, 512)

    if client and selected_model and getattr(client, "api_key", "dummy") != "dummy":
        try:
            # Setup arguments safely
            kwargs = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "stop": stop_sequences if stop_sequences else ["<eos>"],
                "max_tokens": max_tokens,
            }

            response = client.chat.completions.create(**kwargs)

            raw_content = response.choices[0].message.content
            clean_content = raw_content.strip() if raw_content is not None else "[EMPTY RESPONSE]"
            total_tokens = getattr(response.usage, "total_tokens", 0)

            return {
                "content": clean_content,
                "total_tokens": total_tokens
            }
        except Exception as e:
            return {"content": f"[REMOTE ERROR]: {e}", "total_tokens": 0}
    else:
        # Dummy / no-credentials mode: return a placeholder so callers' .get()
        # never crashes. Zero tokens recorded.
        return {"content": f"[DUMMY MODE - no remote call for category '{category}']", "total_tokens": 0}
