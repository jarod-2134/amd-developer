def generate_remote(prompt, client, selected_model, category="unknown", stop_sequences=None):
    # Category-specific, terse system prompts. Keeping completion text short is
    # the primary lever for token efficiency (output tokens dominate cost).
    system_prompts = {
        "Mathematical reasoning": "Respond ONLY with the final numerical answer inside the JSON. Do not include steps, calculations, or math formatting.",
        "Code debugging": "Provide ONLY the functional, corrected code string. Do not include explanations or markdown syntax like ```.",
        "Code generation": (
            "You must output ONLY the raw Python code string inside the JSON field. "
            "Do not write out your thought process, draft code, or considerations. "
            "Begin your JSON string value immediately with the code definition or imports."
        ),
        "Factual knowledge": "Provide a direct, hyper-concise answer to the question. Strip out all prefix repetition and conversational filler.",
        "Sentiment classification": "Classify sentiment as Positive, Negative, or Neutral, followed by a one-sentence justification. Format exactly as 'LABEL: <justification>'.",
        "Named entity recognition": "Extract all unique named entities. List them strictly as 'TYPE: value' separated by newlines. Types: PERSON, ORG, LOCATION.",
        "Text summarisation": "Provide ONLY the direct summary text following the length/format constraints. Strip any 'Here is a summary' introduction.",
        "Logical / deductive reasoning": (
            "Solve the logic puzzle step by step internally. Then, populate the final answer field "
            "with just the explicit concluding name, number, or word. Do not include your working "
            "inside the final JSON answer value string."
        )
    }
    system_instruction = system_prompts.get(category, "Provide extremely concise answers.")

    # Per-category completion-token ceiling. Tight caps save output tokens while
    # leaving enough room for code answers (which can be long). Reasoning models
    # consume thinking tokens from the same budget, so reasoning-heavy categories
    # keep a wider ceiling.
    max_tokens_by_category = {
        "Sentiment classification": 200,
        "Factual knowledge": 150,
        "Named entity recognition": 200,
        "Text summarisation": 200,
        "Mathematical reasoning": 200,
        "Logical / deductive reasoning": 1024,
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
