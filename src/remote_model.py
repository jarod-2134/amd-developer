def generate_remote(prompt, client, selected_model, category="unknown", stop_sequences=None):
    """
    Hit the Fireworks API and return a dictionary containing both the text and token metrics.
    """
    system_prompts = {
        "mathematical_reasoning": "You are a math solver. Provide ONLY the final numerical answer. Do not show work.",
        "code_debugging": "Provide ONLY the corrected code. No explanations, no markdown formatting.",
        "code_generation": "Provide ONLY the generated code. No explanations, no markdown formatting.",
        "factual_knowledge": "Answer the factual question as concisely as possible, ideally in a single sentence or word."
    }
    system_instruction = system_prompts.get(category, "You are a helpful assistant. Provide extremely concise answers.")

    if client and selected_model and client.api_key != "dummy":
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                stop=stop_sequences if stop_sequences else ["<eos>"]
            )

            # Extract content and token metadata safely
            return {
                "content": response.choices[0].message.content.strip(),
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        except Exception as e:
            print(f"API call failed: {e}")
            return {"content": f"[REMOTE ERROR] Failed to call API: {e}", "total_tokens": 0}
    else:
        return {"content": f"[REMOTE PLACEHOLDER] Answer for: {prompt[:30]}...", "total_tokens": 0}
