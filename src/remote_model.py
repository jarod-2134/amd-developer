def generate_remote(prompt, client, selected_model, category="unknown", stop_sequences=None):
    system_prompts = {
        "Mathematical reasoning": "Provide ONLY the final numerical answer.",
        "Code debugging": "Provide ONLY the corrected code. No explanations, no markdown formatting.",
        "Code generation": "Provide ONLY the generated code. No explanations, no markdown formatting.",
        "Factual knowledge": "Answer the factual question as concisely as possible."
    }
    system_instruction = system_prompts.get(category, "Provide extremely concise answers.")

    if client and selected_model and client.api_key != "dummy":
        try:
            # Setup arguments safely
            kwargs = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "stop": stop_sequences if stop_sequences else ["<eos>"]
            }
            
            # CRITICAL FIX FOR REASONING MODELS:
            # Give them a wide ceiling (1524) so reasoning + output both fit comfortably.
            kwargs["max_tokens"] = 1524 

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