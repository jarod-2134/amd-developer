from typing import List, Dict
from pydantic import BaseModel, Field
from pathlib import Path

from local_model import LOCAL_MODEL_ID, fetch_single_profile

class ModelAssignment(BaseModel):
    remote: str = Field(description="Remote Fireworks model ID to use for this category.")
    local: str = Field(description="Local Ollama model ID to use for this category.")
    local_capable: bool = Field(
        default=True,
        description="Whether this category is safe to attempt locally before falling back to remote."
    )

class DynamicRoutingTable(BaseModel):
    factual_knowledge: ModelAssignment = Field(
        alias="Factual knowledge",
        description="Model for explaining concepts, definitions, and how things work."
    )
    mathematical_reasoning: ModelAssignment = Field(
        alias="Mathematical reasoning",
        description="Model for multi-step arithmetic, percentages, word problems, projections."
    )
    sentiment_classification: ModelAssignment = Field(
        alias="Sentiment classification",
        description="Model for labelling sentiment and justifying the classification."
    )
    text_summarisation: ModelAssignment = Field(
        alias="Text summarisation",
        description="Model for condensing passages to a specific format or length constraint."
    )
    named_entity_recognition: ModelAssignment = Field(
        alias="Named entity recognition",
        description="Model for extracting and labelling entities (person, org, location, date)."
    )
    code_debugging: ModelAssignment = Field(
        alias="Code debugging",
        description="Model for identifying bugs in code snippets and providing corrected implementations."
    )
    logical_deductive_reasoning: ModelAssignment = Field(
        alias="Logical / deductive reasoning",
        description="Model for constraint-based puzzles where all conditions must be satisfied."
    )
    code_generation: ModelAssignment = Field(
        alias="Code generation",
        description="Model for writing correct, well-structured functions from a spec."
    )
    stops: Dict[str, List[str]] = Field(
        description="A dictionary mapping every analyzed model name exactly to its list of valid stop sequences."
    )


    # Ensures compatibility whether accessing via dictionary keys or attributes
    model_config = {"populate_by_name": True, "populate_by_alias": True}

def build_routing_table_deterministic(model_ids: List[str], require_serverless: bool = True):
    project_root = Path(__file__).resolve().parent.parent
    metadata_folder = project_root / "metadata"

    profiles = []
    for m_id in model_ids:
        parsed = fetch_single_profile(metadata_folder, m_id)
        if parsed:
            resolved_id, tags, stop_tokens, metrics = parsed

            if require_serverless and not metrics.get("supports_serverless", False):
                print(f"⚠️ Skipping {resolved_id}: Requires On-Demand deployment (Serverless unsupported).")
                continue

            profiles.append({
                "model_id": resolved_id,
                "parameter_count": metrics.get("parameter_count") or 0,
                "is_moe": metrics.get("is_moe", False),
                "supports_tools": metrics.get("supports_tools", False),
                "is_coding": "coding" in tags or "code" in resolved_id.lower(),
                "stop_sequences": stop_tokens,
            })

    if not profiles:
        print("Error: No models available matching the specified serverless requirement constraints.")
        return None

    low_intensity = min(profiles, key=lambda x: x["parameter_count"])["model_id"]
    high_compute = max(profiles, key=lambda x: x["parameter_count"])["model_id"]

    tool_capable = [p for p in profiles if p["supports_tools"]]
    structured_token = min(tool_capable, key=lambda x: x["parameter_count"])["model_id"] if tool_capable else low_intensity

    coding_models = [p for p in profiles if p["is_coding"]]
    domain_specialized = coding_models[0]["model_id"] if coding_models else high_compute

    moe_models = [p for p in profiles if p["is_moe"]]
    if moe_models:
        parametric_knowledge = moe_models[0]["model_id"]
    else:
        sorted_by_size = sorted(profiles, key=lambda x: x["parameter_count"])
        parametric_knowledge = sorted_by_size[len(sorted_by_size) // 2]["model_id"]

    # Every category now carries BOTH a remote and a local target.
    # "local_capable" gates whether a per-task difficulty heuristic is even
    # allowed to route here locally -- some categories are too failure-prone
    # on a 3B model to risk the accuracy gate just to save tokens.
    routing_table = {
        "sentiment_classification": {
            "remote": low_intensity, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "text_summarisation": {
            "remote": low_intensity, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "named_entity_recognition": {
            "remote": structured_token, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "factual_knowledge": {
            "remote": parametric_knowledge, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "code_generation": {
            "remote": domain_specialized, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "code_debugging": {
            "remote": domain_specialized, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "mathematical_reasoning": {
            "remote": high_compute, "local": LOCAL_MODEL_ID, "local_capable": True,
        },
        "logical_deductive_reasoning": {
            "remote": high_compute, "local": LOCAL_MODEL_ID, "local_capable": False,
        },
        "stops": {p["model_id"]: p["stop_sequences"] for p in profiles},
    }

    return routing_table