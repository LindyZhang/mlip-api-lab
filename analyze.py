import json
import os
from typing import Any, Dict
from litellm import completion

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"


def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Set it as an env variable.")
    os.environ["GROQ_API_KEY"] = api_key
    
    system_prompt = (
    	"Return only valid JSON (no markdown, no extra text) with exactly these keys:\n"
	"{\n"
	'    "destination": string,\n'
	'    "ideal_visit_times": string[],\n'
	'    "price_range": string,\n'
	'    "top_attractions": string[]\n'
	"}\n"
    )
    user_prompt = f"Create the JSON travel itinerary for destination: {destination}"
    
    resp = completion(
        model=MODEL,
	messages=[{"content": system_prompt, "role": "system"},
		  {"content": user_prompt, "role": "user"},
	],
    )
    
    text = resp["choices"][0]["message"]["content"].strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"LLM did not return valid JSON. Got: \n{text}")
    
    required_keys = {"destination", "price_range", "ideal_visit_times", "top_attractions"}
    
    if set(data.keys()) != required_keys:
        extra = set(data.keys()) - required_keys
        missing = required_keys - set(data.keys())
        raise RuntimeError(f"Schema keys mismatch. Missing={missing}, Extra={extra}")
    
    if not isinstance(data["destination"], str):
        raise RuntimeError("destination must be a string")
    if not isinstance(data["price_range"], str):
        raise RuntimeError("price_range must be a string")
    if not isinstance(data["ideal_visit_times"], list) or not all(isinstance(x, str) for x in data["ideal_visit_times"]):
        raise RuntimeError("ideal_visit_times must be a list of strings")
    if not isinstance(data["top_attractions"], list) or not all(isinstance(x, str) for x in data["top_attractions"]):
        raise RuntimeError("top_attractions must be a list of strings")

    return data
