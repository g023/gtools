"""
Quick and dirty json string fixing functions

Author: g023 (https://github.com/g023)
License: MIT
"""

import json
import re

class DoubleBraceJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder that handles double braces {{ }} by converting them to single braces
    """
    def decode(self, s, _w=json.decoder.WHITESPACE):
        # Remove double braces before decoding
        s = re.sub(r'{{', '{', s)
        s = re.sub(r'}}', '}', s)
        return super().decode(s)

def fix_json_string(json_string):
    """
    Comprehensive function to fix common JSON formatting issues including double braces
    """
    # Fix double braces
    fixed = re.sub(r'{{', '{', json_string)
    fixed = re.sub(r'}}', '}', fixed)
    
    # Fix potential trailing commas (optional, based on your needs)
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    return fixed

def parse_json_with_auto_fix(json_string, use_custom_decoder=True):
    """
    Attempts to parse JSON with automatic fixing of common issues
    """
    # Try normal parsing first
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            # Try with explicit fix
            fixed_string = fix_json_string(json_string)
            
            if use_custom_decoder:
                # Use custom decoder
                return json.loads(fixed_string, cls=DoubleBraceJSONDecoder)
            else:
                # Standard decoding
                return json.loads(fixed_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON even after fixing: {e}")

# Usage
# Example usage problematic response 
problematic_response = '''{
  "reasoning": "The logical sequence for making a web scraper in Python is: design the scraper, implement it, test it, and then complete it. The end node must be the last node in topological_order and have type 'end'.",
  "dag": {{
    "nodes": [
      {{"id": "design", "label": "Design web scraper", "type": "action"}},
      {{"id": "implement", "label": "Implement web scraper", "type": "action"}},
      {{"id": "test", "label": "Test web scraper", "type": "action"}},
      {{"id": "complete", "label": "Complete", "type": "end"}}
    ],
    "edges": [
      {{"from": "design", "to": "implement", "condition": null, "reason": "Design before implementation"}},
      {{"from": "implement", "to": "test", "condition": null, "reason": "Implementation before testing"}},
      {{"from": "test", "to": "complete", "condition": null, "reason": "Testing before completion"}}
    ],
    "metadata": {{
      "is_acyclic": true,
      "cycle_explanation": "Linear progression",
      "parallel_paths": [],
      "topological_order": ["design", "implement", "test", "complete"]
    }}
  }}
}'''

result = parse_json_with_auto_fix(problematic_response)
print("Parsed result:", result["dag"]["nodes"][0]["id"])  # Should print "design"