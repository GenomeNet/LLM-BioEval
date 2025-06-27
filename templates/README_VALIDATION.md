# Template Validation System

MicrobeLLM now supports a **3-file template system** that provides robust validation and normalization of LLM responses.

## File Structure

For each template, you now have **3 files** instead of 2:

1. **System Template** (`templates/system/templateX.txt`) - The system prompt
2. **User Template** (`templates/user/templateX.txt`) - The user prompt with placeholders
3. **Validation Config** (`templates/validation/templateX.json`) - **NEW!** JSON validation rules

## Example: template3_knowledge

### 1. System Template (`templates/system/template3_knowledge.txt`)
```
"Determine the knowledge level for the binomial strain name..."
```

### 2. User Template (`templates/user/template3_knowledge.txt`)
```
"Respond with a JSON object for {binomial_name} with the knowledge level category in lowercase in this format:

{
    "knowledge_group": "<limited|moderate|extensive|NA>"
}"
```

### 3. Validation Config (`templates/validation/template3_knowledge.json`)
```json
{
  "template_info": {
    "name": "template3_knowledge",
    "type": "knowledge",
    "description": "Knowledge level assessment template with NA support",
    "version": "1.0"
  },
  "expected_response": {
    "format": "json",
    "required_fields": ["knowledge_group"],
    "optional_fields": []
  },
  "field_definitions": {
    "knowledge_group": {
      "type": "string",
      "required": true,
      "description": "Knowledge level category for the organism",
      "allowed_values": ["limited", "moderate", "extensive", "NA"],
      "validation_rules": {
        "case_sensitive": false,
        "trim_whitespace": true,
        "normalize_mapping": {
          "limited": ["limited", "minimal", "basic", "low"],
          "moderate": ["moderate", "medium", "intermediate"],
          "extensive": ["extensive", "comprehensive", "detailed", "high", "full"],
          "NA": ["na", "n/a", "n.a.", "not available", "not applicable", "unknown"]
        }
      },
      "validation_error_messages": {
        "missing": "Required field 'knowledge_group' is missing from response",
        "invalid_value": "Invalid knowledge level. Expected one of: limited, moderate, extensive, NA",
        "wrong_type": "Field 'knowledge_group' must be a string"
      }
    }
  }
}
```

## Benefits

### **Bug Fix: NA Handling**
The original bug where `{"knowledge_group": "NA"}` was incorrectly parsed as `"limited"` is now fixed:
- **Correct**: `"NA"` → `"NA"` 
- **Old behavior**: `"NA"` → `"limited"`

### **Flexible Validation**
Each template can define its own validation rules:
- **Required vs optional fields**
- **Allowed values** (e.g., template1 doesn't allow "NA", template3 does)
- **Normalization mappings** (e.g., "comprehensive" → "extensive")
- **Custom error messages**

### **Better Data Quality**
- Automatic normalization of common variants
- Clear error reporting for invalid responses
- Consistent data formats across different models

### **Easy Maintenance**
- No need to modify Python code to add new validation rules
- JSON configuration is human-readable and version-controlled
- Template-specific validation without code changes

## Validation Config Fields

### `template_info`
Basic metadata about the template:
```json
{
  "name": "template_name",
  "type": "knowledge|phenotype|custom",
  "description": "Human-readable description",
  "version": "1.0"
}
```

### `expected_response`
Defines the expected response structure:
```json
{
  "format": "json",
  "required_fields": ["field1", "field2"],
  "optional_fields": ["field3", "field4"]
}
```

### `field_definitions`
Detailed validation rules for each field:
```json
{
  "field_name": {
    "type": "string|number|boolean",
    "required": true|false,
    "description": "Field description",
    "allowed_values": ["value1", "value2"],
    "validation_rules": {
      "case_sensitive": false,
      "trim_whitespace": true,
      "normalize_mapping": {
        "canonical_value": ["variant1", "variant2"]
      }
    },
    "validation_error_messages": {
      "missing": "Custom error message",
      "invalid_value": "Custom error message",
      "wrong_type": "Custom error message"
    }
  }
}
```

## Creating New Templates

1. **Create the system template** (`templates/system/my_template.txt`)
2. **Create the user template** (`templates/user/my_template.txt`)
3. **Create the validation config** (`templates/validation/my_template.json`)

The system will automatically detect and use the validation config when processing responses.

## Fallback Behavior

If no validation config exists for a template:
- **Knowledge templates**: Uses the fixed `normalize_knowledge_level()` function
- **Phenotype templates**: Uses the original fallback parsing
- **Unknown templates**: Returns response as-is

## Testing Validation

You can test validation configs programmatically:

```python
from microbellm.template_config import validate_template_response_from_file

# Test a response
response_data = {"knowledge_group": "NA"}
validated, errors, validator = validate_template_response_from_file(
    "templates/user/template3_knowledge.txt",
    response_data
)

print(f"Validated: {validated}")  # {'knowledge_group': 'NA'}
print(f"Errors: {errors}")        # []
```

This ensures your validation rules work correctly before using them in production. 