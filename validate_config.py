import json
import argparse
from jsonschema import validate

def validateJson(js, schema):
    """
    Confirms that a given json follows the given schema.
    Throws a "jsonschema.excpetions.ValidationError" otherwise.
    """
    assert type(js) == dict or type(js) == str, "JSON must be a dictionary or string JSON or path!"
    assert type(schema) == dict or type(schema) == str, "Schema must be a dictionary or string JSON or path!"
    validate(instance=loadJson(js), schema=loadJson(schema))

def loadJson(data):
    if type(data) == dict:
        return data
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # This is fine, just need to attempt to load as file
        with open(data, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads the given config json file and confirms the config schema")
    parser.add_argument("config", help="The JSON config string or file to validate")
    parser.add_argument("--schema", default="template_lazy.json", help="The JSON schema to follow or the path to the schema")

    args = parser.parse_args()
    
    # Throws an error if the JSON is not in the proper format
    validateJson(args.config, args.schema)