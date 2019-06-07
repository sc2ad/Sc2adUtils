import json
import argparse
from jsonschema import validate

def validateJson(js, schema):
    """
    Confirms that a given json follows the given schema.
    Throws a "jsonschema.excpetions.ValidationError" otherwise.
    """
    assert type(js) == dict, "JSON must be a dictionary!"
    assert type(schema) == dict, "Schema must be a dictionary!"
    validate(instance=js, schema=schema)

def loadJson(string):
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        # This is fine, just need to attempt to load as file
        with open(string, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads the given config json file and confirms the config schema")
    parser.add_argument("config", help="The JSON config string or file to validate")
    parser.add_argument("--schema", default="template_lazy.json", help="The JSON schema to follow")

    args = parser.parse_args()
    
    js = loadJson(args.config)
    schema = loadJson(args.schema)

    # Throws an error if the JSON is not in the proper format
    validateJson(js, schema)