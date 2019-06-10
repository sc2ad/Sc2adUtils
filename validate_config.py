import json
import argparse
try:
    # python -m pip install jsonschema
    from jsonschema import validate
except:
    raise ImportError("You must install json schema via pip in order to use this validator!")
# python -m pip install pyyaml
try:
    import yaml
    YAML = True
except ImportError:
    YAML = False

MEMOIZE = False

def validateJson(js, schema):
    """
    Confirms that a given json follows the given schema.
    Throws a "jsonschema.exceptions.ValidationError" otherwise.
    """
    assert type(js) == dict or type(js) == str, "JSON must be a dictionary or string JSON or path!"
    assert type(schema) == dict or type(schema) == str, "Schema must be a dictionary or string JSON or path!"
    validate(instance=loadJson(js), schema=loadJson(schema))

def validateYaml(yaml, schema):
    """
    Confirms that a given yaml follows the given schema.
    The schema can be a json string/dict.
    Throws a "jsonschema.exceptions.ValidationError" otherwise.
    """
    assert YAML, "Must install PyYAML before attempting to load yaml files!"
    assert type(yaml) == dict or type(yaml) == str, "YAML must be a dictionary or path to a YAML file!"
    assert type(schema) == dict or type(schema) == str, "Schema must be a dictionary or string JSON or path to schema.json!"
    validate(instance=loadYaml(yaml), schema=loadJson(schema))

def fixTuples(data):
    for key in data.keys():
        if type(data[key]) == tuple:
            data[key] = list(data[key])
        if type(data[key]) == dict:
            data[key] = fixTuples(data[key])
    return data

def isJson(data):
    return data.endswith("}") or data.endswith(".json")

def validateData(data, schema):
    if isJson(data):
        # JSON
        validateJson(data, schema)
    else:
        # YAML
        validateYaml(data, schema)

def loadAndValidateJson(js, schema):
    """
    Loads the JSON config from the given js and schema strings, dicts, or paths
    And returns the JSON if valid. Otherwise, it will throw a "jsonschema.exceptions.ValidationError".
    """
    validateJson(js, schema)
    return loadJson(js)

def loadAndValidateYaml(yaml, schema):
    """
    Loads the YAML config from the given yaml and schema strings, dicts, or paths
    And returns the YAML if valid. Otherwise, it will throw a "jsonschema.exceptions.ValidationError".
    """
    validateYaml(yaml, schema)
    return loadYaml(yaml)

def load(data, schema):
    """
    Loads the given data and validates it according to the schema provided.
    Data must be either JSON or YAML, it must be a dictionary, a path, or a string of JSON.
    Schema must be JSON, it must be a dictionary, a path, or a string of JSON.
    """
    if isJson(data):
        return loadAndValidateJson(data)
    return loadAndValidateYaml(data)

def memoize(f):
    m = {}
    def helper(input):
        if not MEMOIZE:
            return f(input)
        if input not in m:
            m[input] = f(input)
        return m[input]
    return helper

@memoize
def loadJson(data):
    if type(data) == dict:
        return data
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # This is fine, just need to attempt to load as file
        with open(data, 'r') as f:
            return json.load(f)

@memoize
def loadYaml(data):
    if type(data) == dict:
        return data
    with open(data, 'r') as stream:
        return fixTuples(yaml.load(stream, Loader=yaml.FullLoader))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads the given config json file and confirms the config schema")
    parser.add_argument("config", help="The JSON or YAML config string or file to validate")
    parser.add_argument("--schema", default="template_schema.json", help="The JSON schema to follow or the path to the schema")

    args = parser.parse_args()
    
    # Throws an error if the JSON is not in the proper format
    validateData(args.config, args.schema)