import json

def open_config_file(parameter_name):
    """Input file dir; output a list with header removed and elmt decomposed."""
    filename = 'config/config.json'
    with open(filename) as f_obj:
        config = json.load(f_obj)
        
    return config[parameter_name]