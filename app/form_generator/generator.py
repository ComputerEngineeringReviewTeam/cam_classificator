from app.form_generator.html_elements import FormTag, InputTag
from werkzeug.datastructures import ImmutableMultiDict
import copy
import json


CONFIG_FORM_KEY = 'form'
CONFIG_FIELDS_KEY = 'fields'
# keys in each "field" object
CONFIG_FIELD_ATTRIBUTES_KEY = 'attributes'
CONFIG_FIELD_NAME_KEY = 'field_name'
CONFIG_FIELD_TYPE_KEY = 'data_type'

HTML_ATTR_NAME = 'name'
HTML_ATTR_VALUE = 'value'

CONFIG_DATATYPES = {'str': str, 'int': int, 'float': float}


def build_attributes(field: dict[str, str | dict[str, str]]) -> dict[str, str] | None:
    """
    Translate config "field" object into dict[str, str] of HTML attributes
    Specifically:
    1. entire field["attributes"] dict is copied
    2. "field_name" is translated into the "name" attribute
    :param field: a dictionary describing single field
    :return: a dictionary of HTML attributes in the format "name": "value"
        or None if the field is missing vital attributes, such as "field_name"
    """
    attr = copy.deepcopy(field[CONFIG_FIELD_ATTRIBUTES_KEY])
    try:
        attr[HTML_ATTR_NAME] = field[CONFIG_FIELD_NAME_KEY]
        # attr['value'] = field['default_value']
        return attr
    except KeyError:
        return None


def create_form(config: dict[str, any]) -> FormTag:
    """
    Creates FormTag with an InputTag for each field of the form and additional <submit> tag from given config
    :param config: parsed JSON config
    :return: FormTag with the contents of one InputTag for each field of the form
    """
    form_tag = FormTag(attributes=config[CONFIG_FORM_KEY][CONFIG_FIELD_ATTRIBUTES_KEY])
    input_tags = [InputTag(attributes=attr) for attr in map(build_attributes, config[CONFIG_FIELDS_KEY]) if attr]
    input_tags.append(InputTag(attributes={'type': 'submit', 'method': 'POST', 'action': '/handle_post'}))
    form_tag.content = input_tags

    return form_tag


def update_types(form: ImmutableMultiDict[str, str] | dict[str, str], config: dict[str, dict]) -> dict[str, any]:
    """
    For each field in the form casts its value to the appropriate type and returns the new dict.
    If any field fails the conversion (eg. has no "data_type" attribute) it will be omitted
    :param form: form returned from the request, as dictionary os "field name": "field value"
    :param config: parsed JSON config
    :return: dictionary of "field name": value (of appropriate type)
    """
    def unpack_field_type(field: dict[str, str | dict]) -> (str | None, type | None):
        """
        Unpacks field dict into tuple of (field_name, field_type) using the CONFIG_DATATYPES
        :param field: dictionary describing the field
        :return: tuple of (field name, field type)
            or (None, None) if the field is missing - "field_name" or "data"type" attributes
        """
        try:
            return field[CONFIG_FIELD_NAME_KEY], CONFIG_DATATYPES[field[CONFIG_FIELD_TYPE_KEY]]
        except KeyError:
            return None, None
    fields_with_types = map(unpack_field_type, config[CONFIG_FIELDS_KEY])
    return {fname: form.get(fname, type=ftype) for fname, ftype in fields_with_types if fname and ftype}


def load_config(filepath: str) -> dict[str, any]:
    """
    Loads the config from JSON file
    :param filepath: path to JSON file containing the config file
    :return: parsed JSON config
    """
    with open(filepath, "r") as f:
        return json.load(f)

# with open('form.json', 'r') as f:
#     form_template = json.load(f)
#
# # print("JSON\n", form_template)
# # print()
# # print("HTML\n", read_form_from_config(form_template))
# from werkzeug.datastructures import ImmutableMultiDict
#
# dic = ImmutableMultiDict([("username", "xantos"), ("num", "12.75"), ("num2", "123")])
# print()
# for k, v in dic.items():
#     print(k, v, type(v))
# res = interpret_form(dic, form_template)
# print()
# for k, v in res.items():
#     print(k, v, type(v))




