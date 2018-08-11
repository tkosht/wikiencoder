# -*- coding: utf-8 -*-

from configobj import ConfigObj, flatten_errors
from validate import Validator


default_config = "config/project.cfg"
default_spec = "config/project.spc"

class Config(object):
    def __init__(self, config_file=default_config, config_spec=default_spec):
        config = ConfigObj(config_file, configspec=config_spec, file_error=True)
        validator = Validator()
        validated = config.validate(validator, preserve_errors=True)
        if not isinstance(validated, bool) or not validated:
            errors = []
            for (section_list, key, err_info) in flatten_errors(config, validated):
                _s = '/'.join(section_list)
                if key is None:
                    err = f'The following section was missing "{_s}": {err_info}'
                else:
                    err = f'The "{key}" key in the section "{_s}" failed validation: {err_info}'
                errors.append(err)
            raise Exception(*errors)
        self.config = config

    def get(self, *keys):
        v = self.config
        for k in keys:
            v = v[k]
        return v

    def setup(self, keys):
        for k in keys:
            v = self.get(*k)
            setattr(self, k[-1], v)
