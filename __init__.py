# This file does not get installed and is merely a convenience so that
# everything acts the same if the whole directory is placed anywhere in sys.path
#
# This makes it so that import <name>.<module> works
# This is unnecessary for single-file packages
#
# This works by duplicating the contents of the __init__.py
# in the __name__ directory below this one...

import os
__path__[-1] = os.path.join(__path__[-1],__name__)
TEMP_INIT_FILE = os.path.join(__path__[-1],'__init__.py')
del os

with open(TEMP_INIT_FILE) as TEMP_HANDLE:
    TEMP_CODE = compile(TEMP_HANDLE.read(), TEMP_INIT_FILE, 'exec')
    exec(TEMP_CODE)

del TEMP_HANDLE, TEMP_CODE, TEMP_INIT_FILE
