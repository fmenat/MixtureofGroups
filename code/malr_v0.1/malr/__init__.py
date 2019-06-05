# -*- coding: utf-8 -*-
# Copyright (C) 2011, Filipe Rodrigues <fmpr@dei.uc.pt>

try:
    from malr_version import __version__
except ImportError, e:
    import sys
    print >>sys.stderr, '''\
Could not import submodules (exact error was: %s).

There are many reasons for this error the most common one is that you have
either not built the packages or have built (using `python setup.py build`) or
installed them (using `python setup.py install`) and then proceeded to test
fmprlib **without changing the current directory**.

Try installing and then changing to another directory before importing fmprlib.
''' % e

__all__ = [
    '__version__',
    ]
