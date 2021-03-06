#!/usr/bin/env python

from scikits.image._build import cython

import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('io', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_files('_plugins/*.ini')

    # This function tries to create C files from the given .pyx files.  If
    # it fails, we build the checked-in .c files.
    cython(['_plugins/_colormixer.pyx', '_plugins/_histograms.pyx'],
           working_path=base_path)
    cython(['_plugins/_scivi2_utils.pyx'],
           working_path=base_path)

    config.add_extension('_plugins._colormixer',
                         sources=['_plugins/_colormixer.c'],
                         include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_plugins._histograms',
                         sources=['_plugins/_histograms.c'],
                         include_dirs=[get_numpy_include_dirs()])

    config.add_extension('_plugins._scivi2_utils',
                         sources=['_plugins/_scivi2_utils.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits.image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Image I/O Routines',
          url = 'http://stefanv.github.com/scikits.image/',
          license = 'Modified BSD',
          **(configuration(top_path='').todict())
          )
