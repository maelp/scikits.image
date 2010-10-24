#!/usr/bin/env python

from scikits.image._build import cython
import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('transform', parent_package, top_path)
    config.add_data_dir('tests')

    # This function tries to create C files from the given .pyx files.  If
    # it fails, try to build with pre-generated .c files.
    cython(['zoom.pyx'], working_path=base_path)

    config.add_extension('zoom', sources=['zoom.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits.image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Transform Algorithms',
          url = 'http://stefanv.github.com/scikits.image/',
          license = 'Modified BSD',
          **(configuration(top_path='').todict())
          )
