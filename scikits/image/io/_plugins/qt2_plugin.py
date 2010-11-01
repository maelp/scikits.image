from util import prepare_for_display, window_manager, GuiLockError
import numpy as np
import sys

# We try to aquire the gui lock first or else the gui import might
# trample another GUI's PyOS_InputHook.
window_manager.acquire('qt2')

try:
    from PyQt4.QtGui import (QApplication, QMainWindow, QImage, QPixmap,
                             QLabel, QWidget)
    from PyQt4 import QtCore, QtGui
    from scivi2 import _simple_imshow, _advanced_imshow

except ImportError:
    window_manager._release('qt2')

    raise ImportError("""\
    PyQt4 libraries not installed. Please refer to

    http://www.riverbankcomputing.co.uk/software/pyqt/intro

    for more information.  PyQt4 is GPL licensed.  For an
    LGPL equivalent, see

    http://www.pyside.org
    """)

app = None

def imshow(im, flip=None, fancy=False):
    global app
    if not app:
        app = QApplication([])

    if not fancy:
        iw = _simple_imshow(im, flip=flip, mgr=window_manager)
    else:
        iw = _advanced_imshow(im, flip=flip, mgr=window_manager)

    iw.show()

def _app_show():
    global app
    if app and window_manager.has_windows():
        app.exec_()
    else:
        print 'No images to show.  See `imshow`.'


def imsave(filename, img):
    # we can add support for other than 3D uint8 here...
    img = prepare_for_display(img)
    qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], QImage.Format_RGB888)
    saved = qimg.save(filename)
    if not saved:
        from textwrap import dedent
        msg = dedent(
            '''The image was not saved. Allowable file formats
            for the QT imsave plugin are:
            BMP, JPG, JPEG, PNG, PPM, TIFF, XBM, XPM''')
        raise RuntimeError(msg)
