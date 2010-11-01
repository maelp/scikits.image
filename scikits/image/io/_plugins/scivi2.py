#! /usr/bin/env python
# encoding: utf-8

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import numpy as np

from scikits.image.transform import zoom
from scikits.image.io._plugins import _scivi2_utils as utils

# TODO
# HIGH PRIORITY
# - force translations to be an integer number*zoom, thus we do not have subpixel
#   translations when zooming in or out (this can be annoying when zooming out, often
#   we will zoom back to 1.0X with a +0.5 translation)
# - we should have fit zoom, zoom 1.0, reinit zoom,
# - multicore if available (use _plugins.util)
# - the application should have an F10 shortcut to display zoom/histo/etc controls
#
# LOW PRIORITY
# - find some better way to tell the ImageRenderer to reset its state when
#   the rendering changes (we change the zoom, the rendering of level-lines, etc)
#   rather than setting image_renderer.state = None
# - check that the subimages that we extract before we zoom are the minimal
#   subimages that should be extracted for each zoom
# - reduce copying of image by using the ndarray C interface directly in the
#   zoom code (eg. we don't necessarily have to ensure the image is contiguous)
# - we should have rectangle zoom (eg. ctrl+click makes it possible to define a
#   rectangle, and the zoom/panning is computed to fit this rectangle in the view)
# - the application should have an F11 shortcut to display a python interpreter
#   with numpy and scikits.image already loaded
# - we should perhaps cache the zoomed image that was last rendered, if we want to
#   have fast color rescaling (eg. change hue/contrast, but do not recompute zoom
#   each time)
# - allow non-uniform zoom (this is easy, we only have to add a zoom_x and zoom_y
#   parameter in the C code, those values are already used inside the code for
#   zooming, we should adapt all python code with scale_x and scale_y rather than
#   scale, but it isn't obvious whether this will really have an use)

# Suggested application design
#
#   ImageViewer: the control that acts on the view, handles 
#     - it should have viewChanged etc signals
#   BasicImageViewer: handles mouse for basic operations (zoom in, zoom out, 1:1,
#   zoom fit, fit rectangle in view), and keys if requested (+, - and directions etc)
#   Application: has some controls that can interact with the ImageViewer
#     - key press can have an action on the ImageViewer, etc
#     - handles flip, etc
#     - histograms, color modes, levels etc
#   - Extending the Viewer should be very easy (either include it in a QGraphicsView,
#     or have some way to easily extend it to permit to input points, etc)

# FEATURES:
#  - any amount of zoom, any subpixel translation (WARNING: when checking if
#  the scale of the zoom is 1.0, as well as when checking for integer
#  translations at a certain scale, we have a floating-point comparison at
#  precision ImageRenderer.EPS, and therefore the zoom/translation has a precision,
#  which should be way below what we request in practice, so this should not be
#  an issue)
#  - only recompute the parts of the images that change
#  - when the image is rendered at scale 1.0 with an integer translation,
#    we do not do any zoom computation
#  - gray/color images
#  - integer/real images
#  - image flip
#  - level lines (one level, several lines, upper/lower sets, range)
#  - configurable ImageRenderer

# Utility function for zoom windows
def _extract_subwindow_with_border(im, x, y, w, h, scale, border=1):
    # If we need p border pixels, we should have the subimage
    # [ ceil(x)-p, floor(x+(w-1)/scale)+p ]
    x1 = max(0, np.ceil(x)-border)
    y1 = max(0, np.ceil(y)-border)
    x2 = min(im.shape[1]-1.0, np.floor(x+(w-1)/scale)+border)
    y2 = min(im.shape[0]-1.0, np.floor(y+(h-1)/scale)+border)
    return (im[y1:y2+1,x1:x2+1], x-x1, y-y1)

class NearestZoom(object):
    def __str__(self):
        return "NearestZoom"

    def subwindow(self, im, x, y, w, h, scale):
        """
        Extract the minimal subwindow of im required to compute the
        requested (x, y, w, h, scale) zoom
        """
        x1 = max(0, np.ceil(x-0.5)-1)
        y1 = max(0, np.ceil(y-0.5)-1)
        x2 = min(im.shape[1]-1.0, np.ceil(x+(w-1)/scale-0.5)+1)
        y2 = min(im.shape[0]-1.0, np.ceil(y+(h-1)/scale-0.5)+1)
        return (im[y1:y2+1,x1:x2+1], x-x1, y-y1)

    def render(self, im, x, y, w, h, scale, bgcolor):
        """
        Parameters
        ----------
          im: float32 grey image
          x, y: initial point
          w, h: size of zoomed output
          scale: zoom factor

        Output
        ------
          out: zoomed image
        """
        return zoom.fzoom(im, zoom.Zoom.NEAREST, x, y, w, h, scale, bgcolor)

class BilinearZoom(object):
    def __str__(self):
        return "BilinearZoom"

    def subwindow(self, im, x, y, w, h, scale):
        """
        Extract the minimal subwindow of im required to compute the
        requested (x, y, w, h, scale) zoom
        """
        return _extract_subwindow_with_border(im, x, y, w, h, scale, 2)

    def render(self, im, x, y, w, h, scale, bgcolor):
        """
        Parameters
        ----------
          im: float32 grey image
          x, y: initial point
          w, h: size of zoomed output
          scale: zoom factor

        Output
        ------
          out: zoomed image
        """
        return zoom.fzoom(im, zoom.Zoom.BILINEAR, x, y, w, h, scale, bgcolor)

class BicubicZoom(object):
    def __str__(self):
        return "CubicZoom"

    def subwindow(self, im, x, y, w, h, scale):
        """
        Extract the minimal subwindow of im required to compute the
        requested (x, y, w, h, scale) zoom
        """
        return _extract_subwindow_with_border(im, x, y, w, h, scale, 3)

    def render(self, im, x, y, w, h, scale, bgcolor):
        """
        Parameters
        ----------
          im: float32 grey image
          x, y: initial point
          w, h: size of zoomed output
          scale: zoom factor

        Output
        ------
          out: zoomed image
        """
        return zoom.fzoom(im, zoom.Zoom.BICUBIC, x, y, w, h, scale, bgcolor)

class SplineZoom(object):
    def __init__(self, order=3):
        assert order in [3,5,7,9,11]
        self.order = order

    def __str__(self):
        return "SplineZoom("+str(self.order)+")"

    def subwindow(self, im, x, y, w, h, scale):
        """
        Extract the minimal subwindow of im required to compute the
        requested (x, y, w, h, scale) zoom
        """
        return _extract_subwindow_with_border(im, x, y, w, h, scale, (1+self.order)/2+1)

    def render(self, im, x, y, w, h, scale, bgcolor):
        """
        Parameters
        ----------
          im: float32 grey image
          x, y: initial point
          w, h: size of zoomed output
          scale: zoom factor

        Output
        ------
          out: zoomed image
        """
        return zoom.fzoom(im, self.order, x, y, w, h, scale, bgcolor)

# Utility function that converts an image to float32 when required
def _ensures_float(im):
    if not np.issubdtype(im.dtype, float):
        return im.astype(np.float32)
    else:
        return im

class ImageRenderer(object):
    """
    Renders a subimage at any floating-point position and zoom scale.
    This is very efficient (uses caching to avoid complete recomputations).
    It is extensible, and you can register postprocessing functions that act on the
    part of the original image that you want to render -- for instance to change
    the gray value scales, processing functions that act on the zoomed subimage
    as a floating-point image and postprocessing functions that act on the zoomed
    subimage after it has been rescaled as a char image. These processing functions
    can request to work on a wider subimage than the one originally requested, when
    processing computations need an "image border".
    """
    EPS = 1e-5 # precision for floating-point computations for the point coordinates

    def __init__(self, zoom=NearestZoom()):
        # Image
        self.planes = None # Image planes, either 1 (gray) or 3 (r, g, b) elements
        self.width, self.height = 0, 0 # Image size

        # Caching
        self.state = None # Last requested view
        self.cache = None # Last subimage rendering

        self.zoom = zoom # Zoom to use

        self.rescale = None # Rescale images
        # TODO
        # Not used right now, should enable to set the desired range
        # to display (particularly for float images)

        self.show_level_lines = False # Display level-lines

    def set_image(self, image, rescale=None):
        if image is None:
            self.planes = None
            self.width = self.height = 0
        else:
            if image.ndim == 2:
                self.planes = [_ensures_float(image)]
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    self.planes = [_ensures_float(image[:,:,0])]
                elif image.shape[2] == 3 or image.shape[2] == 4:
                    imageR = _ensures_float(image[:,:,0])
                    imageG = _ensures_float(image[:,:,1])
                    imageB = _ensures_float(image[:,:,2])
                    self.planes = [imageR, imageG, imageB]
                else:
                    raise ValueError('Invalid number of planes')
            else:
                raise ValueError('The image must be either a 2D or 3D array')

            self.height, self.width = image.shape[0], image.shape[1]
        self.rescale = rescale
    def set_zoom(self, zoom):
        self.zoom = zoom
    def set_rescale(self, rescale):
        self.rescale = rescale

    def render(self, x, y, w, h, scale=1.0, bgcolor=0.0):
        """
        Parameters
        ----------
         x, y: float
               position (in the first image) of the sample corresponding to the
               zoomed image first pixel
         w, h: int
               size of the output zoomed image
         scale: float
               value of the zoom (1.0: no zooming)
         bgcolor: float
               gray value of the background, when interpolating around the border

        Output
        ------
         out: QPixmap
              the zoomed and cropped image
        """
        def render_all():
            self.state = (x, y, w, h, scale, bgcolor)
            self.cache = self._force_render(x, y, w, h, scale, bgcolor)
            return self.cache

        if self.state is None or self.cache is None:
            return render_all()

        if self.planes is None or w == 0 or h == 0:
            return render_all()

        px, py, pw, ph, pscale, pbgcolor = self.state
        if pscale != scale or pbgcolor != bgcolor:
            return render_all()

        if self.state == (x, y, w, h, scale, bgcolor):
            return self.cache

        self.state = (x, y, w, h, scale, bgcolor)

        # Is the translation (px,py) -> (x,y) an integer amount of the current
        # scale?
        dx = (x-px)*scale
        dy = (y-py)*scale
        if abs(dx-np.rint(dx)) < self.EPS and abs(dy-np.rint(dy)) < self.EPS:
            # convert the floating coords to integer coords
            # we set the origin at (px, py) (ie. i_px = 0, i_py = 0)
            i_px, i_py = 0, 0
            i_x, i_y = int(np.rint(dx)), int(np.rint(dy))

            prev = utils.IntBox(i_px, i_py, pw, ph)
            next = utils.IntBox(i_x, i_y, w, h)

            intersection = next.intersection(prev)

            if intersection.is_empty():
                return render_all()
            else:
                difference_list = next.difference(prev)

                # TODO
                #
                # if the common part is really small compared to the whole area
                # to render, perhaps it is more efficient to render the whole picture
                # rather than rendering parts, and the copying
                #
                # eg.
                # if intersection.area()/next.area() < thresh:
                #     return render_all()
                # or
                # if intersection.area() < thresh:
                #     return render_all()

                out = np.empty((h,w,3), dtype=np.uint8)

                a, b, s, t = intersection.coords() # in cache coords
                c, d = a-i_x, b-i_y # in out coords
                out[d:d+t, c:c+s, :] = self.cache[b:b+t, a:a+s, :]

                for diff in difference_list:
                    a, b, s, t = diff.coords() # in cache coords
                    c, d = a-i_x, b-i_y # in out coords
                    fx = px + float(a)/scale
                    fy = py + float(b)/scale
                    A = self._force_render(fx, fy, s, t, scale, bgcolor)
                    out[d:d+t, c:c+s, :] = A

                self.cache = out
                return out

        else:
            print "px=", px, "x=", x, "dx=", dx
            print "py=", py, "y=", y, "dy=", dy
            print "[WARNING] not an integer translation, force rendering!"

        return render_all()

    def _force_render(self, x, y, w, h, scale, bgcolor=0):

        # Since level_lines can not be computed on last line and col, we should
        # request one line and col more (filled with background color if out of
        # the image)
        if self.show_level_lines:
            w = w+1
            h = h+1

        if self.planes is None or w == 0 or h == 0:
            return np.empty((0,0))
        else:
            zoomed_planes = []
            if abs(scale-1.0) < self.EPS and \
                    abs(x-np.rint(x)) < self.EPS and abs(y - np.rint(y)) < self.EPS:
                # integer translation of scale 1.0, we can immediately crop
                # and copy the image planes
                i_x = int(np.rint(x))
                i_y = int(np.rint(y))
                # Compute the intersection between the original image at
                # (0, 0) of size (self.width, self.height) and the desired
                # subimage at (i_x, i_y) of size (w, h)
                i_px, i_py = 0, 0
                pw, ph = self.width, self.height
                prev = utils.IntBox(i_px, i_py, pw, ph)
                next = utils.IntBox(i_x, i_y, w, h)

                intersection = next.intersection(prev)
                if intersection.is_empty():
                    # empty intersection
                    for plane in self.planes:
                        new_plane = np.ones((h,w), dtype=np.float32)*bgcolor
                        zoomed_planes.append(new_plane)
                else:
                    # non-empty intersection
                    # dimension of the intersecting rectangle
                    a, b, cw, ch = intersection.coords()
                    x0, y0 = a-i_x, b-i_y # in new_plane coords
                    for plane in self.planes:
                        new_plane = np.ones((h,w), dtype=np.float32)*bgcolor
                        new_plane[y0:y0+ch, x0:x0+cw] = \
                                plane[b:b+ch, a:a+cw]
                        zoomed_planes.append(new_plane)
            else:
                for plane in self.planes:
                    # non-integer translation, or scale != 1.0
                    im, new_x, new_y = \
                            self.zoom.subwindow(plane, x, y, w, h, scale)
                    # If required: preprocess im
                    im_zoom = self.zoom.render(im, new_x, new_y, w, h, scale, bgcolor)
                    zoomed_planes.append(im_zoom)

            res = self.process(zoomed_planes)
            assert len(res) == 1 or len(res) == 3

            # Show level-lines and remove extra col
            if self.show_level_lines:
                res = self._display_level_lines(zoomed_planes)
                h = h-1
                w = w-1
                for i in xrange(len(res)):
                    res[i] = res[i][0:h, 0:w]

            # Clip results, avoiding duplicates of the channels
            for i, im in enumerate(res):
                for j in xrange(i):
                    if res[i] is res[j]:
                        break
                else:
                    # If we haven't processed res[i] before
                    res[i] = im.clip(0,255).astype(np.uint8)

            if len(res) == 1:
                im=res[0]
                res = [im, im, im]

            out = np.empty((h, w, 3), dtype=np.uint8)
            out[:,:,0] = res[0]
            out[:,:,1] = res[1]
            out[:,:,2] = res[2]

            return out

    def process(self, planes):
        return planes

    def _display_level_lines(self, planes):
        if len(planes) == 1:
            if self.show_level_lines:
                im = planes[0]
                level_lines = utils._extract_level_lines(im,ofs=0.0,step=27.0,mode=1)
                imR = im
                imG = im.copy()
                imB = im.copy()
                imR[level_lines]=255.0
                imG[level_lines]=0.0
                imB[level_lines]=0.0
                return [imR, imG, imB]
            else:
                return planes
            #return list(planes)
        elif len(planes) == 3:
            if self.show_level_lines:
                im = (planes[0]+planes[1]+planes[2])/3.0
                level_lines = utils._extract_level_lines(im,ofs=0.0,step=27.0,mode=1)
                planes[0][level_lines]=255.0
                planes[1][level_lines]=0.0
                planes[2][level_lines]=0.0
                return planes
            else:
                return planes
        else:
            raise ValueError("Either one or three arguments")

    def reinit_state(self):
        self.state = None
        self.cache = None

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.setBackgroundColor(QPalette.Window)
        self.setForegroundColor(QPalette.Text)
        self.image_renderer = None
        self.pan_x, self.pan_y = 0.0, 0.0
        self.scale = 1.0
        self.dirty = True
        self.pixmap = None

    def setBackgroundColor(self, color):
        self.background_color = color
        self.background_brush = QBrush(self.background_color)
    def setForegroundColor(self, color):
        self.foreground_color = color
        self.foreground_pen = QPen(self.foreground_color)
    def setImageRenderer(self, image_renderer):
        self.image_renderer = image_renderer

    def _image_coords_to_widget_coords(self, ix, iy):
        return (int((self.pan_x+ix)*self.scale+0.5), int((self.pan_y+iy)*self.scale+0.5))
    def _widget_coords_to_image_coords(self, wx, wy):
        return (float(wx)/self.scale-self.pan_x, float(wy)/self.scale-self.pan_y)

    # This function computes the top-left pixel of the image to draw, as well as
    # the rectangle to render
    def _get_image_representation_coordinates(self):
        image_renderer = self.image_renderer
        im_w, im_h = image_renderer.width, image_renderer.height
        pan_x, pan_y = self.pan_x, self.pan_y
        scale = self.scale
        # Upper-left coords of the image to show, in image coords
        x = max(-pan_x, 0.0)
        y = max(-pan_y, 0.0)
        # In widget coords
        wx = int(max(pan_x*scale, 0.0))
        wy = int(max(pan_y*scale, 0.0))
        # Part of the image that is displayed
        ex = min(self.width(), int((pan_x+im_w)*scale))
        ey = min(self.height(), int((pan_y+im_h)*scale))
        w = max(ex - wx, 0)
        h = max(ey - wy, 0)
        return (x, y, wx, wy, w, h)

    def resizeEvent(self, event):
        self.dirty = True
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setClipRect(event.rect())

        # Background
        painter.setPen(self.foreground_pen)
        painter.setBrush(self.background_brush)
        painter.drawRect(self.rect().adjusted(0,0,-1,-1))

        # Image
        w, h = self.width(), self.height()

        if self.image_renderer is None:
            painter.setPen(self.foreground_pen)
            painter.drawLine(QPoint(0, 0), QPoint(w, h))
            painter.drawLine(QPoint(w, 0), QPoint(0, h))
        else:
            image_renderer = self.image_renderer
            im_w, im_h = image_renderer.width, image_renderer.height
            pan_x, pan_y = self.pan_x, self.pan_y
            scale = self.scale

            x, y, wx, wy, w, h = self._get_image_representation_coordinates()
            if self.dirty or self.pixmap is None:
                out = image_renderer.render(x, y, w, h, self.scale)
                self.pixmap = utils._to_pixmap(out)
                self.dirty = False

            if w > 0 and h > 0:
                painter.setPen(QPen(Qt.red))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(wx, wy, w-1, h-1)
                painter.drawPixmap(QPoint(wx, wy), self.pixmap)

        painter.end()

    def sizeHint(self):
        if self.image_renderer is not None:
            i = self.image_renderer
            width, height = i.width, i.height
            return QSize(width, height)
        else:
            return QSize(100, 100)

class MouseImageViewer(ImageViewer):
    def mousePressEvent(self, event):
        wx, wy = event.x(), event.y()
        btn = event.button()

        ix, iy = self._widget_coords_to_image_coords(wx, wy)
        print "You clicked on image position ({0:.3g}, {0:.3g})".format(ix, iy)

        if btn == Qt.LeftButton or btn == Qt.RightButton:
            dx = self.pan_x - wx/self.scale
            dy = self.pan_y - wy/self.scale
            if btn == Qt.LeftButton:
                self.scale *= 2.0
            elif btn == Qt.RightButton:
                self.scale /= 2.0
            self.pan_x = dx + wx/self.scale
            self.pan_y = dy + wy/self.scale
            # TODO
            # should signal pan/zoom change
            # eg. emit self.viewChanged(...)
            self.dirty = True
            self.repaint()

class Controls(QWidget):
    def __init__(self, parent=None):
        super(Controls, self).__init__(parent)
        self.viewer = None
        self.flip = None

        self.zooms = [NearestZoom(), BilinearZoom(), BicubicZoom()]
        for order in [3,5,7,9,11]:
            self.zooms.append(SplineZoom(order))

    def setViewer(self, viewer):
        layout = QVBoxLayout()
        layout.addWidget(viewer)
        self.viewer = viewer
        # TODO: should respond to viewChanged signal of self.viewer,
        # in order to force centering the view if it fits the viewer
        self.setLayout(layout)

    def setImageRenderer(self, image_renderer, flip=None):
        if self.viewer is not None:
            self.viewer.setImageRenderer(image_renderer)
            self.flip = flip

    def _is_view_fitting(self):
        if self.viewer is not None:
            viewer = self.viewer
            ir = viewer.image_renderer
            if ir is not None:
                w, h = ir.width, ir.height
                scale = viewer.scale
                real_width = int(np.ceil(w*scale))
                real_height = int(np.ceil(h*scale))
                return real_width <= viewer.width() and real_height <= viewer.height()
            else:
                return False
        else:
            return False
    def _fit_view(self):
        if self.viewer is not None:
            viewer = self.viewer
            ir = viewer.image_renderer
            if ir is not None:
                w, h = ir.width, ir.height
                scale = viewer.scale
                real_width = int(np.ceil(w*scale))
                real_height = int(np.ceil(h*scale))
                widget_width = viewer.width()
                widget_height = viewer.height()
                if real_width <= widget_width:
                    viewer.pan_x = (widget_width-real_width)/scale/2
                if real_height <= widget_height:
                    viewer.pan_y = (widget_height-real_height)/scale/2

    def resizeEvent(self, event):
        self._fit_view()
        viewer.dirty = True

    def pan(self, dx=0.0, dy=0.0):
        viewer = self.viewer
        viewer.pan_x += dx
        viewer.pan_y += dy
        self._fit_view()
        viewer.dirty = True

    def zoom(self, scale):
        viewer = self.viewer
        viewer.scale = scale
        self._fit_view()
        viewer.dirty = True

    def zoom_fit(self):
        # self.zoom_rect(image_rect())
        pass
    def zoom_rect(self, rect):
        pass

    def keyPressEvent(self, event):
        viewer = self.viewer

        c = event.key()
        if c == Qt.Key_Left:
            self.pan(dx=50.0/viewer.scale)
            viewer.repaint()
        elif c == Qt.Key_Right:
            self.pan(dx=-50.0/viewer.scale)
            viewer.repaint()
        elif c == Qt.Key_Up:
            self.pan(dy=50.0/viewer.scale)
            viewer.repaint()
        elif c == Qt.Key_Down:
            self.pan(dy=-50.0/viewer.scale)
            viewer.repaint()
        elif c == Qt.Key_Plus:
            self.zoom(viewer.scale*2.0)
            viewer.repaint()
        elif c == Qt.Key_Minus:
            self.zoom(viewer.scale/2.0)
            viewer.repaint()
        elif c == Qt.Key_Space:
            # Flip images
            if self.flip is not None:
                image_renderer = self.flip
                self.flip = viewer.image_renderer
                viewer.setImageRenderer(image_renderer)
                viewer.dirty = True
                viewer.repaint()
        elif c == Qt.Key_F:
            # Force redraw
            viewer.image_renderer.state = None
            viewer.dirty = True
            viewer.repaint()
        elif c == Qt.Key_L:
            # Show level-lines
            # TODO
            # parameters for level-lines should be in extra controls accessible
            # with F10
            viewer.image_renderer.show_level_lines = not viewer.image_renderer.show_level_lines
            viewer.image_renderer.state = None
            if self.flip:
                self.flip.show_level_lines = not self.flip.show_level_lines
                self.flip.state = None
            viewer.dirty = True
            viewer.repaint()
        elif c == Qt.Key_B:
            # Change interpolation
            # TODO
            # This should be a menu accessible with F10 (show some additional
            # menus)
            cur_zoom = self.zooms[0]
            self.zooms = self.zooms[1:]
            self.zooms.append(cur_zoom)
            print str(self.zooms[0])
            viewer.image_renderer.zoom = self.zooms[0]
            viewer.image_renderer.state = None
            if self.flip:
                self.flip.zoom = self.zooms[0]
                self.flip.state = None
            viewer.dirty = True
            viewer.repaint()
        elif c == Qt.Key_0:
            # Reinit zoom
            self.zoom(1.0)
            viewer.repaint()

class AdvancedImageViewerApp(QMainWindow):
    def __init__(self, im, flip=None, mgr=None):
        super(AdvancedImageViewerApp, self).__init__()
        self.mgr = mgr
        if mgr is not None:
            self.mgr.add_window(self)

        # Basic image rendering
        im_renderer = ImageRenderer()
        im_renderer.set_image(im)

        if flip is not None:
            # Basic image rendering
            flip_renderer = ImageRenderer()
            flip_renderer.set_image(flip)
        else:
            flip_renderer = None

        # Viewer handling mouse
        viewer = MouseImageViewer()

        # Advanced controls
        controls = Controls()
        controls.setViewer(viewer)
        controls.setImageRenderer(im_renderer, flip=flip_renderer)
        self.setCentralWidget(controls)
        controls.show()

    def keyPressEvent(self, event):
        c = event.key()

        if c == Qt.Key_Q:
            # Close
            self.close()
        else:
            self.centralWidget().keyPressEvent(event)

    def closeEvent(self, event):
        # Allow window to be destroyed by removing any
        # references to it
        if self.mgr is not None:
            self.mgr.remove_window(self)

def _simple_imshow(im, flip=None, mgr=None):
    # TODO: simpler imshow, without complete GUI
    return _advanced_imshow(im, flip=flip, mgr=mgr)

def _advanced_imshow(im, flip=None, mgr=None):
    return AdvancedImageViewerApp(im, flip=flip, mgr=mgr)

if __name__ == "__main__":

    from scikits.image.filter import tvdenoise
    from scikits.image.io import imread, imshow
    import numpy.random as npr
    import os, os.path
    import sys

    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        image = imread(sys.argv[1])
    else:
        import scipy
        image = scipy.lena()

    flip = None
    if len(sys.argv) > 2:
        flip = imread(sys.argv[2])

    viewer = _advanced_imshow(image, flip=flip, mgr=None)
    viewer.show()

    sys.exit(app.exec_())
