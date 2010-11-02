"""scikits.image viewer"""
def main():
    import scikits.image.io as io
    import sys

    if len(sys.argv) < 2:
        print "Usage: scivi <image-file> [<flip-file>]"
        sys.exit(-1)

    io.use_plugin('qt2')
    im = io.imread(sys.argv[1])
    flip = None
    if len(sys.argv) > 2:
        flip = io.imread(sys.argv[2])
    io.imshow(im, flip=flip, fancy=True)
    io.show()

