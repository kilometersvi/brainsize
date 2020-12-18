import fitz
import sys

doc = fitz.open(sys.argv[1])
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha < 4:       # this is GRAY or RGB
            pix.writePNG("h_img/%s_human_%s.png" % (i, xref))
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("h_img/%s_human_%s.png" % (i, xref))
            pix1 = None
        pix = None
