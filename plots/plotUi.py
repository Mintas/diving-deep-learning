import os

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf as pltpdf


class ShowPlotUi():
    def toView(self, plot):
        plot()
        plt.show()

    def close(self):
        pass


class PDFPlotUi():
    def __init__(self, pdfFile) -> None:
        i = 0
        lookupNextName = pdfFile
        while (os.path.isfile(lookupNextName + '.pdf')) :
            i = i + 1
            lookupNextName = pdfFile + '_' + str(i)
        self.pdf = pltpdf.PdfPages(lookupNextName + '.pdf')

    def toView(self, plot):
        fig = plt.figure(figsize=(10, 10))
        plot()
        self.pdf.savefig(fig)

    def close(self):
        self.pdf.close()