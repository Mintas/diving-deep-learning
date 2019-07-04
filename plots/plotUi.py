import os

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf as pltpdf


class ShowPlotUi():
    def toView(self, plot):
        fig = self.figure()
        plot()
        self.show(fig)

    def figure(self):
        return plt.figure(figsize=(10, 10))

    def show(self, fig):
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
        self.lastFig = None

    def toView(self, plot):
        self.figure()
        plot()
        self.show()

    def figure(self):
        figure = plt.figure(figsize=(10, 10))
        self.lastFig = figure
        return figure

    def show(self, fig=None):
        if fig is None: fig = self.lastFig
        self.pdf.savefig(fig)

    def close(self):
        self.pdf.close()