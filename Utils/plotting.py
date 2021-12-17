import numpy as np
from matplotlib import pyplot as plt, widgets as wg

# setup for plotting
plt.style.use('dark_background')


class Plot:
    @staticmethod
    def __init_ax(ax):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax, fig = ax, ax.figure

        return ax

    @staticmethod
    def __plotMulti(count, plotter, rows, columns):
        MAX = rows * columns
        fig = plt.figure()
        axes = [fig.add_subplot(rows + 1, columns, i + 1) for i in range(MAX)]
        butPrev = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 1), '<-', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        butNext = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 2), '->', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        fig.page = 0

        def onclick(_page):
            if _page == 0:
                butPrev.active = False
                butPrev.ax.patch.set_visible(False)
            else:
                butPrev.active = True
                butPrev.ax.patch.set_visible(True)
            if _page == count // MAX:
                butNext.active = False
                butNext.ax.patch.set_visible(False)
            else:
                butNext.active = True
                butNext.ax.patch.set_visible(True)
            fig.page = _page
            [(ax.clear(), ax.set_xticks([]), ax.set_yticks([])) for ax in axes]
            plotter(_page, axes)
            fig.subplots_adjust(.01, .01, .99, .99, 0, 0)
            fig.canvas.draw()

        butNext.on_clicked(lambda *_: onclick(fig.page + 1))
        butPrev.on_clicked(lambda *_: onclick(fig.page - 1))
        onclick(fig.page)

        return axes

    @staticmethod
    def plotHeight(xs, ys=None, cluster=False, join=True,
                   scatter=False, scatterLabels=None, scatterRotation=0, scatterSize=100,
                   text='', textPos=(.01, .01), textC='yellow', ax=None, multi=False, rows=4, columns=4):
        if multi:
            MAX = rows * columns
            length = np.shape(xs)[0]
            if text is None:
                text = range(length)
            if scatterLabels is None:
                scatterLabels = range(length)
            if ys is None:
                ys = [None for _ in range(length)]

            def plotter(_page, axes):
                if (to := (_page + 1) * MAX) > length:
                    to = length
                for ind, x in enumerate(xs[_page * MAX:to]):
                    Plot.plotHeight(x, ys[ind], cluster, join, scatter, scatterLabels, scatterRotation,
                                    text=str(text[_page * MAX + ind]), ax=axes[ind])

            return Plot.__plotMulti(xs.shape[0], plotter, rows, columns)

        ax = Plot.__init_ax(ax)
        args = (xs, ys)
        if ys is None:
            args = (xs,)
        if not cluster:
            args = [[arg] for arg in args]
            scatterLabels = [scatterLabels]
        for i in range(len(xs)):
            points = [arg[i] for arg in args if arg[i] is not None]
            if join:
                ax.plot(*points, c=np.random.rand(3))
            if scatter or scatterLabels is not None:
                ax.scatter(*points, s=scatterSize, color="gray")
                if scatterLabels is not None:
                    for *point, label in zip(*points, scatterLabels[i]):
                        ax.annotate(label, point, rotation=scatterRotation)
        ax.text(*textPos, text, transform=ax.transAxes, c=textC)

        return ax

    @staticmethod
    def plotMap(vect, text=None, textPos=(.01, .01), textC='yellow', ax=None, rows=4, columns=4):
        if len(np.shape(vect)) != 2:
            MAX = rows * columns
            length = np.shape(vect)[0]
            if text is None:
                text = range(length)

            def plotter(_page, axes):
                if (to := (_page + 1) * MAX) > length:
                    to = length
                for i, im in enumerate(vect[_page * MAX:to]):
                    Plot.plotMap(im, text=str(text[_page * MAX + i]), ax=axes[i])

            return Plot.__plotMulti(length, plotter, rows, columns)

        ax = Plot.__init_ax(ax)
        ax.imshow(vect)
        ax.text(*textPos, text, transform=ax.transAxes, c=textC)

        return ax

    @staticmethod
    def show():
        plt.show()
