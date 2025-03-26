if __name__ is not None and "." in __name__:
    from .pbox import Pbox
else:
    from pbox import Pbox

if __name__ is not None and "." in __name__:
    from .interval import Interval
else:
    from interval import Interval

import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Union

__all__ = ["Cbox"]


class Cbox(Pbox):
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics. As such in pba-for-python c-boxes inhert most of their methods from Pbox.

    Args:
        Pbox (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Pbox):

            super().__init__(**vars(args[0]))

        else:

            super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.mean_left == self.mean_right:
            mean_text = f"{round(self.mean_left, 4)}"
        else:
            mean_text = f"[{round(self.mean_left, 4)}, {round(self.mean_right, 4)}]"

        if self.var_left == self.var_right:
            var_text = f"{round(self.var_left, 4)}"
        else:
            var_text = f"[{round(self.var_left, 4)}, {round(self.var_right, 4)}]"

        range_text = f"[{round(np.min([self.left, self.right]), 4), round(np.max([self.left, self.right]), 4)}"

        if self.shape is None:
            shape_text = " "
        else:
            shape_text = f" {self.shape}"  # space to start; see below lacking space

        return f"Cbox: ~ {shape_text} (range={range_text}, mean={mean_text}, var={var_text})"

    __str__ = __repr__

    def __add__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__add__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__add__(other))

    def __radd__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__radd__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__radd__(other))

    def add(self, other, method="f"):
        return Cbox(super().add(other, method=method))

    def __sub__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__sub__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__sub__(other))

    def __rsub__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rsub__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rsub__(other))

    def sub(self, other, method="f"):
        return Cbox(super().sub(other, method=method))

    def __mul__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__mul__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rmul__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rmul__(other))

    def mul(self, other, method="f"):
        return Cbox(super().mul(other, method=method))

    def __truediv__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__truediv__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__truediv__(other))

    def __rtruediv__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rtruediv__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rtruediv__(other))

    def div(self, other, method="f"):
        return Cbox(super().div(other, method=method))

    def __neg__(self):
        return Cbox(super().__neg__())

    def recip(self):
        return Cbox(super().recip())

    def get_confidence_interval(self, alpha1=0.95, alpha2=None) -> Interval:

        if alpha2 is None:

            alpha1 = (1 - alpha1) / 2
            alpha2 = 1 - alpha1

        else:

            assert alpha1 < alpha2

        left = self.get_interval(alpha1)
        right = self.get_interval(alpha2)

        return Interval(left.left, right.right)


def singh(
    cboxes: List[Cbox],
    theta: Union[float, List[float]],
    figax: Tuple[plt.Figure, plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a singh plot. Create s

    Parameters:
    cboxes (list): A list of Cboxes
    theta (float or iterable): A threshold value or list of values used to compare against 'left' and 'right' values of the cboxes.
    figax (tuple, optional): A tuple containing a Matplotlib figure and axes object. If not provided, a new figure and axes are created.

    Returns:
    tuple: A tuple containing the Matplotlib figure and axes objects used for plotting.

    """
    x = np.linspace(0, 1, 1001)

    if hasattr(theta, "__iter__"):
        l_thetas = [sum(cbox.left > a) / cbox.steps for cbox, a in zip(cboxes, theta)]
        r_thetas = [sum(cbox.right > a) / cbox.steps for cbox, a in zip(cboxes, theta)]
    else:
        l_thetas = [sum(cbox.left > theta) / cbox.steps for cbox in cboxes]
        r_thetas = [sum(cbox.right > theta) / cbox.steps for cbox in cboxes]

    left = [sum(i <= j for i in l_thetas) / len(cboxes) for j in x]
    right = [sum(i <= j for i in r_thetas) / len(cboxes) for j in x]

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    ax.plot([0] + list(x) + [1], [0] + left + [1])
    ax.plot([0] + list(x) + [1], [0] + right + [1])
    ax.plot([0, 1], [0, 1], "k--", lw=2)

    return fig, ax
