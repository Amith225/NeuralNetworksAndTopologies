from __init__.NeuralNetworks import base
from __init__ import ReadOnlyProperty


# class Shape(base.BaseShape):
#     @staticmethod
#     def _formatShapes(shapes) -> tuple:
#         return tuple(shapes)
#
#
# s = Shape(1, 2, 3, 4)

x = ReadOnlyProperty([1, 2, 3])
x[0] = 0
