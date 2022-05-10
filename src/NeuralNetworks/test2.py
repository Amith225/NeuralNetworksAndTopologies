from src import Conv

shape = Conv.Shape((10, 3),
                   (10, 4, 3),
                   (5, 3), ..., (5, 4, 5), ..., (10, 3, 3),
                   (10, 2))
print(shape)
