import inspect


class MagicProperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(MagicProperty, self).__init__(fget, self.makeMagicF(fset), self.makeMagicF(fdel), doc)
        self.__obj = self.getCaller()

    def makeMagicF(self, _f):
        def f(*args, **kwargs):
            if self.__magic__():
                return _f(*args, **kwargs)
            else:
                raise AttributeError("Attribute is read only")

        return f

    def __magic__(self, stack=1):
        caller = self.getCaller(stack + 1)
        return any(c1 == c2 and c1 is not None for c1, c2 in zip(caller, self.__obj)) or \
               (any(self.__obj[2] == base.__name__ for base in caller[1].__bases__)
                if self.__obj[:2] == (None, None) and caller[1] is not None else 0)

    @staticmethod
    def getCaller(stack=1):
        caller = (callStack := inspect.stack()[stack + 1][0].f_locals).get('self')
        _return = caller, caller.__class__, caller.__class__.__name__
        if caller is None:
            _return = None, (caller := callStack.get('cls')), caller.__name__ if caller is not None else None
        if caller is None: _return = None, None, callStack.get('__qualname__')
        return _return


def makeMetaMagicProperty(*inherits):
    class MetaProperty(*inherits, type):  # todo: improve implementation method
        def __call__(cls, *args, **kwargs):
            __obj = super(MetaProperty, cls).__call__(*args, **kwargs)
            __dict__ = {}
            for key, val in __obj.__dict__.items():
                if key.isupper():
                    __dict__[(_name := '__magic' + key)] = val
                    setattr(cls, key, MagicProperty(lambda self, _name=_name: getattr(self, _name),
                                                    lambda self, _val, _name=_name: setattr(self, _name, _val)))
            __obj.__dict__.update(__dict__)
            return __obj

    return MetaProperty
