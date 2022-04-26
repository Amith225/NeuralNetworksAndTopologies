import os
from abc import ABCMeta, abstractmethod


class BaseSave(metaclass=ABCMeta):
    DEFAULT_DIR: str
    DEFAULT_NAME: str
    FILE_TYPE: str

    @abstractmethod
    def saveName(self) -> str:
        pass

    @abstractmethod
    def save(self, file: str = None, replace: bool = False) -> str:
        if file is None: file = self.DEFAULT_NAME
        if not (fpath := os.path.dirname(file)):
            fpath = f"{os.getcwd()}\\{self.DEFAULT_DIR}\\"
            fName = file
        else:
            fpath += '\\'
            fName = os.path.basename(file)
        os.makedirs(fpath, exist_ok=True)
        if len(fName) >= (typeLen := 1 + len(self.FILE_TYPE)) and fName[1 - typeLen:] == self.FILE_TYPE:
            fName = fName[:-typeLen]
            savePath = f"{fpath}{fName.replace(' ', '_')}"
        else:
            savePath = f"{fpath}{fName.replace(' ', '_')}_{self.saveName().replace(' ', '')}"

        numSavePath = savePath
        if not replace:
            i = 0
            while 1:
                if i != 0: numSavePath = f"{savePath} ({i})"
                if not os.path.exists(f"{numSavePath}.{self.FILE_TYPE}"): break
                i += 1

        dumpFile = f"{numSavePath}.{self.FILE_TYPE}"
        return dumpFile


class BaseLoad(metaclass=ABCMeta):
    DEFAULT_DIR: str
    FILE_TYPE: str

    @classmethod
    @abstractmethod
    def load(cls, file):
        if file:
            if not (fpath := os.path.dirname(file)):
                fpath = f"{os.getcwd()}\\{cls.DEFAULT_DIR}\\"
                fName = file
            else:
                fpath += '\\'
                fName = os.path.basename(file)
        else:
            raise NameError("file not given")
        if '.' not in fName: fName += cls.FILE_TYPE

        loadFile = fpath + fName
        return loadFile
