import os
from p2j.p2j import p2j

__ignore__ = ["venv", "node_modules", "test.py", "test2.py", "test3.py", "testNumba.py"]
__bytes__ = []
__ignore__ += ["toIPYNB.py", ".git", ".idea", ".vs", "__pycache__"]
__path__ = os.path.dirname(__file__)


def makeIPNB(path):
    for dirs in os.listdir(path):
        if dirs in __ignore__: continue
        if os.path.isfile(_path := path + '/' + dirs):
            if os.path.splitext(_path)[1] == '.py':
                tar = '__ipynb__' + _path.removeprefix(__path__).removesuffix('.py') + '.ipynb'
                os.makedirs(os.path.dirname(tar), exist_ok=True)
                with open(_path, 'r+') as file:
                    content = file.read()
                    file.seek(0, 0)
                    file.write("import import_ipynb\n" + content)
                p2j(source_filename=_path, target_filename=tar, overwrite=True)
                with open(_path, 'r+') as file:
                    content = file.read()
                    file.seek(0, 0)
                    file.write("import import_ipynb\n" + content)
        else:
            makeIPNB(_path)


makeIPNB(__path__)
