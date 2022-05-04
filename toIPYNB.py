import os
from p2j.p2j import p2j

__ignore__ = ["venv", "node_modules", "test.py", "test2.py", "test3.py", "testNumba.py"]
__bytes__ = []
__ignore__ += ["toIPYNB.py", ".git", ".idea", ".vs", "__pycache__", "__ipynb__.py", "process.py"]
__path__ = os.path.dirname(__file__)


def importing_or_content_null(lines, i):
    if 'import' in (_line := lines[i].strip()):
        if lines[i][0] == ' ':
            back = 1
            while True:
                if ':' in lines[i - back]:
                    if "TYPE_CHECKING" in lines[i - back]:
                        return None
                    else:
                        break
                back -= 1
        return True
    return False


def makeIPYNB(path):
    contents = {}
    top_level_imports = {}
    for dirs in os.listdir(path):
        local_imports = set()
        content = ''
        if dirs in __ignore__: continue
        if os.path.isfile(_path := path + '/' + dirs) and os.path.splitext(_path)[1] == '.py':
            print(fName := _path.removeprefix(__path__), ':', sep='')
            content += '\n# # ' + fName + '\n'
            with open(_path, 'r') as f:
                lines = f.readlines()
                skip_next_line = False
                for i, line in enumerate(lines):
                    if skip_next_line:
                        skip_next_line = False
                        continue
                    _line = line.strip()
                    if _to := importing_or_content_null(lines, i):
                        _imp_type = ''
                        if 'from ' == _line[:5] or ' from ' in _line:
                            imp = _line.removeprefix('from ').split(' import ')
                            _imp_type = 'from'
                        elif 'import ' in _line[:7] or ' import ' in _line:
                            imp = _line.removeprefix('import ').split(' as ')
                            if len(imp) == 2: imp[1] = f'as {imp[1]}'
                            _imp_type = 'import'
                        else:
                            content += line

                        if _line[-1] == '\\' and i != len(lines) - 1:
                            _line = _line[:-1] + lines[i + 1].strip()
                            skip_next_line = True

                        p = imp[0].replace('.', '/')
                        if imp[0][0] != '.' and not (os.path.exists(p) or os.path.exists(p + '.py')):
                            print('lib import added: ', _line)
                            (_set := top_level_imports.get(imp[0], set())).update(
                                imp[1].split(', ') if len(imp) == 2 else set())
                            top_level_imports[imp[0]] = _set
                        else:
                            print("local import purged: ", _line, [f"{p}/{i}" for i in imp[1].split(', ')])
                            local_imports.update([imp[0]])
                        content += line.replace(line.strip(), f'pass  # {_line}')
                    elif _to is not None:
                        content += line
                    else:
                        content += line.replace(line.strip(), f'pass  # {_line}')
            print('Done', '\n')
            contents[fName] = [content, local_imports]
        elif os.path.isdir(_path):
            _top_level_imports, _contents = makeIPYNB(_path)
            top_level_imports.update((k, (i.update(top_level_imports.get(k, set())), i)[1])
                                     for k, i in _top_level_imports.items())
            contents.update(_contents)
    return top_level_imports, contents


def find_dot_path(p: str):
    if '..' in p:
        return find_dot_path(os.path.dirname(p) + p[p.index('..') + 1:])
    elif '.' in p:
        return p.replace('.', '/')
    else:
        return p


def sort_c(_c: dict, *, __secret_dict=None):
    if isinstance(_c, dict):
        for k in _c:
            _c[k].append(sort_c(k, __secret_dict=_c))
        return sorted(_c, key=lambda cc: _c[cc][2])
    elif isinstance(_c, str):
        if _c in __secret_dict:
            rank = 0
            for ccc in __secret_dict[_c][1]:
                next_c = find_dot_path(os.path.dirname(_c) + ccc) + '.py'
                rank += sort_c(next_c, __secret_dict=__secret_dict) + 1
            return rank
        else:
            if (p := _c.replace('.py', '/') + '__init__.py') not in __secret_dict:
                return 0
            else: return sort_c(p, __secret_dict=__secret_dict)  # noqa


tli, c = makeIPYNB(__path__)
c_key = sort_c(c)
[print(k, c[k][2]) for k in c_key]
tli = ''.join(f"{'from' if (from_import := len(i) != 0 and tuple(i)[0][:3] != 'as ') else 'import'} {k} "
              f"{'import ' if from_import else ''}"
              f"{', '.join(i)}\n" for k, i in tli.items())
with open('__ipynb__.py', 'w') as f:
    f.write(tli + '\n' + ''.join(c[k][0] for k in c_key))

p2j('__ipynb__.py', '__ipynb__.ipynb', True)
os.remove('__ipynb__.py')
