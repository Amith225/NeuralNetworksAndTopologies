from cmd import Cmd


class Prompt(Cmd):
    def do_run_setup(self, args):
        """Establish Environment variables"""
        args = args.split()


if __name__ == '__main__':
    prompt = Prompt()
    prompt.prompt = '>>: '
    prompt.cmdloop('Neural Network And Topologies Work Base')
