def print_incrementally(func):
    def wrapper(self, *args, **kwargs):
        self.print_interval_bool = [False] * len(self.gid)
        for idx, _gid_ in enumerate(self.gid):
            if _gid_ <= 100 and _gid_ % 4 == 0:
                self.print_interval_bool[idx] = True
            if _gid_ <= 500 and _gid_ % 20 == 0:
                self.print_interval_bool[idx] = True
            if _gid_ <= 1000 and _gid_ % 40 == 0:
                self.print_interval_bool[idx] = True
            if _gid_ <= 2000 and _gid_ % 80 == 0:
                self.print_interval_bool[idx] = True
        return func(self, *args, **kwargs)
    return wrapper

def console_seperator(seperator = '-*',
                      repetitions = 25,
                      ):
    '''
    Prints a nice seperator line
    '''
    _BOLD = '\033[1m'
    _UNDERLINE = '\033[4m'
    _END = '\033[0m'
    from colorama import init, Fore, Back, Style
    init()
    print(_BOLD + Back.YELLOW + Fore.RED + repetitions*seperator + Style.RESET_ALL)
