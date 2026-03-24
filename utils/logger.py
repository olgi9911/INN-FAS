import sys
import os

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = None

    def open(self, file_path, mode='a'):
        self.log = open(file_path, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        # Needed for python 3 compatibility
        pass