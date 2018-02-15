import os
import glob
import subprocess
import traceback

try:
    token = os.environ['BINSTAR_TOKEN']
    cmd = ['binstar', '-t', token, 'upload', '--force', '-u', 'ingeotec']
    cmd.extend(glob.glob('*.tar.bz2'))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        traceback.print_exc()
except KeyError:
    pass
