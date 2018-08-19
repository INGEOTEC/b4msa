import os
import glob
# import subprocess
# import traceback
from binstar_client.scripts import cli


def get_token():
    token = None
    if os.environ.get('TRAVIS_BRANCH', None) == 'master' or os.environ.get('APPVEYOR_REPO_BRANCH', None) == 'master':
        token = os.environ.get('BINSTAR_TOKEN', None)
    return token


token = get_token()
if token is not None:
    cmd = ['-t', token, 'upload', '--force', '-u', 'ingeotec']
    cmd.extend(glob.glob('*.tar.bz2'))
    cli.main(args=cmd)
    # try:
    #     print('*', cmd, platform.system())
    #     subprocess.check_call(cmd)
    # except subprocess.CalledProcessError:
    #     traceback.print_exc()
