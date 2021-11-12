import sys
import os
import yaml
import glob
import shutil
from conda_build.config import Config

config = Config()

with open(os.path.join(sys.argv[1], 'meta.yaml')) as f:
    name = yaml.safe_load(f)['package']['name']

binary_package_glob = os.path.join(config.bldpkgs_dir, '{0}*.tar.bz2'.format(name))
binary_package = glob.glob(binary_package_glob)[0]

shutil.move(binary_package, '.')
