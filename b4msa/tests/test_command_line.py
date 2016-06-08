# Copyright 2016 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_command_line():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', fname]
    c.main()


def test_nparams():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', '-N', '11', fname]
    c.main()
    # assert False


def test_main():
    from b4msa.command_line import main
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-k', '2', fname]
    main()
