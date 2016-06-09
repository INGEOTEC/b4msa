# Copyright 2016 Mario Graff (https://github.com/mgraffg) and Ranyart R. Suarez (https://github.com/RanyartRodrigo)

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
    os.unlink(c.get_output())
    # assert False


def test_nparams():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', '-s', '11', fname]
    c.main()
    os.unlink(c.get_output())


def test_main():
    from b4msa.command_line import params
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname]
    params()
    os.unlink(output)


def test_pool():
    from b4msa.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-k', '2', '-s', '11', '-n', '2', fname]
    c.main()
    os.unlink(c.get_output())


def test_output():
    from b4msa.command_line import CommandLine
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['b4msa', '-o', output, '-k', '2', fname]
    c.main()
    assert os.path.isfile(output)
    os.unlink(output)
