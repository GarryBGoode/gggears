# Copyright 2024 Gergely Bencsik
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gggears.gggears_wrapper as gg
import matplotlib.pyplot as plt
import doctest
import sys
import os


def test_example():
    """Test all the examples in the examples.py file. Can't make a passing assertion,
    except that there should not be any errors when running the examples."""
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, "..", "examples")
    sys.path.append(relative_path)

    from examples import spur_gears

    spur_gears()
    from examples import helical_gears

    helical_gears()
    from examples import worm_approx

    worm_approx()
    from examples import planetary_helical_gear

    planetary_helical_gear()
    from examples import bevel_gear

    bevel_gear()
    from examples import bevel_chain

    bevel_chain()
    from examples import fishbone_bevels

    fishbone_bevels()
    from examples import cycloid_gear

    cycloid_gear()
    from examples import cycloid_drive

    cycloid_drive()


def test_doctest():
    """
    Run doctests on the gggears module.
    """
    # Run doctests and check for failures
    doctest_results = doctest.testmod(m=gg)
    assert doctest_results.failed == 0


if __name__ == "__main__":
    test_example()
    test_doctest()
