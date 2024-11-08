'''
Copyright 2024 Gergely Bencsik
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import gggears.gggears_convert as ggc
import gggears.gggears_core as gg
import numpy as np
import matplotlib.pyplot as plt

n_z = 18
gamma = 0.5*np.pi/2
axis = gg.OUT
m=2

param = gg.InvoluteGearParamManager2()
gear = gg.InvoluteGear(param)
profile = gear.curve_gen_at_z(0).generate_profile_closed()

print(profile(0))
