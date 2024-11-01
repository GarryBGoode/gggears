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
from defs import*
from function_generators import *
from scipy.optimize import root, minimize
import copy
from enum import Enum

class Curve():
    '''
        A class to represent a curve in space.
    '''
    def __init__(self,
                 curve_function: callable,
                 active=True,
                 t0=0,
                 t1=1,
                 params=None,
                 enable_vectorize=True):
        # active is used to show if curve is degenerate, such as a 0 length line
        if params is None:
            params = {}
        self.active = active
        # start and end of the curve in its natural parameter,
        # eg. an arc that starts and ends at 30 and 60 degrees
        self.t_0 = t0
        self.t_1 = t1
        # mathematical function to generate curve points
        self.function = curve_function
        # parameters of the curve function
        self.params = params

        # length of the curve is stored in the object, calculated by update_lengths()
        self.length=0
        self.len_approx_N = 101
        # used for length-parametrization conversion
        self.t2l = {'t': np.array([-1E6, 1E6]), 'l': np.array([-1E6, 1E6])}

        # vectorize feature is used to iterate over np.array inputs,
        # to eg. generate 100 points on the curve with np.linspace
        # if curve_function already handles array inputs, this can be disabled
        self.enable_vectorize=enable_vectorize
        if self.active:
            # update_lengths might be CP expensive so only done if active
            self.update_lengths()

    def __call__(self,p):
        t = self.p2t(p)
        if self.enable_vectorize:
            return vectorize(self.function)(t,**self.params)
        else:
            return self.function(t,**self.params)

    def p2t(self,p):
        # numpy interp is faster and more efficient but cannot extrapolate
        retval = np.interp(p, self.t2l['l'], self.t2l['t'])
        # custom interpolation function used for extrapolation
        if np.ndim(p)>0:
            if any(p<0):
                retval[p<0] = interpolate(p[p<0],
                                        self.t2l['l'][0],
                                        self.t2l['l'][1],
                                        self.t2l['t'][0],
                                        self.t2l['t'][1])
            if any(p>1):
                retval[p>1] = interpolate(p[p>1],
                                        self.t2l['l'][-1],
                                        self.t2l['l'][-2],
                                        self.t2l['t'][-1],
                                        self.t2l['t'][-2])
        else:
            if p<0:
                retval = interpolate(p,
                                        self.t2l['l'][0],
                                        self.t2l['l'][1],
                                        self.t2l['t'][0],
                                        self.t2l['t'][1])
            elif p>1:
                retval = interpolate(p,
                                        self.t2l['l'][-1],
                                        self.t2l['l'][-2],
                                        self.t2l['t'][-1],
                                        self.t2l['t'][-2])
        return retval

    def t2p(self, t):
        retval = np.interp(t, self.t2l['t'], self.t2l['l'])
        if np.ndim(t) > 0:
            if any(t < self.t_0):
                retval[t < self.t_0] = interpolate(t[t < self.t_0],
                                                   self.t2l['t'][0],
                                                   self.t2l['t'][1],
                                                   self.t2l['l'][0],
                                                   self.t2l['l'][1])
            if any(t > self.t_1):
                retval[t > self.t_1] = interpolate(t[t > self.t_1],
                                                   self.t2l['t'][-1],
                                                   self.t2l['t'][-2],
                                                   self.t2l['l'][-1],
                                                   self.t2l['l'][-2])
        else:
            if t < self.t_0:
                retval = interpolate(t,
                                     self.t2l['t'][0],
                                     self.t2l['t'][1],
                                     self.t2l['l'][0],
                                     self.t2l['l'][1])
            elif t > self.t_1:
                retval = interpolate(t,
                                     self.t2l['t'][-1],
                                     self.t2l['t'][-2],
                                     self.t2l['l'][-1],
                                     self.t2l['l'][-2])
        return retval

    def update_lengths(self):
        t_range = np.linspace(self.t_0, self.t_1, self.len_approx_N)
        if self.enable_vectorize:
            value_array = vectorize(self.function)(t_range, **self.params)
        else:
            value_array = self.function(t_range, **self.params)

        curve_len = np.cumsum(np.append(np.array([0]),
                                        np.linalg.norm(value_array[1:, :] -
                                                       value_array[:-1, :], axis=1)))
        self.length = curve_len[-1]
        self.t2l['l'] = curve_len / self.length
        self.t2l['t'] = t_range

    def reverse(self):
        self.t_0 , self.t_1 = self.t_1 , self.t_0
        self.t2l['t']=np.flip(self.t2l['t'])
        return self

    def derivative(self,t, direction=0, n=1, delta=DELTA):
        '''
        Numerically approximate the curve gradient at t.
        t: curve parameter where the derivative is evaluated at.
        direction: 1: forward, -1: backward, 0: balanced derivative.
        n: derivative order (n=2: second derivative, etc.)
            0 and negative value (integral) does not work, this is not calculus.
        delta: small value used for numeric differentiation.
            Hint: consider using larger deltas for higher order derivatives,
            it is easy to run into floating point issues.
        '''
        def numeric_diff(function,t,direction,delta):
            if direction==0:
                return (function(t+delta)-function(t-delta))/2/delta
            elif direction>0:
                return (function(t+delta)-function(t))/delta
            else:
                return (function(t)-function(t-delta))/delta
        if n<=1:
            return numeric_diff(self.function,t,direction,delta)
        else:
            return numeric_diff(self.derivative,t,direction,n-1,delta)

    def cut(self,t):
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)
        curve1.set_end_on(t)
        curve2.set_start_on(t)
        return curve1,curve2

    def copy(self):
        return copy.deepcopy(self)

    def set_start_and_end_on(self,t0,t1):
        self.t_0 = self.p2t(t0)
        self.t_1 = self.p2t(t1)
        self.update_lengths()

    def set_start_on(self,t0):
        self.t_0 = self.p2t(t0)
        self.update_lengths()

    def set_end_on(self,t1):
        self.t_1 = self.p2t(t1)
        self.update_lengths()

    def __add__(self, other:'Curve'):
        return Curve(lambda t: self(t) + other(t),
                     active=self.active and other.active,
                     t0=0, t1=1, params={}, enable_vectorize=False)

    def __subtract__(self, other:'Curve'):
        return Curve(lambda t: self(t) - other(t),
                     active=self.active and other.active,
                     t0=0, t1=1, params={}, enable_vectorize=False)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Curve(lambda t: self(t) * other,
                         active=self.active,
                         t0=self.t_0, t1=self.t_1,
                         params=self.params,
                         enable_vectorize=self.enable_vectorize)
        else:
            raise TypeError(
                "Multiplication with type {} is not supported".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Curve(lambda t: self(t) / other,
                         active=self.active,
                         t0=self.t_0, t1=self.t_1,
                         params=self.params,
                         enable_vectorize=self.enable_vectorize)
        else:
            raise TypeError(
                "Division with type {} is not supported".format(type(other)))

    @property
    def is_closed(self):
        return np.linalg.norm(self(0)-self(1))<DELTA/100

class CurveChain(Curve):
    '''
        A class that chains together multiple curves while also being callable,
          and can be used as a single curve.
    '''
    def __init__(self,*curves : 'Curve', active=True, **kwargs):
        self.curves = [*curves]
        self._active = active
        self.update_lengths()
        # super().__init__() is actually not needed.
        # the internal data of a curve relates to handling its function(),
        #   a curvechain has no function() of its own
        # all handling (lengths, etc.) is deferred to the contained curves,
        #   curvechain handles navigating the chain

    @property
    def active(self):
        return self._active and any([curve.active for curve in self.curves])

    @active.setter
    def active(self, value):
        self._active = value

    def update_lengths(self):
        for curve in self.curves:
            if curve.active:
                curve.update_lengths()

    @property
    def num_curves(self):
        return len(self.curves)

    @property
    def length_array(self):
        return np.array([curve.length if curve.active else 0 for curve in self.curves])

    @property
    def length(self):
        return np.sum(self.length_array)

    # these might still show up some time
    def p2t(self,p):
        return p
    def t2p(self,t):
        return t

    @property
    def idx_active_min(self):
        try:
            return [curve.active for curve in self.curves].index(True)
        except ValueError:
            return len(self.curves)

    @property
    def idx_active_max(self):
        try:
            return len(self.curves) - \
                [curve.active for curve in reversed(self.curves)].index(True) - 1
        except ValueError:
            return -1

    def get_p_index(self,t):
        '''find which curve index t belongs to and how far along it is in the curve'''
        length_portions = self.get_length_portions()
        idx = np.searchsorted(length_portions,t)

        if idx>self.idx_active_max+1:
            idx=self.idx_active_max+1
        if idx<self.idx_active_min+1:
            idx=self.idx_active_min+1

        if (length_portions[idx] - length_portions[idx-1]) != 0:
            t3 = (t-length_portions[idx-1]) / \
                (length_portions[idx] - length_portions[idx-1])
        else:
            t3 = 0.5
        return idx-1,t3

    def get_t_for_index(self,idx):
        length_portions = self.get_length_portions()
        return length_portions[idx], length_portions[idx+1]

    def get_length_portions(self):
        length_sum = np.cumsum(self.length_array)
        length_portions = np.concatenate([[0],length_sum/self.length])
        return length_portions

    def curve_list_eval(self,t):

        idx, t2 = self.get_p_index(t)
        point_out = self.curves[idx](t2)
        return point_out

    def __call__(self,p):
        # no need of params of **params for the function
        # the functions technically belong to members,
        #   function of the chain makes little sense
        return vectorize(self.curve_list_eval)(p)

    def get_curves(self, lerp_inactive=False):
        curve_list = []
        for curve in self.curves:
            if isinstance(curve,CurveChain):
                curve_list.extend(curve.get_curves())
            else:
                curve_list.append(curve)

        if lerp_inactive:
            for k in range(len(curve_list)):
                ii = k%len(curve_list)
                ii_p1 = (k+1)%len(curve_list)
                ii_m1 = (k-1)%len(curve_list)
                if not curve_list[ii].active:
                    curve_list[ii] = LineCurve(curve_list[ii_m1](1),
                                               curve_list[ii_p1](0),
                                               active=False)
        return curve_list

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, index):
        return self.curves[index]

    def __setitem__(self, index, value):
        self.curves[index] = value


    def __delitem__(self, index):
        self.curves[index] = []

    def __iter__(self):
        return iter(self.curves)

    # TODO: later
    # def __contains__(self, item):
    #     pass
    # def __mul__(self, other):
    #     pass

    # def __add__(self, other):
    #     self.curves = [self.curves, other]
    #     self.update_lengths()

    def append(self, value):
        self.curves.append(value)
        self.update_lengths()

    def extend(self, iterable):
        self.curves = [*self.curves, *iterable]
        self.update_lengths()

    def insert(self, index, value):
        self.curves.insert(index,value)
        self.update_lengths()

    def pop(self, index=-1):
        ret_curve = self.curves.pop(index)
        self.update_lengths()
        return ret_curve

    def clear(self):
        self.curves.clear()

    def reverse(self):
        self.curves.reverse()
        for curve in self.curves:
            curve.reverse()

    def set_start_and_end_on(self, t0, t1, preserve_inactive_curves=False):
        idx,t2 = self.get_p_index(t0)
        if preserve_inactive_curves:
            for curve in self.curves[:idx]:
                curve.active=False
            self.curves[idx].set_start_on(t2)
        else:
            self.curves = self.curves[idx:]
            self.curves[0].set_start_on(t2)

        idx,t2 = self.get_p_index(t1)
        if preserve_inactive_curves:
            for curve in self.curves[idx+1:]:
                curve.active=False
            self.curves[idx].set_end_on(t2)
        else:
            self.curves = self.curves[:idx+1]
            self.curves[-1].set_end_on(t2)
        self.update_lengths()


    def set_start_on(self,t0, preserve_inactive_curves=False):
        idx,t2 = self.get_p_index(t0)
        if preserve_inactive_curves:
            for curve in self.curves[:idx]:
                curve.active=False
            self.curves[idx].set_start_on(t2)
        else:
            self.curves = self.curves[idx:]
            self.curves[0].set_start_on(t2)
        self.update_lengths()

    def set_end_on(self,t1, preserve_inactive_curves=False):
        idx,t2 = self.get_p_index(t1)
        if preserve_inactive_curves:
            for curve in self.curves[idx+1:]:
                curve.active=False
            self.curves[idx].set_end_on(t2)
        else:
            self.curves = self.curves[:idx+1]
            self.curves[-1].set_end_on(t2)

        self.update_lengths()

    def cut(self,t,preserve_inactive_curves=False):
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)

        curve2.set_start_on(t,preserve_inactive_curves)
        curve1.set_end_on(t,preserve_inactive_curves)
        curve1.update_lengths()
        curve2.update_lengths()

        return curve1,curve2

    def fillet_at_locations(self,radius,locations=[0.5,0.6]):
        arc, t1, t2 = fillet_curve(self,radius,start_locations=locations)
        curve1 = copy.deepcopy(self)
        curve2 = copy.deepcopy(self)
        curve1.set_end_on(t1)
        curve2.set_start_on(t2)
        return CurveChain(*curve1.get_curves(),arc,*curve2.get_curves())

    def fillet(self,radius,location=0.5):
        return self.fillet_at_locations(radius,
                                        [location+radius/self.length,
                                         location-radius/self.length])

    def fillet_at_index(self,radius,index):
        location = self.get_t_for_index(index%self.num_curves)[1]
        return self.fillet(radius,location)

    @property
    def continuity_list(self):
        out_list = []
        for k in range(self.num_curves):
            diff = np.linalg.norm(self.curves[k](1) -  \
                                  self.curves[(k+1)%self.num_curves](0))
            out_list.append(diff < DELTA/100 or
                            not self.curves[k].active or
                            not self.curves[(k+1)%self.num_curves].active)
        return out_list
    @property
    def is_continuous(self):
        return all(self.continuity_list[:-1])

    @property
    def is_closed(self):
        return np.linalg.norm(self(0)-self(1))<DELTA/100 and self.is_continuous


class IntersectMethod(Enum):
    EQUALITY = 1
    MINDISTANCE = 2

def find_curve_intersect(curve1: Curve,
                         curve2: Curve,
                         guess=[0.5,0.5],
                         method:'IntersectMethod'=IntersectMethod.EQUALITY):
    if method == IntersectMethod.EQUALITY:
        res = root(lambda t: curve1(t[0])-curve2(t[1]),np.array([guess[0],guess[1],0]))
    elif method == IntersectMethod.MINDISTANCE:
        def minfunc(t):
            diff = (curve1(t[0])-curve2(t[1]))/DELTA
            return np.dot(diff,diff)
        res = minimize(minfunc,np.array([guess[0],guess[1]]))
    return res



def find_curve_line_intersect(curve,
                              offset=ORIGIN,
                              line_direction=RIGHT,
                              guess=0):
    res = root(lambda t: np.linalg.norm(np.cross((curve(t)-offset),
                                                 line_direction)),guess)
    return res

def find_curve_plane_intersect(curve,plane_normal = OUT,offset=ORIGIN,  guess=0):
    def target_func(t):
        val =  np.dot((curve(t[0])-offset),plane_normal)
        return val
    res = root(target_func,guess)
    return res

def find_curve_nearest_point (curve: Curve, point, guesses = [0.5]):
    results = []
    for guess in guesses:
        results.append(minimize(lambda t: np.dot((curve(t[0])-point),
                                                 (curve(t[0])-point)),guess))
    pass

    return min(results, key=lambda res: res.fun).x[0]

def fit_bezier_hermite(target_curve: Curve):
    points = np.zeros((4,3))
    points[0] = target_curve(0)
    points[3] = target_curve(1)
    points[1] = points[0] + target_curve.derivative(0,1, delta=1E-4)/3
    points[2] = points[3] - target_curve.derivative(1,-1, delta=1E-4)/3
    return points

def fit_bezier_optim(target_curve: Curve):

    def point_allocator(x):
        points = np.zeros((4,3))
        points[0] = target_curve(0)
        points[3] = target_curve(1)

        points[1] = np.array([x[0],x[1],x[2]])
        points[2] = np.array([x[3],x[4],x[5]])
        return points

    def inverse_allocator(points):
        x = np.zeros((6))
        x[0], x[1], x[2] = points[1][:]
        x[3], x[4], x[5] = points[2][:]
        return x

    # the 2 edge points are enforced
    n_points=9
    tvals = np.linspace(0,1,n_points+2)[1:-1]
    initpoints = target_curve(np.linspace(0,1,4))
    init_guess = inverse_allocator(initpoints)

    BZcurve = Curve(bezier,params={'points':initpoints,})
    BZcurve.len_approx_N=n_points*2
    target_points = target_curve(tvals)

    def cost_func(x):
        points = point_allocator(x)
        BZcurve.params['points'] = points
        bz_tvals = np.array([find_curve_nearest_point(BZcurve,
                                                      target_point,
                                                      [tval]) \
                             for target_point,tval in zip(target_points,tvals)])
        return np.sum(np.linalg.norm(target_points-BZcurve(bz_tvals),axis=1))

    def cost_func2(x):
        points = point_allocator(x)
        BZcurve.params['points'] = points
        BZcurve.update_lengths()
        return np.sum(np.linalg.norm(target_points-BZcurve(tvals),axis=1))
    sol = minimize(cost_func2,init_guess)
    point_out = point_allocator(sol.x)
    return sol,point_out


def fit_nurb_points(target_points: np.ndarray, n_points=4, force_2D=False):
    N_target = target_points.shape[0]

    N_Dim = 2 if force_2D else 3
    scaler = 1

    def point_allocator(x):
        points = np.zeros((n_points,N_Dim))
        points[0] = target_points[0,:N_Dim]*scaler
        points[-1] = target_points[-1,:N_Dim]*scaler
        weights = np.ones((n_points))
        for k in range(1,n_points-1):
            ii = N_Dim*(k-1)
            points[k] = np.array([x[ii+j] for j in range(N_Dim)])
            weights[k] = x[N_Dim*(n_points-2)+k-1]
        # t = np.linspace(0,1,N_target)
        t = x[(N_Dim+1)*(n_points-2):(N_Dim+1)*(n_points-2)+N_target]
        return points, weights, t

    def inverse_allocator(points,weights,t):
        x = np.zeros((N_Dim+1)*(n_points-2)+N_target)

        for k in range(1,n_points-1):
            ii = N_Dim*(k-1)
            for j in range(N_Dim):
                x[ii+j] = points[k,j]
            x[N_Dim*(n_points-2)+k-1] = weights[k]

        x[(N_Dim+1)*(n_points-2):(N_Dim+1)*(n_points-2)+N_target] = t
        return x

    initguess_x = inverse_allocator(points=bezierdc(t=np.linspace(0,1,n_points),
                                                    points=target_points),
                                                    weights=np.ones(n_points),
                                                    t=np.linspace(0,1,N_target))

    def cost_fun(x):
        points,weights,t = point_allocator(x)
        diff = target_points[:,:N_Dim]-nurbezier(t,points,weights)
        return np.sum(diff**2)

    sol = minimize(cost_fun,initguess_x,method='BFGS')

    points,weights,t = point_allocator(sol.x)

    return sol, points,weights

def fit_nurb_optim(target_curve: Curve, n_points=4, N_Dim=3):

    scaler = 1
    # N_Dim = 2 if force_2D else 3


    def point_allocator(x):
        points = np.zeros((n_points,N_Dim))
        points[0] = target_curve(0)[:N_Dim]*scaler
        points[-1] = target_curve(1)[:N_Dim]*scaler
        weights = np.ones((n_points))
        for k in range(1,n_points-1):
            ii = N_Dim*(k-1)
            points[k] = np.array([x[ii+j] for j in range(N_Dim)])
            weights[k] = x[N_Dim*(n_points-2)+k-1]

        return points, weights

    def inverse_allocator(points,weights):
        x = np.zeros(((n_points-2)*4))
        for k in range(1,n_points-1):
            ii = N_Dim*(k-1)
            for j in range(N_Dim):
                x[ii+j] = points[k,j]
            x[N_Dim*(n_points-2)+k-1] = weights[k]

        return x

    n_fit_points = (n_points-2) * 7
    # the 2 edge points are enforced
    tvals = np.linspace(0,1,n_fit_points+2)[1:-1]
    initpoints = target_curve(np.linspace(0,1,n_points))[:,:N_Dim]*scaler
    initweights = np.ones((n_points))
    init_guess = inverse_allocator(initpoints,initweights)

    BZcurve = Curve(nurbezier,params={'points':initpoints,'weights':initweights},
                    enable_vectorize=False)
    BZcurve.len_approx_N=n_fit_points*3
    target_curve.len_approx_N = n_fit_points*3
    target_curve.update_lengths()
    target_points = (target_curve(tvals)*scaler)[:,:N_Dim]

    def cost_func(x):
        points,weights = point_allocator(x)
        BZcurve.params['points'] = points
        BZcurve.params['weights'] = weights
        bz_tvals = np.array([find_curve_nearest_point(BZcurve,target_point,[tval]) \
                             for target_point,tval in zip(target_points,tvals)])
        return np.sum(np.linalg.norm(target_points-BZcurve(bz_tvals),axis=1))

    def cost_func2(x):
        points,weights = point_allocator(x)
        BZcurve.params['points'] = points
        BZcurve.params['weights'] = weights
        BZcurve.update_lengths()
        return np.sum(np.linalg.norm(target_points-BZcurve(tvals),axis=1))

    sol = minimize(cost_func2,init_guess,method='Newton-CG')
    point_out, weight_out = point_allocator(sol.x)

    # if force_2D:
    #     point_out = np.pad(point_out,((0,0),(0,3-N_Dim)))

    return sol,point_out/scaler, weight_out


def fit_nurb_optim2(target_curve: Curve, n_points=4, force_2D=False, samp_ratio=1.5):
    N_Dim = 2 if force_2D else 3

    scaler = 1
    # each bezier point brings 4 DoF unknown (xyz + w)
    # each eval point uses 2 DoF known (xyz - t)
    # on average at least 2 eval points are needed per bezier point

    n_fit_points = int(np.ceil((n_points) * 2 * samp_ratio))
    # the 2 edge points are enforced
    tvals = np.linspace(0,1,n_fit_points)
    target_points = (target_curve(tvals)*scaler)[:,:N_Dim]
    sol,points,weights = fit_nurb_points(target_points,n_points, force_2D=force_2D)
    if force_2D:
        points= np.pad(points,
                       [(0,0),(0,1)],
                       constant_values=np.mean(target_curve(tvals)[:,2]))

    return sol, points, weights

def convert_curve_nurbezier(input_curve: Curve, skip_inactive=True,**kwargs):
    if hasattr(input_curve,'__iter__'):
        out_curve_list = []
        for curve in input_curve:
            if curve.active or not skip_inactive:
                out_curve_list.append(convert_curve_nurbezier(curve,**kwargs))
        return NURBSCurve(*out_curve_list)

    else:
        if isinstance(input_curve,LineCurve):
            bz_points = np.array([input_curve(0),input_curve(1)])
            bz_weights = np.ones((2))
        elif isinstance(input_curve,ArcCurve):
            if abs(input_curve.angle)<PI:
                bz_points,bz_weights = calc_nurbezier_arc(input_curve(0),
                                                          input_curve(1),
                                                          input_curve.center)
            else:
                sol, bz_points, bz_weights = fit_nurb_optim2(input_curve,**kwargs)
        else:
            sol, bz_points, bz_weights = fit_nurb_optim2(input_curve,**kwargs)
        # out_curve = Curve(nurbezier,params={'points':bz_points,'weights':bz_weights})
        out_curve = NurbCurve(bz_points,bz_weights,active=input_curve.active)

    return out_curve

def calc_tangent_arc(curve1:Curve,
                     curve2:Curve,
                     radius:float,
                     start_locations=[1,0],
                     method=IntersectMethod.EQUALITY):
    def calc_centers(t1,t2):
        p1 = curve1(t1)
        p2 = curve2(t2)
        tan1 = normalize_vector(curve1.derivative(t1))
        tan2 = normalize_vector(curve2.derivative(t2))

        arc_axis = np.cross(tan1,tan2)
        angle = np.linalg.norm(arc_axis)
        arc_axis = arc_axis/angle

        normal1 = np.cross(tan1,arc_axis)
        normal2 = np.cross(tan2,arc_axis)

        center1 = p1-normal1*radius
        center2 = p2-normal2*radius
        return center1,center2

    def cost_fun(x):
        t1 = start_locations[0]-x[0]
        t2 = start_locations[1]+x[1]
        center1,center2 = calc_centers(t1,t2)
        return center1-center2
    def cost_fun2(x):
        t1 = start_locations[0]-x[0]
        t2 = start_locations[1]+x[1]
        center1,center2 = calc_centers(t1,t2)
        return np.dot(center1-center2,center1-center2)

    if method == IntersectMethod.EQUALITY:
        sol1 = root(cost_fun,np.array([0,0,0]))
    elif method == IntersectMethod.MINDISTANCE:
        sol1 = minimize(cost_fun2,np.array([0,0]))

    t1 = start_locations[0]-sol1.x[0]
    t2 = start_locations[1]+sol1.x[1]
    center1,center2 = calc_centers(t1,t2)
    center = (center1 + center2)/2

    arc = ArcCurve.from_2_point_center(p0=curve1(t1),
                                       p1=curve2(t2),
                                       center=center)
    return arc, t1, t2, sol1


def fillet_curve(input_curves: CurveChain,
                 radius:float,
                 start_locations=[0.5,0.5]):

    def calc_centers(t1,t2):
        p1 = input_curves(t1)
        p2 = input_curves(t2)
        tan1 = normalize_vector(input_curves.derivative(t1))
        tan2 = normalize_vector(input_curves.derivative(t2))

        arc_axis = np.cross(tan1,tan2)
        angle = np.linalg.norm(arc_axis)
        arc_axis = arc_axis/angle

        normal1 = np.cross(tan1,arc_axis)
        normal2 = np.cross(tan2,arc_axis)

        center1 = p1-normal1*radius
        center2 = p2-normal2*radius
        return center1,center2
    def cost_fun(x):
        t1 = start_locations[0]-x[0]
        t2 = start_locations[1]+x[1]
        center1,center2 = calc_centers(t1,t2)
        # return [np.linalg.norm(center1-center2),np.linalg.norm(center1-center2)]
        return center1-center2


    sol1 = root(cost_fun,np.array([0,0,0]))
    if not sol1.success:
        # try again with different first guess
        guess2 = radius / input_curves.length
        sol1 = root(cost_fun,np.array([guess2,guess2,0]))
    if not sol1.success:
        # try again with different first guess
        guess2 = radius / input_curves.length
        sol1 = root(cost_fun,np.array([DELTA,DELTA,0]))
    t1 = start_locations[0]-sol1.x[0]
    t2 = start_locations[1]+sol1.x[1]

    center1,center2 = calc_centers(t1,t2)
    center = (center1 + center2)/2


    arc = ArcCurve.from_2_point_center(p0=input_curves(t1),
                                       p1=input_curves(t2),
                                       center=center)

    return arc, t1, t2


class LineCurve(Curve):
    def __init__(self,
                 p0=ORIGIN,
                 p1=ORIGIN,
                 active=True,
                 enable_vectorize=False):
        self.p0 = p0
        self.p1 = p1
        super().__init__(self.line_func,
                         active,
                         t0=0,
                         t1=1,
                         params={},
                         enable_vectorize=enable_vectorize)

    def line_func(self,t):
        if isinstance(t,np.ndarray):
            return (1-t)[:,np.newaxis] * self.p0[np.newaxis,:] + \
                       t[:,np.newaxis] * self.p1[np.newaxis,:]
        else:
            return self.p0*(1-t) + self.p1*t

    def update_lengths(self):
        self.length = np.linalg.norm(self.p1-self.p0)
        self.t2l['l'] = np.array([0,1])
        self.t2l['t'] = np.array([self.t_0,self.t_1])

class ArcCurve(Curve):
    def __init__(self,
                 radius=1,
                 angle=PI/2,
                 center=ORIGIN,
                 yaw=0,
                 pitch=0,
                 roll=0,
                 active=True):
        self._radius = radius
        self._angle = angle
        self._center = center
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll
        self._rotmat = self.gen_rotmat()
        super().__init__(self.arcfunc,active=active,enable_vectorize=False)


    def gen_rotmat(self):
        return scp_Rotation.from_euler('zyx',
                                       [self._yaw,
                                        self._pitch,
                                        self._roll]).as_matrix()

    def arcfunc(self,t):
        rot_arc = scp_Rotation.from_euler('z',self._angle*t).as_matrix()
        points =  self._rotmat @ rot_arc @ ( RIGHT*self._radius) + self._center
        return points

    def update_lengths(self):
        self.length = self.radius * self.angle
        self.t2l['l'] = np.array([0,1])
        self.t2l['t'] = np.array([self.t_0,self.t_1])

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self,value):
        self._radius = value

    @property
    def r(self):
        return self._radius

    @property
    def angle(self):
        return self._angle

    @property
    def center(self):
        return self._center

    @property
    def p0(self):
        return self(0)

    @property
    def p1(self):
        return self(1)

    @property
    def curvature(self):
        return 1/self._radius

    @property
    def revolutions(self):
        return self._angle//(PI*2)

    @property
    def axis(self):
        return self._rotmat @ OUT



    @classmethod
    def from_2_point_center(cls,
                            p0=RIGHT,
                            p1=UP,
                            center=ORIGIN,
                            revolutions=0,
                            active=True):
        r = np.linalg.norm(p0-center)
        x = normalize_vector(p0-center)
        z = normalize_vector(np.cross(p0-center,p1-center))
        y = np.cross(z,x)
        R = np.transpose(np.array([x,y,z]))
        yaw,pitch,roll = scp_Rotation.from_matrix(R).as_euler('zyx')
        return cls(radius=r,
                   angle=angle_between_vectors(p0-center,p1-center) + revolutions*PI*2,
                   center=center,
                   yaw=yaw,
                   pitch=pitch,
                   roll=roll,
                   active=active)

    @classmethod
    def from_2_point_curvature(cls,
                               p0=RIGHT,
                               p1=UP,
                               curvature=1,
                               axis=OUT,
                               revolutions=0,
                               active=True):
        r = 1/curvature
        if any((p1-p0)!=0):
            dp = normalize_vector(p1-p0)
        else:
            # this is baaad but can't deal with it rn
            dp = UP

        axis = normalize_vector(axis-np.dot(axis,dp)*dp)

        if abs(r)<np.linalg.norm(p1-p0)/2:
            r=np.linalg.norm(p1-p0)/2*np.sign(r)

        h = np.sqrt(r**2-(np.linalg.norm(p1-p0)/2)**2)*np.sign(r)

        center = (p0+p1)/2-np.cross(dp,axis)*h
        return cls.from_2_point_center(p0=p0,
                                       p1=p1,
                                       center=center,
                                       revolutions=revolutions,
                                       active=active)

    @classmethod
    def from_point_center_angle(cls,
                                p0=RIGHT,
                                center=ORIGIN,
                                angle=PI/2,
                                axis=OUT,
                                active=True):
        r = np.linalg.norm(p0-center)
        x = normalize_vector(p0-center)
        y = normalize_vector(np.cross(axis,x))
        z = np.cross(x,y)
        R = np.transpose(np.array([x,y,z]))
        yaw,pitch,roll = scp_Rotation.from_matrix(R).as_euler('zyx')
        return cls(radius=r,
                   angle=angle,
                   center=center,
                   yaw=yaw,
                   pitch=pitch,
                   roll=roll,
                   active=active)

    @classmethod
    def from_radius_center_angle(cls,
                                 radius=1,
                                 center=ORIGIN,
                                 angle_start=0,
                                 angle_end=PI/2,
                                 axis=OUT,
                                 active=True):
        z = axis
        x = np.cross(UP,z)
        if np.linalg.norm(x)==0:
            x = OUT
        else:
            x = normalize_vector(x)
        y = np.cross(z,x)
        R = np.transpose(np.array([x,y,z]))
        # in theory this should not return any yaw value
        yaw,pitch,roll = scp_Rotation.from_matrix(R).as_euler('zyx')
        return cls(radius=radius,
                   angle=angle_end-angle_start,
                   center=ORIGIN,
                   yaw=angle_start,
                   pitch=pitch,
                   roll=roll,
                   active=active)


class InvoluteCurve(Curve):
    def __init__(self,
                 r=1,
                 t0=0,
                 t1=1,
                 angle=0,
                 v_offs=ORIGIN,
                 z_offs=0,
                 active=True,
                 enable_vectorize=True):
        self.r=r
        self.angle=angle
        self.v_offs=v_offs
        self.z_offs=z_offs
        super().__init__(lambda t: involute_circle(t,
                                                   r=self.r,
                                                   angle=self.angle,
                                                   v_offs=self.v_offs,
                                                   z_offs=self.z_offs),
                         active=active,
                         t0=t0,
                         t1=t1,
                         params={},
                         enable_vectorize=enable_vectorize)

class SphericalInvoluteCurve(Curve):
    def __init__(self,
                 r=1,
                 t0=0,
                 t1=1,
                 angle=0,
                 c_sphere=1,
                 v_offs=ORIGIN,
                 z_offs=0,
                 active=True,
                 enable_vectorize=True):
        self.r=r
        self.angle=angle
        self.c_sphere = c_sphere
        self.v_offs=v_offs
        self.z_offs=z_offs
        super().__init__(lambda t: involute_sphere(t,
                                                   r=self.r,
                                                   C=self.c_sphere,
                                                   angle=self.angle,
                                                   v_offs=self.v_offs,
                                                   z_offs=self.z_offs),
                        active,
                        t0,
                        t1,
                        params={},
                        enable_vectorize=enable_vectorize)
    @property
    def center(self):
        return (np.sqrt(self.R**2-self.r**2)+self.z_offs) * OUT
    @property
    def R(self):
        return 1/self.c_sphere

class TransformedCurve(Curve):
    def __init__(self,
                 transform: callable,
                 curve: Curve,
                 params=None,
                 enable_vectorize=False):
        self.target_curve = curve
        self.transform_method = transform
        if isinstance(curve,CurveChain):
            TransformedCurveChain.__init__(self,transform,curve)
        else:
            super().__init__(lambda t: self.apply_transform(self.target_curve(t)),
                            active=self.target_curve.active,
                            t0=0,
                            t1=1,
                            params=params,
                            enable_vectorize=enable_vectorize)

    def apply_transform(self,point):
        return self.transform_method(point,**self.params)

class TransformedCurveChain(CurveChain):
    def __init__(self,
                 transform: callable,
                 curve: CurveChain,
                 params=None,
                 enable_vectorize=False):
        self.target_curve = curve
        self.transform_method = transform

        super().__init__(* [TransformedCurve(transform,
                                             curve,
                                             params=params,
                                             enable_vectorize=enable_vectorize) \
                             for curve in self.target_curve])

    def apply_transform(self,point):
        return self.transform_method(point,**self.params)


class MirroredCurve(TransformedCurve,TransformedCurveChain):
    def __init__(self, curve: Curve, plane_normal=RIGHT, center=ORIGIN):
        self.plane_normal = normalize_vector(plane_normal)
        self.center=center
        def mirror_func(p):
            p2 = p-self.center
            h = np.dot(p2,self.plane_normal)
            if hasattr(h,'__iter__'):
                return p2-2* h[:,np.newaxis] * self.plane_normal[np.newaxis,:] \
                      + self.center
            else:
                return p2-2* h * self.plane_normal + self.center
        super().__init__(mirror_func, curve)


class RotatedCurve(TransformedCurve,TransformedCurveChain):
    def __init__(self, curve: Curve, angle=0, axis=OUT, center=ORIGIN):
        self.axis = normalize_vector(axis)
        self.angle = angle
        self.center = center
        def rotate_func(p):
            p2 = p-self.center
            return scp_Rotation.from_rotvec(self.angle*self.axis).apply(p2) \
                + self.center
        super().__init__(rotate_func, curve)


class NurbCurve(Curve):
    def __init__(self,
                 points,
                 weights,
                 active=True):
        self.points=points
        self.weights=weights
        super().__init__(lambda t: nurbezier(t,self.points,self.weights),
                         active,
                         t0=0,
                         t1=1,
                         params={},
                         enable_vectorize=False)
    @property
    def n_points(self):
        return self.points.shape[0]


class NURBSCurve(CurveChain):
    def __init__(self,*curves : 'NurbCurve', active=True, **kwargs):
        self.curves = [*curves]
        self._active = active
        self.update_lengths()

    @property
    def n_points(self):
        return sum([curve.n_points for curve in self.curves])
    @property
    def points(self):
        out_arr = np.concatenate([curve.points[:-1] for curve in self.curves])
        out_arr = np.append(out_arr,self.curves[-1].points[-1,np.newaxis],axis=0)
        return out_arr

    @property
    def weights(self):
        out_arr = np.concatenate([curve.weights[:-1] for curve in self.curves])
        out_arr = np.append(out_arr,self.curves[-1].weights[-1,np.newaxis],axis=0)
        return out_arr

    @property
    def knots(self):
        out_arr = np.array([0])
        for curve in self.curves:
            out_arr = np.append(out_arr,curve.n_points-1+out_arr[-1])
        return out_arr

    def enforce_continuity(self):
        for curve1,curve2 in zip(self.curves[:-1],self.curves[1:]):
            midpoints = (curve1.points[-1]+curve2.points[0])/2
            midweights = (curve1.weights[-1]+curve2.weights[0])/2
            curve1.points[-1] = midpoints
            curve2.points[0] = midpoints
            curve1.weights[-1] = midweights
            curve2.weights[0] = midweights
        if self.is_closed:
            midpoints = (self.curves[-1].points[-1]+self.curves[0].points[0])/2
            midweights = (self.curves[-1].weights[-1]+self.curves[0].weights[0])/2
            self.curves[-1].points[-1] = midpoints
            self.curves[0].points[0] = midpoints
            self.curves[-1].weights[-1] = midweights
            self.curves[0].weights[0] = midweights
