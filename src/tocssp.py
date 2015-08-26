""" The MIT License (MIT)

    Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import itertools as it
import ctypes as ct

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "nova", "python"))
from nova.mdp import *
from nova.pomdp import *

from toc import *
from tocpomdp import *
from tocpath import *


class ToCSSP(MDP):
    """ A class which models the ToC SSP problem. """

    def __init__(self):
        """ The constructor of the ToC SSP class. """

        super().__init__()

        self.states = list()
        self.actions = list()
        self.observations = list()

    def _compute_rho(self, tocpomdp, sWeight):
        """ Compute the probabilities of reaching terminal `end result' states.

            Parameters:
                tocpomdp    --  The ToC POMDP model in question.
                sWeight     --  The weight (floored) of the state being considered.

            Returns:
                The NumPy array of size 3, corresponding to Pr(success),
                Pr(failure), and Pr(aborted).
        """

        return np.array([3.0, 2.0, 1.0]) / 6.0

    def create(self, toc, path):
        """ Create the MDP SSP given the ToC's path planning problem and the ToC problem itself.

            Parameters:
                toc     --  The transfer of control problems, a pair, two of them: h->v and v->h.
                path    --  The weighted directed graph: (V, E, w, v0, vg).
        """

        # TODO: Implement.
        tocpomdp = None

        calE = ["success", "failure", "aborted"]
        X = ["human", "vehicle"]
        Ad = ["left", "straight", "right"]
        Ac = ["keep", "switch"]

        self.states = list(it.product(path.V, X)) + list(calE)
        self.n = len(self.states)

        self.actions = list(it.product(Ad, Ac))
        self.m = len(self.actions)

        def direction_matches_edge(v, vp, ad):
            adIndex = Ad.index(ad)
            vEdges = [e for e in path.E if e[0] == v]
            vEdges = sorted(vEdges, key=lambda z: z[1])
            try:
                return vEdges[adIndex][1] == vp
            except Exception:
                return False

        def direction_exists(v, ad):
            adIndex = Ad.index(ad)
            vEdges = [e for e in path.E if e[0] == v]
            return adIndex < len(vEdges)

        # The maximum number of successor states is always bounded by 3, because
        # the uncertainty is only ever over the result of the ToC POMDP final
        # absorbing state. All other cases are deterministic.
        self.ns = 3

        S = [[[int(-1) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        T = [[[float(0.0) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            v = None
            x = None
            rho = None
            if state not in calE:
                v = state[0]
                x = state[1]
                rho = self._compute_rho(tocpomdp, int(path.w[v]))

            for a, action in enumerate(self.actions):
                ad = action[0]
                ac = action[1]

                cur = 0

                for sp, statePrime in enumerate(self.states):
                    vp = None
                    xp = None
                    if statePrime not in calE:
                        vp = statePrime[0]
                        xp = statePrime[1]

                    # Successful transfer of control over current state.
                    if state not in calE and statePrime not in calE and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v != path.vg and \
                            direction_exists(v, ad) and \
                            ac == "switch" and \
                            direction_matches_edge(v, vp, ad) and \
                            (v, vp) in path.E and x != xp:
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[0]
                        cur += 1

                    # Failed to transfer control over current state, but next state is autonomy-capable.
                    if state not in calE and statePrime not in calE and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v != path.vg and \
                            direction_exists(v, ad) and \
                            ac == "switch" and \
                            direction_matches_edge(v, vp, ad) and \
                            (v, vp) in path.E and x == xp:
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[1]
                        cur += 1

                    # Aborted transfer of control over current state.
                    if state not in calE and statePrime == "aborted" and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v != path.vg and \
                            direction_exists(v, ad) and \
                            ac == "switch":
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[2]
                        cur += 1

                    # Kept current controller. No transfer of control.
                    if state not in calE and statePrime not in calE and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v != path.vg and \
                            direction_exists(v, ad) and \
                            ac == "keep" and \
                            (v, vp) in path.E and x == xp and \
                            direction_matches_edge(v, vp, ad):
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Anytime at a non-problem and non-goal state, that the
                    # action doesn't work.. Basically when it is not available.
                    if state not in calE and statePrime == "failure" and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v != path.vg and \
                            not direction_exists(v, ad):
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Reached goal. Successfully reached success state!
                    if state not in calE and statePrime == "success" and \
                            not (v not in path.Vac and x == "vehicle") and \
                            v == path.vg:
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Reached a road that is not autonomy-capable, but vechile is in control. Death.
                    if state not in calE and statePrime == "failure" and \
                            v not in path.Vac and x == "vehicle":
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Absorbing states.
                    if state in calE and statePrime in calE and state == statePrime:
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)

        self.S = array_type_nmns_int(*np.array(S).flatten())
        self.T = array_type_nmns_float(*np.array(T).flatten())

        epsilon = 1.0
        wmax = max(path.w)

        R = [[0.0 for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            v = None
            x = None
            if state not in calE:
                v = state[0]
                x = state[1]

            for a, action in enumerate(self.actions):
                ad = action[0]
                ac = action[1]

                # Normal cost of being on a road.
                if state not in calE and not (v in path.Vap and x == "human"):
                    R[s][a] = -path.w[v]

                # Extra penalty included on a normal road which is autonomy-preferred but
                # the human is driving.
                if state not in calE and v in path.Vap and x == "human":
                    R[s][a] = -path.w[v] - epsilon

                # Constant for the absorbing 'failure' state.
                if state in calE:
                    R[s][a] = -wmax - epsilon

        self.Rmax = np.array(R).max()
        self.Rmin = np.array(R).min()

        array_type_nm_float = ct.c_float * (self.n * self.m)

        self.R = array_type_nm_float(*np.array(R).flatten())


if __name__ == "__main__":
    print("Performing ToCSSP Unit Test...")

    tocHtoV = ToC(randomize=(2, 2, 2, 5))
    tocVtoH = ToC(randomize=(2, 2, 2, 5))
    toc = (tocHtoV, tocVtoH)

    tocpath = ToCPath()
    tocpath.create()
    print(tocpath)

    tocssp = ToCSSP()
    tocssp.create(toc, tocpath)
    print(tocssp)

    print("Done.")

