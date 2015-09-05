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

        self.theta = None

    def _compute_rho(self, tocpomdp, sWeight):
        """ Compute the probabilities of reaching terminal `end result' states.

            Parameters:
                tocpomdp    --  The ToC POMDP model in question.
                sWeight     --  The weight (floored) of the state being considered.

            Returns:
                The NumPy array of size 3, corresponding to Pr(success),
                Pr(failure), and Pr(aborted).
        """

        #return np.array([1.0, 0.0, 0.0])
        #return np.array([3.0, 2.0, 1.0]) / 6.0
        return np.array([0.9, 0.0, 0.1])

    def create(self, toc, tocpomdp, path, controller=None):
        """ Create the MDP SSP given the ToC's path planning problem and the ToC problem itself.

            Parameters:
                toc         --  The transfer of control problems, a pair, two of them: h->v and v->h.
                tocpomdp    --  The POMDPs, a pair, two of them: h->v and v->h.
                path        --  The weighted directed graph: (V, E, w, v0, vg).
                controller  --  Optionally restrict the ToC SSP to only "human" or "vehicle" control.
                                Default is None, meaning both are included.
        """

        calE = ["success", "failure", "aborted"]
        X = ["human", "vehicle"]
        Ad = ["direction %i" % (i) for i in range(path.maxOutgoingDegree)]
        Ac = ["keep", "switch"]

        if controller == "human":
            X = ["human"]
            Ac = ["keep"]
        elif controller == "vehicle":
            X = ["vehicle"]
            Ac = ["keep"]

        self.states = list(it.product(path.V, X)) + list(calE)
        self.n = len(self.states)

        self.actions = list(it.product(Ad, Ac))
        self.m = len(self.actions)

        self.theta = {(v, ad): None for v, ad in it.product(path.V, Ad)}
        for v in path.V:
            vEdges = [e for e in path.E if e[0] == v]
            vEdges = sorted(vEdges, key=lambda z: z[1])
            for adIndex, e in enumerate(vEdges):
                self.theta[(v, Ad[adIndex])] = e[1]

        # The maximum number of successor states is always bounded by 3, because
        # the uncertainty is only ever over the result of the ToC POMDP final
        # absorbing state. All other cases are deterministic.
        self.ns = 3

        S = [[[int(-1) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        T = [[[float(0.0) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            v = None
            x = None
            if state not in calE:
                v = state[0]
                x = state[1]
                xIndex = X.index(x)

            for a, action in enumerate(self.actions):
                ad = action[0]
                ac = action[1]
                adIndex = Ad.index(ad)
                acIndex = Ac.index(ac)

                e = None
                aleph = None
                rho = None
                if state not in calE:
                    e = (v, self.theta[(v, ad)])
                    aleph = not (e not in path.Eac and x == "vehicle") and v != path.vg
                    try:
                        # This will only fail for e = theta returning a None for the impossible
                        # action. This essentially handles the A(s) case.
                        rho = self._compute_rho(tocpomdp[xIndex], int(path.w[e]))
                    except KeyError:
                        pass

                cur = 0

                for sp, statePrime in enumerate(self.states):
                    vp = None
                    xp = None
                    if statePrime not in calE:
                        vp = statePrime[0]
                        xp = statePrime[1]

                    # Successful transfer of control over current state.
                    if state not in calE and statePrime not in calE and \
                            aleph and \
                            self.theta[(v, ad)] == vp and \
                            ac == "switch" and \
                            x != xp:
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[0]
                        cur += 1

                    # Failed to transfer control over current state, but next state is autonomy-capable.
                    if state not in calE and statePrime not in calE and \
                            aleph and \
                            self.theta[(v, ad)] == vp and \
                            ac == "switch" and \
                            x == xp:
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[1]
                        cur += 1

                    # Aborted transfer of control over current state. This now means a self-loop, since
                    # the car pulls over safely and waits for the driver.
                    #if state not in calE and statePrime == "aborted" and \
                    #        aleph and \
                    #        self.theta[(v, ad)] is not None and \
                    #        ac == "switch":
                    if state not in calE and statePrime not in calE and \
                            aleph and \
                            self.theta[(v, ad)] is not None and \
                            ac == "switch" and \
                            state == statePrime:
                        S[s][a][cur] = sp
                        T[s][a][cur] = rho[2]
                        cur += 1

                    # Kept current controller. No transfer of control.
                    if state not in calE and statePrime not in calE and \
                            aleph and \
                            self.theta[(v, ad)] == vp and \
                            ac == "keep" and \
                            x == xp:
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Anytime at a non-problem and non-goal state, that the
                    # action doesn't work.. Basically when it is not available
                    # following A(s).
                    #if state not in calE and statePrime == "failure" and \
                    #        aleph and \
                    #        self.theta[(v, ad)] == None:
                    #    S[s][a][cur] = sp
                    #    T[s][a][cur] = 1.0
                    #    cur += 1

                    # Reached goal. Successfully reached success state!
                    if state not in calE and statePrime == "success" and \
                            self.theta[(v, ad)] is not None and \
                            not (e not in path.Eac and x == "vehicle") and \
                            v == path.vg:
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Reached a road that is not autonomy-capable, but vehicle is in control. Death.
                    if state not in calE and statePrime == "failure" and \
                            self.theta[(v, ad)] is not None and \
                            e not in path.Eac and x == "vehicle":
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # The action is not available here. This is the A(s) case. Death.
                    if state not in calE and statePrime == "failure" and \
                            self.theta[(v, ad)] is None:
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

        wmin = min(path.w.values())
        wmax = max(path.w.values())

        self.epsilon = 0.001
        self.gamma = 0.999 # 1.0 # TODO: Change once you implement LAO*.
        #self.horizon = max(10000, len(path.V) + 1) # This must be very large horizon, since wmin and wmax are very far apart.
        self.horizon = int(np.log(2.0 * (wmax - wmin) / (self.epsilon * (1.0 - self.gamma))) / np.log(1.0 / self.gamma))

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
                adIndex = Ad.index(ad)
                acIndex = Ac.index(ac)

                e = None
                if state not in calE:
                    e = (v, self.theta[(v, ad)])

                # Normal cost of being on a road.
                if state not in calE and self.theta[(v, ad)] is not None and not (e in path.Eap and x == "human"):
                    R[s][a] = -path.w[e]

                # Extra penalty included on a normal road which is autonomy-preferred but
                # the human is driving.
                if state not in calE and self.theta[(v, ad)] is not None and e in path.Eap and x == "human":
                    epsilonPenalty = 0.1
                    R[s][a] = -path.w[e] - epsilonPenalty #wmax

                # Handle the "not in A(s)" case, which sets the penalty to an impossibly large number.
                if state not in calE and self.theta[(v, ad)] is None:
                    R[s][a] = -wmax # * self.horizon

                ## Constant for the absorbing 'success' state.
                if state == "success":
                    R[s][a] = 0.0

                ## Constant for the absorbing 'failure' state.
                if state == "failure":
                    R[s][a] = -wmax # * self.horizon

                ## Constant for the absorbing 'aborted' state. This is no longer a reachable state.
                if state == "aborted":
                    R[s][a] = -wmax # * self.horizon

        self.Rmax = np.array(R).max()
        self.Rmin = np.array(R).min()

        array_type_nm_float = ct.c_float * (self.n * self.m)

        self.R = array_type_nm_float(*np.array(R).flatten())

        # The initial state is the initial state in the graph, with the human driving, except
        # if the vehicle is the only one that can control the ToC SSP.
        if controller is None or controller == "human":
            self.s0 = self.states.index((path.v0, "human"))
        else:
            self.s0 = self.states.index((path.v0, "vehicle"))

        # The goal state is the success state.
        self.goals = [self.states.index("success")]


if __name__ == "__main__":
    print("Performing ToCSSP Unit Test...")

    tocHtoV = ToC(randomize=(2, 2, 2, 5))
    tocVtoH = ToC(randomize=(2, 2, 2, 5))
    toc = (tocHtoV, tocVtoH)

    tocpomdpHtoV = None
    tocpomdpVtoH = None
    tocpomdp = (tocpomdpHtoV, tocpomdpVtoH)

    print("Creating the ToC Path... ", end='')
    sys.stdout.flush()

    tocpath = ToCPath()
    tocpath.random(numVertexes=10)

    print("Done.\nCreating the ToC SSP... ", end='')
    sys.stdout.flush()

    tocssp = ToCSSP()
    tocssp.create(toc, tocpomdp, tocpath)

    print("Done.\nSolving the ToC SSP... ", end='')
    sys.stdout.flush()

    V, pi = tocssp.solve()

    print("Done.\nResults:")

    print(tocpath)
    print(tocssp)
    print(V)
    print(pi.tolist())

    print("Done.")

