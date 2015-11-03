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

sys.path.append(thisFilePath)
from additional_functions import *

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

        self.toc = None
        self.tocpomdp = None
        self.tocpomdpGamma = None
        self.tocpomdppi = None

    def _compute_rho(self, bfa, bfhata, timeRemaining, numIterations=25):
        """ Compute the probabilities of reaching terminal `end result' states, following Equations 14 and 15.

            Parameters:
                bfa             --  The current actor. Note we assume in this code that the TOC model is the same for all v.
                bfhata          --  The desired actor.
                timeRemaining   --  How much time is allotted to TOC.
                numIterations   --  Optionally specify the number of iterations used to sample rho. Default is 25.

            Return:
                A 3-array corresponding to: [Pr(human), Pr(vehicle), Pr(side of road)].
        """

        calA = ["human", "vehicle", "side of road"]

        bfaIndex = calA.index(bfa)
        bfhataIndex = calA.index(bfhata)

        toc = self.toc[bfaIndex]
        tocpomdp = self.tocpomdp[bfaIndex]
        Gamma = self.tocpomdpGamma[bfaIndex]
        pi = self.tocpomdppi[bfaIndex]

        rho = np.array([0.0, 0.0, 0.0])

        # Before anything, handle the two cases with 1.0.

        # Equation 14.
        if bfa == "side of road":
            if bfhata != "human":
                rho[2] = 1.0
                return rho

        # Equation 15.
        elif bfa != "side of road":
            if bfhata in [bfa, "side of road"]:
                rho[bfhataIndex] = 1.0
                return rho

        # Now ensure a valid time remaining.
        timeRemaining = min(len(toc.T) - 1, timeRemaining)

        numSuccess = 0
        numFailure = 0
        numAborted = 0

        for k in range(numIterations):
            validInitialStates = [tocpomdp.states.index((timeRemaining, h, "nop", 0)) for h in toc.H]
            b = np.array([1.0 / len(validInitialStates) * (i in validInitialStates) for i in range(tocpomdp.n)])
            s = rnd.choice(validInitialStates)

            for t in range(timeRemaining + 1):
                a, v = take_action(tocpomdp, Gamma, pi, b)
                sp = transition_state(tocpomdp, s, a)
                o = make_observation(tocpomdp, a, sp)
                b = update_belief(tocpomdp, b, a, o)
                s = sp

            if s == tocpomdp.states.index("success"):
                numSuccess += 1
            elif s == tocpomdp.states.index("failure"):
                numFailure += 1
            elif s == tocpomdp.states.index("aborted"):
                numAborted += 1

        # Equation 14.
        if bfa == "side of road":
            # Note: bfhata = "human" (lambda) here because otherwise it would
            # have returned already in the beginning...
            rho[0] = float(numSuccess) # rho[0] == rho[a' = human]
            rho[2] = float(numFailure + numAborted) # rho[2] = rho[a' = side of road]
            rho /= float(numIterations)
            return rho

        # Equation 15.
        elif bfa != "side of road":
            # Note: bfhata not in [bfa, "side of road"] here because otherwise it would
            # have returned already in the beginning...
            rho[bfhataIndex] = float(numSuccess)
            rho[bfaIndex] = float(numFailure)
            rho[2] = float(numAborted) # rho[2] = rho[a' = side of road]
            rho /= float(numIterations)
            return rho

        return None

    def create(self, toc, tocpomdp, path, controller=None):
        """ Create the MDP SSP given the ToC's path planning problem and the ToC problem itself.

            Parameters:
                toc         --  The transfer of control problems, a pair, two of them: h->v and v->h.
                tocpomdp    --  The POMDPs, a pair, two of them: h->v and v->h.
                path        --  The weighted directed graph: (V, E, w, v0, vg).
                controller  --  Optionally restrict the ToC SSP to only "human" or "vehicle" control.
                                Default is None, meaning both are included.
        """

        self.toc = toc
        self.tocpomdp = tocpomdp

        self.tocpomdpGamma = list()
        self.tocpomdppi = list()
        for pomdp in self.tocpomdp:
            Gamma, pi, timing = pomdp.solve()
            self.tocpomdpGamma += [Gamma]
            self.tocpomdppi += [pi]

        calE = ["success", "failure", "aborted"]
        calA = ["human", "vehicle", "side of road"]
        D = ["direction %i" % (i) for i in range(path.maxOutgoingDegree)]

        V = path.V + ["vf"]

        if controller == "human":
            calA = ["human"]
        elif controller == "vehicle":
            calA = ["vehicle"]

        self.states = list(it.product(V, calA))
        self.n = len(self.states)

        self.actions = list(it.product(D, calA))
        self.m = len(self.actions)

        theta = {(v, d): None for v, d in it.product(V, D)}
        for v in path.V:
            vEdges = [e for e in path.E if e[0] == v]
            vEdges = sorted(vEdges, key=lambda z: z[1])
            for dIndex, e in enumerate(vEdges):
                theta[(v, D[dIndex])] = e[1]
        for d in D:
            theta[("vf", d)] = "vf"
            theta[(path.vg, d)] = path.vg

        # For 'simple_tocssp.py', which lets you export for the visualizer, we need to know theta. Store it.
        self.theta = theta

        # Compute all the possible rho values, given all possible states (each having a different time to TOC).
        rho = [[[self._compute_rho(bfa, bfhata, timeRemaining) for bfhata in calA] for bfa in calA] for timeRemaining in self.toc[0].T]

        # The maximum number of successor states is always bounded by 3, because
        # the uncertainty is only ever over the result of the ToC POMDP final
        # absorbing state. All other cases are deterministic.
        self.ns = 3

        S = [[[int(-1) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        T = [[[float(0.0) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            v = state[0]
            bfa = state[1]
            bfaIndex = calA.index(bfa)

            for a, action in enumerate(self.actions):
                d = action[0]
                bfhata = action[1]
                dIndex = D.index(d)
                bfhataIndex = calA.index(bfhata)

                e = (v, theta[(v, d)])
                rho_s_bfhata = None
                try:
                    # This will only fail for e = theta returning a None for the impossible
                    # action. This essentially handles the A(s) case. If this is the failure
                    # vertex, then it doesn't matter what the weight is, since it will have
                    # issues regardless being in a dead end.
                    if v == "vf":
                        rho_s_bfhata = rho[1][bfaIndex][bfhataIndex]
                    else:
                        timeRemaining = min(len(self.toc[0].T) - 1, int(path.w[e]))
                        rho_s_bfhata = rho[timeRemaining][bfaIndex][bfhataIndex]
                except KeyError:
                    # We handle invalid actions by immediately transitioning to the failure
                    # vertex "vf".
                    S[s][a][0] = self.states.index(("vf", bfa))
                    T[s][a][0] = 1.0
                    continue

                cur = 0

                for sp, statePrime in enumerate(self.states):
                    vp = statePrime[0]
                    bfap = statePrime[1]
                    bfapIndex = calA.index(bfap)

                    if bfa == "human": # lambda
                        # Equation 11.
                        T_lambda = 0.0
                        if vp == theta[(v, d)]:
                            T_lambda = 1.0

                        # Don't store a 0.0 state transition...
                        if T_lambda != 0.0:
                            # Equation 1.
                            if bfa == bfhata and bfhata == bfap:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_lambda
                                cur += 1
                            elif bfa != bfhata:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_lambda * rho_s_bfhata[bfapIndex]
                                cur += 1

                    if bfa == "vehicle": # nu
                        # Equation 12.
                        T_nu = 0.0
                        if e in path.Ec and vp == theta[(v, d)]:
                            T_nu = 1.0
                        elif (v, theta[(v, d)]) not in path.Ec and vp == "vf":
                            T_nu = 1.0

                        # Don't store a 0.0 state transition...
                        if T_nu != 0.0:
                            # Equation 1.
                            if bfa == bfhata and bfhata == bfap:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_nu
                                cur += 1
                            elif bfa != bfhata:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_nu * rho_s_bfhata[bfapIndex]
                                cur += 1

                    if bfa == "side of road":
                        # Equation 13.
                        T_sigma = 0.0
                        if vp == v:
                            T_sigma = 1.0

                        # Don't store a 0.0 state transition...
                        if T_sigma != 0.0:
                            # Equation 1.
                            if bfa == bfhata and bfhata == bfap:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_sigma
                                cur += 1
                            elif bfa != bfhata:
                                S[s][a][cur] = sp
                                T[s][a][cur] = T_sigma * rho_s_bfhata[bfapIndex]
                                cur += 1

        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)

        self.S = array_type_nmns_int(*np.array(S).flatten())
        self.T = array_type_nmns_float(*np.array(T).flatten())

        wmin = min(path.w.values())
        wmax = max(path.w.values())

        self.epsilon = 0.001
        self.gamma = 1.0
        self.horizon = 10000
        #self.horizon = max(10000, int(np.log(2.0 * (wmax - wmin) / (self.epsilon * (1.0 - self.gamma))) / np.log(1.0 / self.gamma)))

        R = [[0.0 for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            v = state[0]
            bfa = state[1]

            for a, action in enumerate(self.actions):
                d = action[0]
                bfhata = action[1]

                e = (v, theta[(v, d)])

                # Equation 16, plus the tie-breaking and handling invalid actions.
                if v == path.vg:
                    R[s][a] = 0.0
                elif v == "vf":
                    R[s][a] = (wmax + wmin * 3.0)
                elif theta[(v, d)] == None:
                    R[s][a] = (wmax + wmin * 2.0)
                elif e in path.Ep and bfa != "vehicle":
                    R[s][a] = (path.w[e] + wmin)
                else:
                    R[s][a] = path.w[e]

        self.Rmax = np.array(R).max()
        self.Rmin = np.array(R).min()

        array_type_nm_float = ct.c_float * (self.n * self.m)

        self.R = array_type_nm_float(*np.array(R).flatten())

        # The initial state is the initial state in the graph, with the human driving, except
        # if the vehicle is the only one that can control the ToC SSP. Similarly for the goal.
        if controller is None:
            self.s0 = self.states.index((path.v0, "human"))

            self.ng = 3
            array_type_ng_uint = ct.c_uint * (self.ng)

            self.goals = array_type_ng_uint(*np.array([self.states.index((path.vg, bfa)) for bfa in calA]))
        elif controller == "human":
            self.s0 = self.states.index((path.v0, "human"))

            self.ng = 1
            array_type_ng_uint = ct.c_uint * (self.ng)

            self.goals = array_type_ng_uint(*np.array([self.states.index((path.vg, "human"))]))
        elif controller == "vehicle":
            self.s0 = self.states.index((path.v0, "vehicle"))

            self.ng = 1
            array_type_ng_uint = ct.c_uint * (self.ng)

            self.goals = array_type_ng_uint(*np.array([self.states.index((path.vg, "vehicle"))]))


if __name__ == "__main__":
    print("Performing ToCSSP Unit Test...")

    tocHuman = ToC(randomize=(2, 2, 2, 5))
    tocpomdpHuman = ToCPOMDP()
    tocpomdpHuman.create(tocHuman)

    tocVehicle = ToC(randomize=(2, 2, 2, 5))
    tocpomdpVehicle = ToCPOMDP()
    tocpomdpVehicle.create(tocVehicle)

    tocSideOfRoad = ToC(randomize=(2, 2, 2, 5))
    tocpomdpSideOfRoad = ToCPOMDP()
    tocpomdpSideOfRoad.create(tocSideOfRoad)

    toc = (tocHuman, tocVehicle, tocSideOfRoad)
    tocpomdp = (tocpomdpHuman, tocpomdpVehicle, tocpomdpSideOfRoad)

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

    #V, pi, timing = tocssp.solve(algorithm='lao*', process='cpu')
    V, pi, timing = tocssp.solve(algorithm='vi', process='gpu')

    print("Done.\nResults:")

    print(tocpath)
    print(tocssp)
    print(V)
    print(pi.tolist())

    print("Done.")

