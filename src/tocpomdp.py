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


class ToCPOMDP(POMDP):
    """ A class which models the ToC POMDP problem. """

    def __init__(self):
        """ The constructor of the ToC POMDP class. """

        super().__init__()

        self.states = list()
        self.actions = list()
        self.observations = list()

    def create(self, toc):
        """ Create the POMDP given the ToC problem.

            Parameters:
                toc --  The transfer of control object.
        """

        calZ = ["success", "failed", "aborted"]

        self.states = list(it.product(toc.T, toc.H, toc.M, toc.T)) + list(calZ)
        self.n = len(self.states)

        self.actions = list(toc.M) + ["abort"]
        self.m = len(self.actions)

        self.observations = list(toc.O) + calZ
        self.z = len(self.observations)

        self.ns = len(toc.H) + 1 # Add the possible "success" state as a successor, too.

        S = [[[int(-1) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        T = [[[float(0.0) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                cur = 0

                for sp, statePrime in enumerate(self.states):
                    # Absorbing states always self-loop.
                    if state in calZ and statePrime in calZ and state == statePrime:
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # If the agent aborts, then transition to aborted. This is always an option.
                    if state not in calZ and action == "abort" and statePrime == "aborted":
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0
                        cur += 1

                    # Randomly, it may actually succeed to transfer control.
                    if state not in calZ and action != "abort" and statePrime == "success":
                        S[s][a][cur] = sp
                        T[s][a][cur] = toc.Pc[(state[1], state[2], state[3])]
                        cur += 1

                    # If the countdown runs out of time, then it is over if control fails to transfer.
                    # At this point, it is too late to "abort".
                    if state not in calZ and state[0] == 0 and action != "abort" and statePrime == "failed":
                        S[s][a][cur] = sp
                        T[s][a][cur] = 1.0 - toc.Pc[(state[1], state[2], state[3])]
                        cur += 1

                    # If the agent does not abort, then there are two cases. First, (1) no message is sent (i.e., "nop").
                    # In this case, the distribution follows toc.Ph, but only if the next state:
                    # (a) decrements the timer, (b) distribution over human states, (c) same message, (d) increased time
                    # since the last message, or self-loop for the maximal case.
                    if state not in calZ and state[0] > 0 and action != "abort" and statePrime not in calZ and \
                            action == "nop" and statePrime[0] == state[0] - 1 and statePrime[2] == state[2]:
                        if statePrime[3] == state[3] + 1:
                            S[s][a][cur] = sp
                            T[s][a][cur] = (1.0 - toc.Pc[(state[1], state[2], state[3])]) * toc.Ph[(state[1], state[2], state[3], statePrime[1])]
                            cur += 1
                        elif statePrime[3] == state[3] and statePrime[3] == len(toc.T) - 1 and statePrime[1] == state[1]:
                            S[s][a][cur] = sp
                            T[s][a][cur] = 1.0 - toc.Pc[(state[1], state[2], state[3])]
                            cur += 1

                    # Next, the other case: (2) a message is sent.
                    if state not in calZ and state[0] > 0 and action != "abort" and statePrime not in calZ and \
                            action != "nop" and statePrime[0] == state[0] - 1 and statePrime[2] == action and statePrime[3] == 0:
                        S[s][a][cur] = sp
                        T[s][a][cur] = (1.0 - toc.Pc[(state[1], state[2], state[3])]) * toc.Ph[(state[1], state[2], state[3], statePrime[1])]
                        cur += 1

        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)

        self.S = array_type_nmns_int(*np.array(S).flatten())
        self.T = array_type_nmns_float(*np.array(T).flatten())

        O = [[[0.0 for z in range(self.z)] for sp in range(self.n)] for a in range(self.m)]
        for a, action in enumerate(self.actions):
            for sp, statePrime in enumerate(self.states):
                for o, observation in enumerate(self.observations):
                    # In terminal states, there's a 1.0 probability of the agent knowing it is there.
                    if statePrime in calZ and observation == statePrime:
                        O[a][sp][o] = 1.0

                    # Otherwise, the probability follows from the ToC object, given the observation is valid.
                    if statePrime not in calZ and observation not in calZ:
                        O[a][sp][o] = toc.Po[(statePrime[1], observation)]

        array_type_mnz_float = ct.c_float * (self.m * self.n * self.z)
        self.O = array_type_mnz_float(*np.array(O).flatten())

        # Compute the maximum and non-zero minimum ToC's cost.
        Cmax = max([value for key, value in toc.C.items()])
        Cmin = min([value for key, value in toc.C.items() if value > 0.0])

        R = [[0.0 for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                # There is no cost in the goal 'success' state.
                if state == "success":
                    R[s][a] = 0.0

                # Failure repeatedly has the worst cost.
                if state == "failure":
                    R[s][a] = -Cmax #-1e+35

                # The aborted state is not ideal, but is better than failure.
                if state == "aborted":
                    R[s][a] = 0.0 #-Cmin

                # Abort pays the maximal cost over all time steps; much better
                # than failure though.
                if state not in calZ and action == "abort":
                    if state[0] > 0:
                        R[s][a] = -Cmax #-Cmax# * len(toc.T)
                    else:
                        R[s][a] = 0.0

                # NOP has a very small immediate cost, as well as "abort".
                if state not in calZ and action == "nop":
                    # Note: This value has to be less than the reward for aborting at state[0] == 0 above.
                    # If not, then it will fail to choose abort.
                    R[s][a] = -0.01

                # All other states have a cost equal to the toc.C values. This is based on
                # the human's state, how long it has been since the human was annoyed,
                # and the *new* message just chosen.
                if state not in calZ and action not in ["nop", "abort"]:
                    R[s][a] = -toc.C[(state[1], state[2], state[3], action)] * (toc.Pc[(state[1], state[2], state[3])] > 0.0)

        self.Rmax = np.array(R).max()
        self.Rmin = np.array(R).min()

        array_type_nm_float = ct.c_float * (self.n * self.m)

        self.R = array_type_nm_float(*np.array(R).flatten())

        # Setup the belief points. We begin with a seed state uniform over all three
        # human states, and maximal time remaining, maximal time since last message,
        # and the last message was NOP.
        Z = [[self.states.index((len(toc.T) - 1, h, "nop", len(toc.T) - 1)) for h in toc.H]]
        B = [[1.0 / float(len(toc.H)) for h in toc.H]]

        # Add beliefs for the final time step in order to guarantee abort can always be executed.
        Z += [[self.states.index((0, h, m, t)) for h in toc.H] for m, t in it.product(toc.M, toc.T)]
        B += [[1.0 / float(len(toc.H)) for h in toc.H] for m, t in it.product(toc.M, toc.T)]

        self.r = len(B)
        self.rz = len(toc.H)

        array_type_rrz_int = ct.c_int * (self.r * self.rz)
        array_type_rrz_float = ct.c_float * (self.r * self.rz)

        self.Z = array_type_rrz_int(*np.array(Z).flatten())
        self.B = array_type_rrz_float(*np.array(B).flatten())

        # There was only one reward, and there's a simple discount factor.
        self.k = 1
        self.gamma = 0.95

        # We know the maximal horizon necessary, since we have a countdown timer, but
        # make sure it also realizes how bad some of the absorbing states are.
        self.horizon = len(toc.T) * 10

        #self.expand(method='random', numBeliefsToAdd=100)
        #for i in range(3):
        #    self.expand(method='distinct_beliefs')
        for i in range(30):
            self.expand(method='pema')


if __name__ == "__main__":
    print("Performing ToCPOMDP Unit Test...")

    toc = ToC(randomize=(3, 2, 3, 5))
    tocpomdp = ToCPOMDP()
    tocpomdp.create(toc)
    print(tocpomdp)

    Gamma, pi, timing = tocpomdp.solve()
    print("Gamma:\n", Gamma)
    print("pi:\n", pi.tolist())

    print("Done.")

