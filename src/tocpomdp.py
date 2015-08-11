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

        print(self.states)
        print(self.actions)
        print(self.observations)

        S = [[[int(-1) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        T = [[[float(0.0) for sp in range(self.ns)] for a in range(self.m)] for s in range(self.n)]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                cur = 0

                try:
                    for sp, statePrime in enumerate(self.states):
                        # Absorbing states always self-loop.
                        if state in calZ and statePrime in calZ and state == statePrime:
                            S[s][a][cur] = sp
                            T[s][a][cur] = 1.0
                            cur += 1

                        # If the countdown runs out of time, then it is over if control fails to transfer.
                        #if state not in calZ and state[0] == 0 and statePrime == "failed":
                        #    S[s][a][cur] = sp
                        #    T[s][a][cur] = 1.0 - toc.Pc[(state[1], state[2], state[3])]
                        #    cur += 1

                        # If the agent aborts, then transition to aborted.
                        #if state not in calZ and state[0] > 0 and action == "abort" and statePrime == "aborted":
                        #    S[s][a][cur] = sp
                        #    T[s][a][cur] = 1.0
                        #    cur += 1

                        # Randomly, it may actually succeed to transfer control.
                        if state not in calZ and statePrime == "success":
                            S[s][a][cur] = sp
                            T[s][a][cur] = toc.Pc[(state[1], state[2], state[3])]
                            cur += 1

                        # The agent may not have sent a message, so align everything else and
                        # properly form a distribution over toc.Ph. The first case considers when
                        # the agent actually sends a message.
                        if state not in calZ and statePrime not in calZ and state[0] > 0 and statePrime[0] == state[0] - 1 and \
                                action not in ["abort", "nop"] and state[2] == action and statePrime[3] == 0:
                            S[s][a][cur] = sp
                            T[s][a][cur] = (1.0 - toc.Pc[(state[1], state[2], state[3])]) * toc.Ph[(state[1], state[2], state[3], statePrime[1])]
                            cur += 1

                        if state not in calZ and statePrime not in calZ and state[0] > 0 and statePrime[0] == state[0] - 1 and \
                                action == "nop" and statePrime[2] == state[2] and statePrime[3] == state[3] + 1:
                            S[s][a][cur] = sp
                            T[s][a][cur] = (1.0 - toc.Pc[(state[1], state[2], state[3])]) * toc.Ph[(state[1], state[2], state[3], statePrime[1])]
                            cur += 1

                except Exception:
                    print(cur)
                    pass

        print(np.array(S))
        print(np.array(T))

        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)

        self.S = array_type_nmns_int(*np.array(S).flatten())
        self.T = array_type_nmns_float(*np.array(T).flatten())

if __name__ == "__main__":
    print("Performing ToCPOMDP Unit Test...")

    toc = ToC(randomize=(2, 2, 2, 2))
    tocpomdp = ToCPOMDP()
    tocpomdp.create(toc)
    print(tocpomdp)

    #Gamma, pi, timing = tocpomdp.solve()
    #print("Gamma:\n", Gamma)
    #print("pi:\n", pi.tolist())


    print("Done.")

