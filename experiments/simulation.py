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

import numpy as np
import random as rnd

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from toc import *
from tocpomdp import *


def take_action(pomdp, Gamma, pi, b):
    """ Take the best action at the current belief point.

        Parameters:
            pomdp   --  The POMDP.
            Gamma   --  The set of alpha-vectors.
            pi      --  The corresponding actions for each alpha-vector.
            b       --  The current belief point.

        Returns:
            The best action index to take at this belief point.
    """

    bestVal = np.dot(Gamma[0, :], b)
    bestAction = pi[0]

    for i in range(pomdp.r):
        val = np.dot(Gamma[i, :], b)
        if val > bestVal:
            bestVal = val
            bestAction = pi[i]

    return bestAction


def transition_state(pomdp, s, a):
    """ Randomly transition the true underlying state.

        Parameters:
            pomdp   --  The POMDP.
            s       --  The true current state index.
            a       --  The action index taken.

        Returns:
            The successor state index.
    """

    num = rnd.random()
    count = 0.0

    for i in range(pomdp.ns):
        count += pomdp.T[s * pomdp.m * pomdp.ns + a * pomdp.ns + i]
        if count >= num:
            return pomdp.S[s * pomdp.m * pomdp.ns + a * pomdp.ns + i]

    print("Warning (transition_state): The random belief is being weird...")

    return pomdp.S[s * pomdp.m * pomdp.ns + a * pomdp.ns + (pomdp.ns - 1)]


def make_observation(pomdp, a, sp):
    """ Randomly make an observation based on the action taken and the true state of the system.

        Parameters:
            pomdp   --  The POMDP.
            a       --  The action index taken.
            sp      --  The true successor state index.

        Returns:
            The index of the observation.
    """

    num = rnd.random()
    count = 0.0

    for o in range(pomdp.z):
        count += pomdp.O[a * pomdp.n * pomdp.z + sp * pomdp.z + o]
        if count >= num:
            return o

    print("Warning (make_observations): The random belief is being weird...")

    return pomdp.z - 1


def update_belief(pomdp, b, a, o):
    """ Perform a belief update given the POMDP and the current belief.

        Parameters:
            pomdp   --  The POMDP.
            b       --  The current belief.
            a       --  The action index taken.
            o       --  The observation index made.

        Returns:
            The next belief after taking an action and observing something.
    """

    bp = np.array([0.0 for i in range(pomdp.n)])

    for s in range(pomdp.n):
        for i in range(pomdp.ns):
            sp = pomdp.S[s * pomdp.m * pomdp.ns + a * pomdp.ns + i]
            if sp < 0:
                break
            bp[sp] += pomdp.T[s * pomdp.m * pomdp.ns + a * pomdp.ns + i] * b[s]

    for sp in range(pomdp.n):
        bp[sp] *= pomdp.O[a * pomdp.n * pomdp.z + sp * pomdp.z + o]

    bp = bp / bp.sum()

    return bp

def initialize():
    """ Setup the POMDP and solve it.

        Returns:
            toc         --  The ToC problem.
            tocpomdp    --  The ToC POMDP.
            Gamma       --  The policy's alpha-vectors.
            pi          --  The policy's corresponding actions for each alpha-vector.
    """

    print("Creating the ToC POMDP.")
    toc = ToC(randomize=(3, 2, 3, 5))
    tocpomdp = ToCPOMDP()
    tocpomdp.create(toc)
    print(tocpomdp)

    print("Solving the ToC POMDP.")
    Gamma, pi, timing = tocpomdp.solve()
    #print("Gamma:\n", Gamma)
    #print("pi:\n", pi.tolist())

    return toc, tocpomdp, Gamma, pi


def simulation(toc, tocpomdp, Gamma, pi, numIterations):
    """ Execute a single simulation of a randomized ToC POMDP.

        Parameters:
            toc             --  The ToC problem.
            tocpomdp        --  The ToC POMDP.
            Gamma           --  The policy's alpha-vectors.
            pi              --  The policy's corresponding actions for each alpha-vector.
            numIterations   --  The number of iterations to execute.
    """

    for k in range(numIterations):
        print("------------------------------------------------")
        print("Simulating Execution %i of %i." % (k + 1, numIterations))
        print("------------------------------------------------")

        validInitialStates = [tocpomdp.states.index((len(toc.T) - 1, h, "nop", len(toc.T) - 1)) for h in toc.H]
        b = np.array([1.0 / len(validInitialStates) * (i in validInitialStates) for i in range(tocpomdp.n)])
        s = rnd.choice(validInitialStates)

        for t in toc.T:
            print("Time Remaining:   %i" % (len(toc.T) - 1 - t))
            print("Belief:           %s" % (str(["%s: %.2f" % (str(tocpomdp.states[i]), b[i]) for i in range(tocpomdp.n) if b[i] > 0.0])))
            print("True State:       %s" % (str(tocpomdp.states[s])))

            a = take_action(tocpomdp, Gamma, pi, b)
            sp = transition_state(tocpomdp, s, a)
            o = make_observation(tocpomdp, a, sp)
            b = update_belief(tocpomdp, b, a, o)
            s = sp

            print("Action Taken:     %s" % (tocpomdp.actions[a]))
            print("Observation Made: %s" % (tocpomdp.observations[o]))
            print("------------------------------------------------")

        print("Belief:           %s" % (str(["%s: %.2f" % (tocpomdp.states[i], b[i]) for i in range(tocpomdp.n) if b[i] > 0.0])))
        print("Final State:      %s" % (tocpomdp.states[s]))

        print("------------------------------------------------")


if __name__ == "__main__":
    numIterations = 1
    if len(sys.argv) == 2:
        numIterations = int(sys.argv[1])

    print("Executing Simulation Experiments...")

    toc, tocpomdp, Gamma, pi = initialize()

    simulation(toc, tocpomdp, Gamma, pi, numIterations)

    print("Done.")

