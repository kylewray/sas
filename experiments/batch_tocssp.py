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
sys.path.append(thisFilePath)

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from tocpath import *
from tocssp import *
from tocinteract import *


def simulate(tocssp, tocpath, pi):
    """ Simulate the ToC SSP from its initial state and return the relevant metrics.

        Parameters:
            tocssp      --  The ToC SSP.
            tocpath     --  The ToC Path.
            pi          --  The mapping from states to actions.

        Returns:
            isGoalReachable         --  If the goal was reached. Either 'Y' or 'N'.
            percentageAutonomous    --  The percentage of time it was autonomous, out of the roads
                                        it traveled on which could have been.
            travelTime              --  The actual travel time on the roads, which are the undiscounted weights.
    """

    calEIndexes = [tocssp.states.index("success"), tocssp.states.index("failure"), tocssp.states.index("aborted")]

    # A counter of either 0 or 1 for each state visited if it was autonomous or not and they were driving autonomously.
    autonomousCounter = list()

    travelTime = 0.0
    s = tocssp.s0

    while s not in calEIndexes and len(autonomousCounter) < 20:
        v, x = tocssp.states[s]

        print("State: %i (%s, %s)" % (s, v, x))

        # Randomly transition to a new state following the state transition function.
        targetValue = rnd.random()
        currentValue = 0.0
        sp = tocssp.S[s * tocssp.m * tocssp.ns + pi[s] * tocssp.ns + 0]

        for i in range(tocssp.ns):
            spPotential = tocssp.S[s * tocssp.m * tocssp.ns + pi[s] * tocssp.ns + i]
            if spPotential < 0:
                break

            currentValue += tocssp.T[s * tocssp.m * tocssp.ns + pi[s] * tocssp.ns + i]
            if currentValue >= targetValue:
                sp = spPotential
                break

        print("Action: %i %s" % (pi[s], tocssp.actions[pi[s]]))

        # The last transision from s to sp at an absorbing state has no extra properties
        # in terms of time or autonomy. Skip it.
        if sp not in calEIndexes:
            # Increment the autonomy counter and travel time. Note that we count it if
            # the autonomous driving was done on *this* state, since it is assumed that
            # transfer of control will occur near end of the edge itself (if at all).
            vp, xp = tocssp.states[sp]
            e = (v, vp)

            print("State Prime: %i (%s, %s)" % (sp, vp, xp))

            autonomousCounter += [x == "vehicle" and e in tocpath.Eap]
            travelTime += tocpath.w[e]
        else:
            print("State Prime: %i (%s)" % (sp, tocssp.states[sp]))

        s = sp

    isGoalReachable = (s == calEIndexes[0])

    percentageAutonomous = (tocssp.states[tocssp.s0][1] == "vehicle")
    if len(autonomousCounter) > 0:
        percentageAutonomous = np.mean(autonomousCounter)

    travelTime = travelTime

    return isGoalReachable, percentageAutonomous, travelTime


cities = {("Boston", "maps/boston/boston.osm", 61341179, 61340485),
         }

resultsFilename = "results/results.csv"
numTrials = 1


def batch():
    """ Execute a batch run for each city, and each configuration, then save the results to a file. """

    # As our model allows, we create one POMDP for each 'scenario' which works for any ToC SSP (city or map).
    # Here, we allow for two scenarios: human to vehicle and vehicle to human.
    tocHtoV = ToCInteract(nt=5) #randomize=(2, 2, 2, 5))
    tocVtoH = ToCInteract(nt=5) #randomize=(2, 2, 2, 5))
    toc = (tocHtoV, tocVtoH)

    tocpomdpHtoV = None
    tocpomdpVtoH = None
    tocpomdp = (tocpomdpHtoV, tocpomdpVtoH)

    for city, filename, startVertex, goalVertex in cities:
        print("Experiment '%s'" % (city), end='')
        sys.stdout.flush()

        # Load the OSM file as a ToCPath object.
        tocpath = ToCPath()
        tocpath.load(filename)
        tocpath.v0 = startVertex
        tocpath.vg = goalVertex

        # We record: 1) percentage of times it reached the goal, 2) percentage of time it was
        # autonomous, and 3) total travel time, for each of the three scenarios.
        isGoalReachable = {'h': None, 'v': None, 'h+v': None}
        percentageAutonomous = {'h': None, 'v': None, 'h+v': None}
        travelTime = {'h': None, 'v': None, 'h+v': None}

        # Step 1: Construct the ToCSSP with just a human controller.
        tocssp = ToCSSP()
        tocssp.create(toc, tocpomdp, tocpath, controller="human")
        V, pi = tocssp.solve()

        isGoalReachable['h'], percentageAutonomous['h'], travelTime['h'] = simulate(tocssp, tocpath, pi)

        print(".", end='')
        sys.stdout.flush()

        # Step 2: Construct the ToCSSP with just a vehicle controller.
        tocssp = ToCSSP()
        tocssp.create(toc, tocpomdp, tocpath, controller="vehicle")
        V, pi = tocssp.solve()

        isGoalReachable['v'], percentageAutonomous['v'], travelTime['v'] = simulate(tocssp, tocpath, pi)

        print(".", end='')
        sys.stdout.flush()

        # Step 3: Construct the ToCSSP with both a human and vehicle controller.
        tocssp = ToCSSP()
        tocssp.create(toc, tocpomdp, tocpath, controller=None)
        V, pi = tocssp.solve()

        travelTimeTrial = np.array([0.0 for i in range(numTrials)])
        for i in range(numTrials):
            isGoalReachable['h+v'], percentageAutonomous['h+v'], travelTimeTrial[i] = simulate(tocssp, tocpath, pi)
        travelTime['h+v'] = np.mean(travelTimeTrial)
        travelTimeStd = np.std(travelTimeTrial)

        print(".", end='')
        sys.stdout.flush()

        with open(resultsFilename, 'w') as f:
            f.write("%s,%i,%i,%s,%.4f,%.4f,%s,%.4f,%.4f,%s,%.4f,%.4f\n" % (city, tocssp.n, tocssp.m,
                    isGoalReachable['h'], percentageAutonomous['h'], travelTime['h'],
                    isGoalReachable['v'], percentageAutonomous['v'], travelTime['v'],
                    isGoalReachable['h+v'], percentageAutonomous['h+v'], travelTime['h+v']))

        print(" Done.")

    print("Done.")


if __name__ == "__main__":
    print("Performing Batch ToCSSP Experiments...")
    batch()
    print("Done.")


