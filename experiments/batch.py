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


def simulate(tocssp, tocpath, pi, printTrajectory=False):
    """ Simulate the ToC SSP from its initial state and return the relevant metrics.

        Parameters:
            tocssp          --  The ToC SSP.
            tocpath         --  The ToC Path.
            pi              --  The mapping from states to actions.
            printTrajectory --  Optionally print the trajectory. Default is False.

        Returns:
            isGoalReachable         --  If the goal was reached. Either 'Y' or 'N'.
            percentageAutonomous    --  The percentage of time it was autonomous, out of the roads
                                        it traveled on which could have been.
            travelTime              --  The actual travel time on the roads, which are the undiscounted weights.
    """

    maxIterations = 10000

    # A counter of either 0 or 1 for each state visited if it was autonomous or not and they were driving autonomously.
    autonomousCounter = list()

    travelTime = 0.0
    s = tocssp.s0

    while tocssp.states[s][0] != tocpath.vg and len(autonomousCounter) < maxIterations:
        v, x = tocssp.states[s]

        if printTrajectory:
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

        if printTrajectory:
            print("Action: %i %s" % (pi[s], tocssp.actions[pi[s]]))

        # The last transision from s to sp at an absorbing state has no extra properties
        # in terms of time or autonomy. Skip it.
        if tocssp.states[sp][0] != tocpath.vg:
            # Increment the autonomy counter and travel time. Note that we count it if
            # the autonomous driving was done on *this* state, since it is assumed that
            # transfer of control will occur near end of the edge itself (if at all).
            vp, xp = tocssp.states[sp]
            e = (v, vp)

            if printTrajectory:
                print("State Prime: %i (%s, %s)" % (sp, vp, xp))

            autonomousCounter += [x == "vehicle" and e in tocpath.Ep]

            # Add to the travel time. Also, handle the special case in which ToC Failed.
            # This, as per the definition, assumes that the maximal amount of time is
            # spent waiting to try ToC again. Sometimes there are self-loops in roads,
            # so the edge will exist in w. When the edge does not exist, that is when
            # we have s == sp at a normal road.
            try:
                travelTime += tocpath.w[e]
            except KeyError:
                travelTime += max([time for edge, time in tocpath.w.items()])
        else:
            if printTrajectory:
                print("State Prime: %i (%s)" % (sp, tocssp.states[sp]))

        s = sp

    isGoalReachable = (tocssp.states[s][0] == tocpath.vg)

    percentageAutonomous = (tocssp.states[tocssp.s0][1] == "vehicle")
    if len(autonomousCounter) > 0:
        percentageAutonomous = np.mean(autonomousCounter)

    travelTime = travelTime

    return isGoalReachable, percentageAutonomous, travelTime


cities = [("Austin", "maps/austin/austin.osm", 152702349, 282401347),
          ("Baltimore", "maps/baltimore/baltimore.osm", 49466255, 37358743),
          ("Boston", "maps/boston/boston.osm", 61353309, 896277838),
          ("Chicago", "maps/chicago/chicago.osm", 353785883, 315396746),
          ("Denver", "maps/denver/denver.osm", 3257743261, 176071158),
          ("Los Angeles", "maps/los_angeles/los_angeles.osm", 123739873, 1499113026),
          ("New York City", "maps/new_york_city/new_york_city.osm", 42960179, 448041300),
          ("Pittsburgh", "maps/pittsburgh/pittsburgh.osm", 104866192, 105701915),
          ("San Francisco", "maps/san_francisco/san_francisco.osm", 259392513, 65307150),
          ("Seattle", "maps/seattle/seattle.osm", 1506967740, 1730115922),
         ]


resultsFilename = os.path.join(thisFilePath, "..", "results", "results_" + str(int(round(time.time() * 1000))) + ".csv")
numTrials = 100


def batch():
    """ Execute a batch run for each city, and each configuration, then save the results to a file. """

    # As our model allows, we create one POMDP for each 'scenario' which works for any ToC SSP (city or map).
    # Here, we allow for two scenarios: human to vehicle and vehicle to human.
    tocHuman = ToCInteract(nt=8)
    tocpomdpHuman = ToCPOMDP()
    tocpomdpHuman.create(tocHuman)

    tocVehicle = ToCInteract(nt=8)
    tocpomdpVehicle = ToCPOMDP()
    tocpomdpVehicle.create(tocVehicle)

    tocSideOfRoad = ToCInteract(nt=8)
    tocpomdpSideOfRoad = ToCPOMDP()
    tocpomdpSideOfRoad.create(tocSideOfRoad)

    toc = (tocHuman, tocVehicle, tocSideOfRoad)
    tocpomdp = (tocpomdpHuman, tocpomdpVehicle, tocpomdpSideOfRoad)

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
        #V, pi, timing = tocssp.solve(algorithm='lao*', process='cpu')
        V, pi, timing = tocssp.solve(algorithm='vi', process='gpu')

        isGoalReachable['h'], percentageAutonomous['h'], travelTime['h'] = simulate(tocssp, tocpath, pi)

        print(".", end='')
        sys.stdout.flush()

        # Step 2: Construct the ToCSSP with just a vehicle controller.
        tocssp = ToCSSP()
        tocssp.create(toc, tocpomdp, tocpath, controller="vehicle")
        #V, pi, timing = tocssp.solve(algorithm='lao*', process='cpu')
        V, pi, timing = tocssp.solve(algorithm='vi', process='gpu')

        isGoalReachable['v'], percentageAutonomous['v'], travelTime['v'] = simulate(tocssp, tocpath, pi)

        print(".", end='')
        sys.stdout.flush()

        # Step 3: Construct the ToCSSP with both a human and vehicle controller.
        tocssp = ToCSSP()
        tocssp.create(toc, tocpomdp, tocpath, controller=None)
        #V, pi, timing = tocssp.solve(algorithm='lao*', process='cpu')
        V, pi, timing = tocssp.solve(algorithm='vi', process='gpu')

        isGoalReachableTrials = np.array([0.0 for i in range(numTrials)])
        percentageAutonomousTrials = np.array([0.0 for i in range(numTrials)])
        travelTimeTrials = np.array([0.0 for i in range(numTrials)])
        for i in range(numTrials):
            isGoalReachableTrials[i], percentageAutonomousTrials[i], travelTimeTrials[i] = simulate(tocssp, tocpath, pi)
        isGoalReachable['h+v'] = bool(np.mean(isGoalReachableTrials))
        percentageAutonomous['h+v'] = np.mean(percentageAutonomousTrials)
        travelTime['h+v'] = np.mean(travelTimeTrials)

        print(".", end='')
        sys.stdout.flush()

        with open(resultsFilename, 'a') as f:
            f.write("%s,%i,%i,%s,%.4f,%.4f,%s,%.4f,%.4f,%s,%.4f,%.4f\n" % (city, tocssp.n, tocssp.m,
                    isGoalReachable['h'], percentageAutonomous['h'], travelTime['h'],
                    isGoalReachable['v'], percentageAutonomous['v'], travelTime['v'],
                    isGoalReachable['h+v'], percentageAutonomous['h+v'], travelTime['h+v']))

        print(" Done.")


if __name__ == "__main__":
    print("Performing Batch ToCSSP Experiments...")
    batch()
    print("Done.")


