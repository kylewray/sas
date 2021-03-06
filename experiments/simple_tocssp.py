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
from tocinteract import *

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from tocpath import *
from tocssp import *


def save_policy_for_visualizer(tocssp, tocpath, V, pi, filename):
    """ Save the policy for the SSP to a policy file for use in the LOSM visualizer.

        Parameters:
            tocssp      --  The ToC SSP.
            tocpath     --  The ToC Path.
            V           --  The values of the states.
            pi          --  The mapping from states to actions.
            filename    --  The filename of the output policy file.
    """

    with open(filename, 'w') as f:
        for s, state in enumerate(tocssp.states):
            v = state[0]
            bfa = state[1]

            # Only save the states expanded and visited by LAO*.
            if pi[s] == tocssp.m:
                continue

            action = tocssp.actions[pi[s]]
            d = action[0]
            bfhata = action[1]

            e = (v, tocssp.theta[(v, d)])

            if v != tocpath.vg and v != "vf": # and not (e not in tocpath.Ec and bfa == "vehicle")
                currentVertexUID = state[0]
                currentAutonomy = int(state[1] == "vehicle" or state[1] == "side of road")

                actionDirection, desiredActor = tocssp.actions[pi[s]]
                nextVertexUID = tocssp.theta[(currentVertexUID, actionDirection)]

                # This means the action was undefined at this state, i.e., it wasn't in A(s).
                if nextVertexUID is None:
                    raise Exception()
                    #nextVertexUID = currentVertexUID

                nextAutonomy = int(desiredActor == "vehicle" or desiredActor == "side of road")

                previousVertexUIDs = [e[0] for e in tocpath.E if e[1] == currentVertexUID]

                for previousVertexUID in previousVertexUIDs:
                    for currentTiredness in [0, 1]:
                        f.write(",".join([str(previousVertexUID), str(currentVertexUID),
                                          str(currentTiredness), str(currentAutonomy),
                                          str(nextVertexUID), str(nextAutonomy),
                                          #str(V[s])
                                         ]) + "\n")


if __name__ == "__main__":
    print("Performing Basic ToCSSP Experiment...")

    if len(sys.argv) != 5:
        print("Must specify an input OSM file path, start UID, goal UID, and output policy file name, in that order.")
        sys.exit(0)

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

    print("Creating the ToC Path... ", end='')
    sys.stdout.flush()

    tocpath = ToCPath()
    tocpath.load(sys.argv[1])
    tocpath.v0 = int(sys.argv[2])
    tocpath.vg = int(sys.argv[3])

    print("Done.\nCreating the ToC SSP... ", end='')
    sys.stdout.flush()

    tocssp = ToCSSP()
    #tocssp.create(toc, tocpomdp, tocpath, controller='human')
    #tocssp.create(toc, tocpomdp, tocpath, controller='vehicle')
    tocssp.create(toc, tocpomdp, tocpath, controller=None)

    print("Done.\nSolving the ToC SSP... ", end='')
    sys.stdout.flush()

    V, pi, timing = tocssp.solve(algorithm='lao*', process='cpu')
    #V, pi, timing = tocssp.solve(algorithm='vi', process='gpu')

    print("Done.\nSaving the Policy... ", end='')
    sys.stdout.flush()

    save_policy_for_visualizer(tocssp, tocpath, V, pi, sys.argv[4])

    print("Done.\nResults:")

    print(tocpath)
    print(tocssp)
    print(V)
    print(pi.tolist())

    print("Done.")

