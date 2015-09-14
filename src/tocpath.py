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

import random as rnd
import itertools as it

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thisFilePath)

sys.path.append(os.path.join(thisFilePath, "..", "..", "losm", "python", "losm", "converter"))
from losm_converter import *


class ToCPath(object):
    """ A class which holds the weighted directed graph of the ToC Path. """

    def __init__(self):
        """ The constructor for the ToC Path class. """

        self.V = list()
        self.E = list()
        self.Eac = list()
        self.Eap = list()
        self.w = dict()
        self.v0 = None
        self.vg = None

        # For visualization purposes, we include a parallel array for each
        # (x, y) location of the vertexes in V.
        self.loc = list()

        # The max degree over all the vertexes, used to know the maximum
        # number of actions in the ToC SSP.
        self.maxOutgoingDegree = 0

        # The LOSM object which is optionally loadable from an XML file.
        self.losm = None

    def load(self, osmXMLFile):
        """ Load an OSM XML file and construct the ToC Path from that.

            Parameters:
                osmXMLFile  --  An XML file exported from OpenStreetMap.
        """

        self.losm = LOSMConverter()
        self.losm.open(osmXMLFile)

        #nodes = self.losm.nodes
        #edges = self.losm.edges
        nodes, edges = self.losm.simplified_graph()

        # The vertexes are the UIDs within the graph.
        self.V = [n.uid for n in nodes]

        # The edges are the vertexes above plus all valid successors following the graph.
        self.E = [(e.uid1, e.uid2) for e in edges] + [(e.uid2, e.uid1) for e in edges]

        self.maxOutgoingDegree = max([n.degree for n in nodes])

        # The autonomy-capable edges are those with a higher speed limit.
        self.Eac = [(e.uid1, e.uid2) for e in edges if e.speedLimit >= 30.0] + \
                    [(e.uid2, e.uid1) for e in edges if e.speedLimit >= 30.0]

        # The autonomy-preferred edges are the same. (Perhaps add a constraint on longer distances.)
        self.Eap = [(e.uid1, e.uid2) for e in edges if e.speedLimit >= 30.0] + \
                    [(e.uid2, e.uid1) for e in edges if e.speedLimit >= 30.0]

        # The weight is equal to the time on the road edge.
        self.w = {(e.uid1, e.uid2): e.distance / e.speedLimit * 3600.0 for e in edges}
        self.w.update({(e.uid2, e.uid1): e.distance / e.speedLimit * 3600.0 for e in edges})

        # The initial state and goal state are randomly chosen here. They must be different.
        self.v0 = 0
        self.vg = 0
        while self.v0 == self.vg:
            self.v0 = rnd.choice(self.V)
            self.vg = rnd.choice(self.V)

    def random(self, numVertexes=3, probAddEdge=0.25, probAutonomyCapable=0.5, probAutonomyPreferred=0.5):
        """ Create a random ToC Path object, given the number of desired vertexes.

            Parameters:
                numVertexes             --  The number of desired vertexes. Default is 3.
                probAddEdge             --  The probability of adding an edge during the process.
                                            Default is 0.25.
                probAutonomyCapable     --  The probability of being autonomy capable. Default is 0.5.
                probAutonomyPreferred   --  The probability of being autonomy preferred, given it is already
                                            autonomy capable. Default is 0.5.
        """

        self.V = list(range(numVertexes))

        self.E = list()
        for v in self.V:
            for vp in self.V:
                if v == vp:
                    continue

                # Note: To ensure it is connected, we build a chain, plus some
                # random probability of linking extra edges along it.
                if rnd.random() < probAddEdge or vp == v + 1 or vp == v - 1:
                    e = (v, vp)
                    self.E += [e]

        self.maxOutgoingDegree = max([len([e for e in self.E if e[0] == v]) for v in self.V])

        self.Eac = list()
        self.Eap = list()
        while len(self.Eac) == 0 and len(self.Eap) == 0:
            self.Eac = list()
            self.Eap = list()
            for e in self.E:
                if rnd.random() < probAutonomyCapable:
                    self.Eac += [e]
                    if rnd.random() < probAutonomyPreferred:
                        self.Eap += [e]

        self.w = {e: rnd.uniform(3.0, 10.0) for e in self.E}

        self.v0 = 0
        self.vg = numVertexes - 1

    def __str__(self):
        """ Print a pretty string of the ToCPath object.

            Returns:
                A pretty string of the object.
        """

        result = "Num Vertexes:           %i\n" % (len(self.V))
        result += "Initial Vertex:         %s\n" % (str(self.v0))
        result += "Goal Vertex:            %s\n\n" % (str(self.vg))

        result += "Num Edges:              %i\n" % (len(self.E))
        result += "Num Autonomy-Capable:   %i\n" % (len(self.Eac))
        result += "Num Autonomy-Preferred: %i\n\n" % (len(self.Eap))

        result += "Edges:\n"
        for e in self.E:
            result += "%s\n" % (str(e))
        result += "\n"

        result += "Weights:\n"
        for e in self.E:
            result += "w[%s] = %.3f\n" % (str(e), self.w[e])

        return result


if __name__ == "__main__":
    print("Performing ToCPath Unit Test...")

    tocpath = ToCPath()

    if len(sys.argv) == 2:
        tocpath.load(sys.argv[1])
    else:
        tocpath.random()

    print(tocpath)

    print("Done.")

