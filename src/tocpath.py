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


class ToCPath(object):
    """ A class which holds the weighted directed graph of the ToC Path. """

    def __init__(self):
        """ The constructor for the ToC Path class. """

        self.V = list()
        self.E = list()
        self.w = dict()
        self.v0 = None
        self.vg = None

        # For visualization purposes, we include a parallel array for each
        # (x, y) location of the vertexes in V.
        self.loc = list()

    def create(self, numVertexes=3, probAddEdge=0.25, probAutonomyCapable=0.5, probAutonomyPreferred=0.5):
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
        self.Vac = list()
        self.Vap = list()
        while len(self.Vac) == 0 and len(self.Vap) == 0:
            self.Vac = list()
            self.Vap = list()
            for v in self.V:
                if rnd.random() < probAutonomyCapable:
                    self.Vac += [v]
                    if rnd.random() < probAutonomyPreferred:
                        self.Vap += [v]

        self.w = [rnd.uniform(3.0, 10.0) for v in self.V]

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

        self.v0 = 0
        self.vg = numVertexes - 1

    def __str__(self):
        """ Print a pretty string of the ToCPath object.

            Returns:
                A pretty string of the object.
        """

        result = "Num Vertexes:       %i\n" % (len(self.V))
        result += "Autonomy Capable:   %s\n" % (str(self.Vac))
        result += "Autonomy Preferred: %s\n" % (str(self.Vap))
        result += "Num Edges:          %i\n" % (len(self.E))
        result += "Initial Vertex:     %i\n" % (self.v0)
        result += "Goal Vertex:        %i\n\n" % (self.vg)

        result += "Edges:\n"
        for e in self.E:
            result += "%s\n" % (str(e))
        result += "\n"

        result += "Weights:\n"
        for v in self.V:
            result += "w[%i] = %.3f\n" % (v, self.w[v])

        return result


if __name__ == "__main__":
    print("Performing ToCPath Unit Test...")

    tocpath = ToCPath()
    tocpath.create()
    print(tocpath)

    print("Done.")

