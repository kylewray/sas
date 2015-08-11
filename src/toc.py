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

class ToC(object):
    """ Transfer of Control (ToC) formal problem statement. """

    def __init__(self, randomize=None):
        """ The constructor for the ToC class.

            Parameters:
                randomize   --  Randomly assign values as (nh, nm, no, nt). Default is None.
        """

        self.H = list()
        self.M = list()
        self.O = list()
        self.T = dict()
        self.Ph = dict()
        self.Pc = dict()
        self.Po = dict()
        self.C = dict()

        if randomize is not None and len(randomize) == 4:
            self.random(randomize[0], randomize[1], randomize[2], randomize[3])

    def random(self, nh=2, nm=2, no=2, nt=5):
        """ Define a ToC problem using a random with an optional assignment of set sizes.

            Parameters:
                nh  --  The number of of human states.
                nm  --  The number of messages.
                no  --  The number of observations.
                nt  --  The number of time steps (tau).
        """

        self.H = ["h%i" % (i) for i in range(nh)]
        self.M = ["m%i" % (i) for i in range(nm)] + ["nop"]
        self.O = ["o%i" % (i) for i in range(no)]
        self.T = [int(i) for i in range(nt)]

        for h in self.H:
            for m in self.M:
                for t in self.T:
                    values = [rnd.random() for hp in self.H]
                    norm = sum(values)

                    for i, hp in enumerate(self.H):
                        self.Ph[(h, m, t, hp)] = values[i] / norm

        for h in self.H:
            for m in self.M:
                for t in self.T:
                    self.Pc[(h, m, t)] = rnd.random()

        for h in self.H:
            values = [rnd.random() for o in self.O]
            norm = sum(values)

            for i, o in enumerate(self.O):
                self.Po[(h, o)] = values[i] / norm

        for h in self.H:
            for m in self.M:
                for t in self.T:
                    self.C[(h, m, t)] = float(rnd.randint(1, 100))

    def __str__(self):
        """ Convert this ToC object to a string representation.

            Returns:
                The string representation of the ToC object.
        """

        result = "H = %s\n" % (str(self.H))
        result += "M = %s\n" % (str(self.M))
        result += "O = %s\n" % (str(self.O))
        result += "T = %s\n\n" % (str(self.T))

        result += "Ph(h, m, t, h'):\n"
        for key, value in self.Ph.items():
            result += "%s: %.3f\n" % (str(key), value)
        result += "\n"

        result += "Pc(h, m, t):\n"
        for key, value in self.Pc.items():
            result += "%s: %.3f\n" % (str(key), value)
        result += "\n"

        result += "Po(h, o):\n"
        for key, value in self.Po.items():
            result += "%s: %.3f\n" % (str(key), value)
        result += "\n"

        result += "C(h, m, t):\n"
        for key, value in self.C.items():
            result += "%s: %.3f\n" % (str(key), value)
        result += "\n"

        return result


if __name__ == "__main__":
    print("Performing ToC Unit Test...")

    toc = ToC(randomize=(4, 1, 3, 5))
    print(toc)

    print("Done.")

