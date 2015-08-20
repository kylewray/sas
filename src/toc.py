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
        self.T = [int(i) for i in range(nt + 1)]

        for k, h in enumerate(self.H):
            for i, m in enumerate(self.M):
                for t in self.T:
                    # NOP has a uniform random noise chance of transitioning
                    # the human state around. Or, if there were more messages
                    # than human states, then they change the state at random.
                    if m == "nop" or i > k:
                        values = [rnd.random() for hp in self.H]
                        norm = sum(values)

                        for j, hp in enumerate(self.H):
                            self.Ph[(h, m, t, hp)] = values[j] / norm

                    # Otherwise, each message type has a high likelihood of
                    # transitioning the human to the corresponding human state.
                    else:
                        values = [rnd.random() + float(len(self.H) * int(i == j)) for j, hp in enumerate(self.H)]
                        norm = sum(values)

                        for j, hp in enumerate(self.H):
                            self.Ph[(h, m, t, hp)] = values[j] / norm

        for j, h in enumerate(self.H):
            for i, m in enumerate(self.M):
                #values = [rnd.uniform(0.0, 0.25) / pow(float(t + 1), float(i) * 3.0 / float(len(self.M)) + 0.25) for t in self.T]
                values = [rnd.uniform(0.5 * float(t) / float(len(self.T)), \
                                      0.5 * float((t + 1)) / float(len(self.T))) for t in self.T]
                values = sorted(values, reverse=True)

                for t in self.T:
                    # The probability of transferring control slowly decreases over time.
                    # Messages with lower index raise this probability, but cost much more.
                    # Also, the lower human state index, the better chance of transferring
                    # control; i.e., this is desired.
                    if values[t] > 0.3:
                        self.Pc[(h, m, t)] = values[t] * float(pow(len(self.M) - 1 - i, 2)) / float(pow(len(self.M), 2)) * float(len(self.H) - j) / float(len(self.H))
                    else:
                        self.Pc[(h, m, t)] = 0.0

        for i, h in enumerate(self.H):
            # Note: It is much more likely to make a particular observation if the human
            # is in a particular state. If there are more human states than observations,
            # then some human states we will always be very uncertain about. Conversely,
            # if there are more observations than human states, then we will make
            # quasi-uniformly weaker observations about human states.
            values = [rnd.random() + float(len(self.O) * int(i == j)) for j, o in enumerate(self.O)]
            norm = sum(values)

            for i, o in enumerate(self.O):
                self.Po[(h, o)] = values[i] / norm

        for h in self.H:
            for i, m in enumerate(self.M):
                for t in self.T:
                    msgCost = rnd.uniform(0.25, 1.0) * 10.0

                    # It doesn't matter which of the new messages you send; they are only
                    # dependent on the previous message and how long it has been since you've
                    # sent it.
                    for mp in self.M:
                        # NOP costs nothing.
                        if mp == "nop":
                            self.C[(h, m, t, mp)] = 0.0

                        # Other messages cost more for lower-indexes, but all of them degrade in
                        # cost over time.
                        if mp != "nop":
                            # Messages with a lower index cost more.
                            msgCost += abs((i + 1) / len(self.M) - (len(self.T) - t) / len(self.T))
                            self.C[(h, m, t, mp)] = msgCost + 0.01

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
        for h, m, t, hp in it.product(self.H, self.M, self.T, self.H):
            result += "(%s, %s, %i, %s): %.3f\n" % (h, m, t, hp, self.Ph[(h, m, t, hp)])
        result += "\n"

        result += "Pc(h, m, t):\n"
        for h, m, t in it.product(self.H, self.M, self.T):
            result += "(%s, %s, %i): %.8f\n" % (h, m, t, self.Pc[(h, m, t)])
        result += "\n"

        result += "Po(h, o):\n"
        for h, o in it.product(self.H, self.O):
            result += "(%s, %s): %.3f\n" % (h, o, self.Po[(h, o)])
        result += "\n"

        result += "C(h, m, t, m'):\n"
        for h, m, t, mp in it.product(self.H, self.M, self.T, self.M):
            result += "(%s, %s, %i, %s): %.3f\n" % (h, m, t, mp, self.C[(h, m, t, mp)])
        result += "\n"

        return result


if __name__ == "__main__":
    print("Performing ToC Unit Test...")

    toc = ToC(randomize=(3, 2, 3, 5))
    print(toc)

    print("Done.")

