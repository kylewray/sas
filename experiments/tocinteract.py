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


import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from toc import *


class ToCInteract(ToC):
    """ A custom ToC class for the interact experiments. """

    def __init__(self, nt=5):
        """ The constructor for the ToCInteract class.

            Parameters:
                nt  --  The number of time steps available. Must be at least 5. Default is 5.
        """

        super().__init__()

        if nt < 5:
            print("Warning: Assigned nt as 5 since the number provided was too small.")
            nt = 5


        self.H = ["attentive", "distracted"]
        self.M = ["visual", "visual and auditory", "nop"]
        self.O = ["attentive", "distracted"]
        self.T = [int(i) for i in range(nt + 1)]


        # NOP tends to keep the human state as it is. By default, all other messages follow
        # this pattern after long enough. These will be slightly overridden below.
        for m in self.M:
            for t in self.T:
                self.Ph[("attentive", m, t, "attentive")] = 0.9
                self.Ph[("attentive", m, t, "distracted")] = 0.1
                self.Ph[("distracted", m, t, "attentive")] = 0.1
                self.Ph[("distracted", m, t, "distracted")] = 0.9

        # The visual message is less likely to shift the human to attentive. It degrades
        # after 2 time steps to noise as above.
        for t in range(2):
            adjustment = 0.4 * (2 - t)
            self.Ph[("distracted", "visual", t, "attentive")] = 0.1 + adjustment
            self.Ph[("distracted", "visual", t, "distracted")] = 0.9 - adjustment

        # The visual and auditory message is less likely to shift the human to attentive. It degrades
        # after 4 time steps to noise as above.
        for t in range(4):
            adjustment = 0.2 * (4 - t)
            self.Ph[("distracted", "visual and auditory", t, "attentive")] = 0.1 + adjustment
            self.Ph[("distracted", "visual and auditory", t, "distracted")] = 0.9 - adjustment


        # NOP has zero chance to transfer control. Also, set the default values for the others here.
        for h in self.H:
            for m in self.M:
                for t in self.T:
                    self.Pc[(h, m, t)] = 0.0

        # The visual message is likely to transfer control, but only if the human is attentive. If
        # they are distracted, then there is a very low chance. The delay of this reaction is longer, too.
        for t in range(3):
            self.Pc[("attentive", "visual", t + 2)] = 0.05 + 0.1 * (3 - t)
            self.Pc[("distracted", "visual", t + 4)] = 0.0 + 0.025 * (3 - t)

        # The visual and auditory message is more likely to transfer control, in comparison. But
        # still has the same chance if distracted. The delay of this reaction is a bit shorter.
        for t in range(3):
            self.Pc[("attentive", "visual and auditory", t + 1)] = 0.05 + 0.2 * (3 - t)
            self.Pc[("distracted", "visual and auditory", t + 2)] = 0.05 + 0.05 * (3 - t)


        # Observations are a fairly accurate assessment of the true state, but not perfect.
        for h in self.H:
            for o in self.O:
                if h == o:
                    self.Po[(h, o)] = 0.8
                else:
                    self.Po[(h, o)] = 0.2


        # Finally, the costs are by default 0, especially for NOP.
        for h in self.H:
            for m in self.M:
                for t in self.T:
                    self.C[(h, m, t)] = 0.0

        # NOTE: Changing this range directly affects how long it will wait before resending
        # another message.

        # The visual message is only mildly annoying, but dies off quickly. It's also
        # less annoying if you were distracted, since already being attentive and
        # getting yelled at is of course annoying.
        for t in range(2):
            self.C[("attentive", "visual", t)] = max(1.0, 4.0 - 2.0 * t)
            self.C[("distracted", "visual", t)] = max(1.0, 4.0 - t)

        # Lastly, they are very high for the annoying extra auditory beeps. And you pay for
        # this cost for a much longer duration. Furthermore, it is also annoying when
        # you are distracted, since the sound is alarm-like, and breaks the user out of
        # sleep, good books, or research.
        for t in range(5):
            self.C[("attentive", "visual and auditory", t)] = max(1.0, 5.0 - t)
            self.C[("distracted", "visual and auditory", t)] = max(1.0, 2.0 - t)

