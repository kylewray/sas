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

import sdl2
import sdl2.ext
import sdl2.sdlgfx
import sdl2.sdlmixer

import numpy as np
import ctypes as ct

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(thisFilePath)
from additional_functions import *

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from toc import *
from tocpomdp import *


class Interact(object):
    """ The interactive version of the ToC POMDP. """

    def __init__(self, width=1024, height=768, scale=1.0):
        """ The constructor for the interactive version of the ToC POMDP.

            Parameters:
                width   --  The width of the window. Default is 1024.
                height  --  The height of the window. Default is 768.
                scale   --  The scale of objects within the window. Default is 1.0.
        """

        self.width = width
        self.height = height
        self.scale = scale

        self.window = None
        self.renderer = None

        self.messageText = None
        self.successText = None
        self.failureText = None
        self.abortedText = None
        self.audioRequestControl = None

        self.updateRate = 1000
        self.lastTime = 0
        self.currentTime = 0

        self.paused = False
        self.buttonPressed = False

        self.toc = None
        self.tocpomdp = None

        self.Gamma = None
        self.pi = None

        self.calE = ["success", "failure", "aborted"]

        self.action = None
        self.actionIndex = None
        self.observation = None
        self.observationIndex = None
        self.b = None

        self.beliefFactorTimeRemaining = None
        self.beliefFactorHumanState = None
        self.beliefFactorLastMessage = None
        self.beliefFactorTimeSinceLastMessage = None
        self.beliefFactorEndResultState = None

    def initialize(self):
        """ Create and solve the ToC problem and ToC POMDP. """

        print("Initializing...")

        self.toc = ToC(randomize=(2, 2, 2, 5))

        print("Create POMDP... ", end='')
        sys.stdout.flush()

        self.tocpomdp = ToCPOMDP()
        self.tocpomdp.create(self.toc)

        print("Done.")
        print("Solve POMDP... ", end='')
        sys.stdout.flush()

        self.Gamma, self.pi, timings = self.tocpomdp.solve()

        self._reset()
        self._update_belief_factors()

        print("Done.")

    def _reset(self):
        """ Reset the belief and other variables so that this can run again. """

        self.lastTime = sdl2.SDL_GetTicks()
        self.currentTime = self.lastTime + self.updateRate

        self.paused = False
        self.buttonPressed = False

        self.action = None
        self.actionIndex = None
        self.observation = None
        self.observationIndex = None

        validInitialStates = [self.tocpomdp.states.index((len(self.toc.T) - 1, h, "nop", 0)) for h in self.toc.H]
        self.b = np.array([1.0 / len(validInitialStates) * (i in validInitialStates) for i in range(self.tocpomdp.n)])

        self._update_belief_factors()

    def _initialize_core(self):
        """ Initialize the core components of SDL, the window, and the renderer. """

        sdl2.ext.init()

        self.window = sdl2.ext.Window("SAS - ToC POMDP - Interact", size=(self.width, self.height))
        self.window.show()

        self.renderer = sdl2.ext.Renderer(self.window)

    def _initialize_textures(self):
        """ Initialize textures. Must be called after the renderer is created. """

        self.fontManager = sdl2.ext.FontManager(font_path="fonts/OpenSans-Regular.ttf", size=48)
        self.spriteFactory = sdl2.ext.SpriteFactory(renderer=self.renderer)

        self.messageText = self.spriteFactory.from_text("Transfer of Control Requested", fontmanager=self.fontManager)
        self.successText = self.spriteFactory.from_text("Success", fontmanager=self.fontManager)
        self.failureText = self.spriteFactory.from_text("Failure", fontmanager=self.fontManager)
        self.abortedText = self.spriteFactory.from_text("Aborted", fontmanager=self.fontManager)

    def _initialize_audio(self):
        """ Initialize audio and load wav files. Must be called after SDL has been initialized. """

        sdl2.sdlmixer.Mix_OpenAudio(22050, sdl2.sdlmixer.MIX_DEFAULT_FORMAT, 2, 4096)
        self.audioRequestControl = sdl2.sdlmixer.Mix_LoadMUS(b"audio/request.mp3")

    def execute(self):
        """ The main execution loop of the program. """

        self._initialize_core()
        self._initialize_textures()
        self._initialize_audio()

        running = True
        while running:
            events = sdl2.ext.get_events()

            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False

                self._check_keyboard(event)
                self._check_mouse(event)

            # Perform the update but only every so often.
            self.currentTime = sdl2.SDL_GetTicks()
            if self.currentTime > self.lastTime + self.updateRate:
                self.lastTime = self.lastTime + self.updateRate
                self._update()

            # Render the frame to the window.
            self.renderer.color = sdl2.ext.Color(0, 0, 0)
            self.renderer.clear()
            self._render()
            self.renderer.present()

        self._uninitialize_audio()
        self._uninitialize_textures()
        self._uninitialize_sdl()

    def _uninitialize_audio(self):
        """ Uninitialize the audio. """

        sdl2.sdlmixer.Mix_FreeMusic(self.audioReqestControl)
        sdl2.sdlmixer.Mix_CloseAudio()

    def _uninitialize_textures(self):
        """ Uninitialize the textures. """

        pass

    def _uninitialize_sdl(self):
        """ Uninitialize SDL, the window, and the renderer. """

        sdl2.ext.quit()

    def _update(self):
        """ Update the belief of the POMDP, decide to take an action, and make an observation. """

        print("------------------------------------------------")

        # This structure guarantees that the first pass through the update function takes an action.
        # Since this is a real application of a POMDP, and actions are taken at the current state,
        # with observations being observed at the next state, we have to wait for the human to
        # essentially respond to the action before making an observation. Hence the 'delay' here.
        if self.action is not None:
            # We use OpenCV to make an observation, or observe the end result state.
            self._update_make_observation()
            print("Observation Made:               %s" % (self.observation))

            # This will fail if there is an impossible observation. Since our model assigns various
            # zero probabilities (e.g., after 111 years of waiting for a message, there's a 0% chance
            # of transfer of control), if the user transfers control here it will be `impossible'
            # within the model, even though it is possible after 111 years, technically. So, we handle
            # this case by using the observation to assign the belief.
            try:
                self.b = update_belief(self.tocpomdp, self.b, self.actionIndex, self.observationIndex)
            except Exception:
                if self.observation in self.calE:
                    self.b = np.array([1.0 * (s == self.observation) for s in self.pomdp.states])
                else:
                    raise Exception()

            self._update_belief_factors()

        if self.beliefFactorTimeRemaining is not None:
            print("Time Remaining Factor:          %i" % (self.beliefFactorTimeRemaining))
            print("Human State Factor:             %s" % (str(self.beliefFactorHumanState)))
            print("Last Message Factor:            %s" % (self.beliefFactorLastMessage))
            print("Time Since Last Message Factor: %i" % (self.beliefFactorTimeSinceLastMessage))
        elif self.beliefFactorEndResultState is not None:
            print("End Result State:               %s" % (self.beliefFactorEndResultState))

        #print("Belief:             %s" % (str(["%s: %.2f" % (str(self.tocpomdp.states[i]), self.b[i]) \
        #                                        for i in range(self.tocpomdp.n) if self.b[i] > 0.0])))

        # With the potential new information from the observation, we take an action.
        self.actionIndex, v = take_action(self.tocpomdp, self.Gamma, self.pi, self.b)
        self.action = self.tocpomdp.actions[self.actionIndex]

        print("Action Taken:                   %s" % (self.action))
        print("------------------------------------------------")

    def _update_make_observation(self):
        """ Make an observation using OpenCV, or set it as the final end result state if done. """

        # The user's input changes the observation to "success" in the button handling code.
        if self.observation == "success" or (self.observation not in self.calE and self.action != "abort" and \
                self.beliefFactorLastMessage != "nop" and self.buttonPressed):
            self.observation = "success"

        # If we run out of time, we have failed forever.
        elif self.observation == "failure" or (self.observation not in self.calE and self.action != "abort" and \
                self.beliefFactorTimeRemaining is not None and self.beliefFactorTimeRemaining == 0):
            self.observation = "failure"

        # If the vehicle decides to abort, then this sets or preserves the aborted observation.
        elif self.observation == "aborted" or (self.observation not in self.calE and self.action == "abort"):
            self.observation = "aborted"

        # Finally, if we still have time left, then use OpenCV to detect the human state.
        elif self.beliefFactorTimeRemaining is not None and self.beliefFactorTimeRemaining > 0:
            # TODO: Use OpenCV here to get the human state of 'attentive' or 'distracted'.
            self.observation = self.tocpomdp.observations[0] # Implement this line.

        self.observationIndex = self.tocpomdp.observations.index(self.observation)

    def _update_belief_factors(self):
        """ Get the components (factors) of the belief state. """

        nonZeroBeliefStates = [(self.tocpomdp.states[i], self.b[i]) for i in range(self.tocpomdp.n) if self.b[i] > 0.0]

        # Note: As proven formally, belief is only over human states. Thus, it doesn't matter which
        # non-zero belief state we pick to get the other state factor information.
        if nonZeroBeliefStates[0][0] in self.calE:
            self.beliefFactorTimeRemaining = None
            self.beliefFactorHumanState = None
            self.beliefFactorLastMessage = None
            self.beliefFactorTimeSinceLastMessage = None
            self.beliefFactorEndResultState = nonZeroBeliefStates[0][0]
        else:
            nonZeroBeliefStates = sorted(nonZeroBeliefStates, key=lambda x: self.toc.H.index(x[0][1])) # Human factor = [1].

            self.beliefFactorTimeRemaining = nonZeroBeliefStates[0][0][0]
            self.beliefFactorHumanState = [(s[1], b) for s, b in nonZeroBeliefStates]
            self.beliefFactorLastMessage = nonZeroBeliefStates[0][0][2]
            self.beliefFactorTimeSinceLastMessage = nonZeroBeliefStates[0][0][3]
            self.beliefFactorEndResultState = None

    def _render(self):
        """ Render the current POMDP belief, the last observation, and the message """

        # Based on how much time is remaining, render either the current message or
        # the final terminal state symbol.
        if self.beliefFactorTimeRemaining is not None:
            self._render_message()
        elif self.beliefFactorEndResultState is not None:
            self._render_end_result()

    def _render_end_result(self):
        """ Render the end result state: text plus the symbol of square, x-square, or diamond. """

        # TODO: Implement. Draws square, x-square, or diamond with accompanying text.
        size = min(self.width, self.height) / 3

        if self.beliefFactorEndResultState == "success":
            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    20, 150, 20, 255)

            self.renderer.copy(self.successText, dstrect=(int(self.width / 2 - self.successText.size[0] / 2),
                                                         int(self.height / 2 - self.successText.size[1] / 2),
                                                         self.successText.size[0],
                                                         self.successText.size[1]))

        elif self.beliefFactorEndResultState == "failure":
            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    150, 20, 20, 255)

            self.renderer.copy(self.failureText, dstrect=(int(self.width / 2 - self.failureText.size[0] / 2),
                                                         int(self.height / 2 - self.failureText.size[1] / 2),
                                                         self.failureText.size[0],
                                                         self.failureText.size[1]))

        elif self.beliefFactorEndResultState == "aborted":
            sdl2.sdlgfx.polygonRGBA(self.renderer.renderer,
                                    np.array([int(size / 2), int(size * 3 / 4), int(size / 2), int(size / 4)]),
                                    np.array([int(size / 4), int(size / 2), int(size * 3 / 4), int(size / 2)]),
                                    4,
                                    150, 150, 150, 150)

            self.renderer.copy(self.abortedText, dstrect=(int(self.width / 2 - self.abortedText.size[0] / 2),
                                                         int(self.height / 2 - self.abortedText.size[1] / 2),
                                                         self.abortedText.size[0],
                                                         self.abortedText.size[1]))


    def _render_message(self):
        """ Render the message as given by the variable 'previousAction'. """

        maximal = len(self.toc.T) + 1
        total = maximal - self.currentTime / self.updateRate

        # Render the count down timer as a faint 'pie' shape.
        if self.action == "nop":
            sdl2.sdlgfx.filledPieRGBA(self.renderer.renderer,
                                      int(self.width / 2), int(self.height / 2),
                                      int(min(self.width, self.height) / 3),
                                      0, int(360.0 * (total / maximal)),
                                      #0, int(min(360.0, 360.0 * (self.currentTime - self.lastTime) / self.updateRate)),
                                      20, 20, 20, 255)
            sdl2.sdlgfx.pieRGBA(self.renderer.renderer,
                                      int(self.width / 2), int(self.height / 2),
                                      int(min(self.width, self.height) / 3),
                                      0, int(360.0 * (total / maximal)),
                                      #0, int(min(360.0, 360.0 * (self.currentTime - self.lastTime) / self.updateRate)),
                                      100, 100, 100, 255)

        # Render the blinking light and text message.
        if self.action == "m0":
            if sdl2.sdlmixer.Mix_PlayingMusic() == 0:
                sdl2.sdlmixer.Mix_PlayMusic(self.audioRequestControl, 1)

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - self.messageText.size[0] / 2 - 30), int(self.height / 2 - self.messageText.size[1] / 2 - 30),
                                    int(self.width / 2 + self.messageText.size[0] / 2 + 30), int(self.height / 2 + self.messageText.size[1] / 2 + 30),
                                    150, 20, 20, 255)

            self.renderer.copy(self.messageText, dstrect=(int(self.width / 2 - self.messageText.size[0] / 2),
                                                         int(self.height / 2 - self.messageText.size[1] / 2),
                                                         self.messageText.size[0],
                                                         self.messageText.size[1]))

        elif self.action == "m1":
            if sdl2.sdlmixer.Mix_PlayingMusic() == 0:
                sdl2.sdlmixer.Mix_PlayMusic(self.audioRequestControl, 1)

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - self.messageText.size[0] / 2 - 30), int(self.height / 2 - self.messageText.size[1] / 2 - 30),
                                    int(self.width / 2 + self.messageText.size[0] / 2 + 30), int(self.height / 2 + self.messageText.size[1] / 2 + 30),
                                    20, 150, 20, 255)

            self.renderer.copy(self.messageText, dstrect=(int(self.width / 2 - self.messageText.size[0] / 2),
                                                         int(self.height / 2 - self.messageText.size[1] / 2),
                                                         self.messageText.size[0],
                                                         self.messageText.size[1]))



    def _check_keyboard(self, event):
        """ Check the keyboard input given the event.

            Parameters:
                event   --  The event provided.
        """

        # TODO: Implement a reset key, and maybe a play/pause key.
        pass

    def _check_mouse(self, event):
        """ Check the mouse input given the event.

            Parameters:
                event   --  The event provided.
        """

        if event.type == sdl2.SDL_MOUSEBUTTONUP:
            self.buttonPressed = True


if __name__ == "__main__":
    print("Executing Interact Experiment...")

    interact = Interact()
    interact.initialize()
    interact.execute()

    print("Done.")

