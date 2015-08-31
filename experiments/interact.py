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

import cv2

import time

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(thisFilePath)
from additional_functions import *
from tocinteract import *

sys.path.append(os.path.join(thisFilePath, "..", "src"))
from toc import *
from tocpomdp import *
from tocpath import *


class Interact(object):
    """ The interactive version of the ToC POMDP. """

    def __init__(self, width=1600, height=900):
        """ The constructor for the interactive version of the ToC POMDP.

            Parameters:
                width       --  The width of the window. Default is 1600.
                height      --  The height of the window. Default is 900.
        """

        self.width = width
        self.height = height

        self.window = None
        self.renderer = None

        self.spriteFactory = None
        self.tocFontMangager = None
        self.navFontManager = None

        self.messageText = None
        self.successText = None
        self.failureText = None
        self.abortedText = None

        self.navText = None
        self.navSymbols = None

        self.navMap = None

        self.audioRequestControl = None

        self.updateRate = 1000
        self.runTime = 0
        self.runTimeLastUpdate = 0

        self.navigating = True
        self.navDirections = ["straight" for i in range(5)]

        self.initialDelay = 1000
        self.initialDelayTime = 0

        self.paused = True
        self.buttonPressed = False

        self.toc = None
        self.tocpomdp = None
        self.tocpath = None

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

        self.faceCascade = None
        self.videoCapture = None
        self.videoFaces = [False for i in range(10)]

        self.videoOutputSDL = None
        self.videoOutputCam = None
        self.videoOutputFilePrefix = "video/" + str(int(round(time.time() * 1000)))

    def _reset(self):
        """ Reset the belief and other variables so that this can run again. """

        self.runTime = 0
        self.runTimeLastUpdate = 0

        self.navigating = True
        #self.navDirections = ["straight" for i in range(10)]

        self.initialDelayTime = 0

        self.paused = False
        self.buttonPressed = False

        self.action = None
        self.actionIndex = None
        self.observation = None
        self.observationIndex = None

        validInitialStates = [self.tocpomdp.states.index((len(self.toc.T) - 1, h, "nop", 0)) for h in self.toc.H]
        self.b = np.array([1.0 / len(validInitialStates) * (i in validInitialStates) for i in range(self.tocpomdp.n)])

        self._update_belief_factors()

    def _initialize_toc(self):
        """ Create and solve the ToC problem and ToC POMDP. """

        print("Initializing...")

        self.toc = ToC(randomize=(2, 2, 2, 3))
        #self.toc = ToCInteract()

        # TODO: Implement the path code.
        #self.tocpath = ToCPathInteract()
        #self.tocpath.

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

    def _initialize_sdl(self):
        """ Initialize the core components of SDL, the window, and the renderer. """

        sdl2.ext.init()

        self.window = sdl2.ext.Window("SAS - ToC POMDP - Interact", size=(self.width, self.height))
        self.window.show()

        self.renderer = sdl2.ext.Renderer(self.window)

        sdl2.SDL_ShowCursor(False)

    def _initialize_textures(self):
        """ Initialize textures. Must be called after the renderer is created. """

        print("Loading Textures...", end='')
        sys.stdout.flush()

        self.spriteFactory = sdl2.ext.SpriteFactory(renderer=self.renderer)

        self.tocFontManager = sdl2.ext.FontManager(font_path="fonts/OpenSans-Regular.ttf", size=64)
        self.navFontManager = sdl2.ext.FontManager(font_path="fonts/OpenSans-Regular.ttf", size=48)

        self.messageText = self.spriteFactory.from_text("Transfer of Control Requested", fontmanager=self.tocFontManager)
        self.successText = self.spriteFactory.from_text("Success", fontmanager=self.tocFontManager)
        self.failureText = self.spriteFactory.from_text("Failure", fontmanager=self.tocFontManager)
        self.abortedText = self.spriteFactory.from_text("Aborted", fontmanager=self.tocFontManager)

        navImages = {"int_cross":   ("Cross Ahead", "images/nav_intersection_cross.png"),
                     "int_t_left":  ("Left T Ahead", "images/nav_intersection_t_left.png"),
                     "int_t_right": ("Right T Ahead", "images/nav_intersection_t_right.png"),
                     "left_curve":  ("Slight Left Ahead", "images/nav_left_curve.png"),
                     "left_sharp":  ("Sharp Left Ahead", "images/nav_left_sharp.png"),
                     "left_turn":   ("Take Next Left", "images/nav_left_turn.png"),
                     "right_curve": ("Slight Right Ahead", "images/nav_right_curve.png"),
                     "right_sharp": ("Sharp Right Ahead", "images/nav_right_sharp.png"),
                     "right_turn":  ("Take Next Right", "images/nav_right_turn.png"),
                     "straight":    ("Continue Straight", "images/nav_straight.png")}

        self.navSymbols = dict()
        self.navText = dict()

        for key, values in navImages.items():
            self.navText[key] = self.spriteFactory.from_text(values[0], fontmanager=self.navFontManager)
            self.navSymbols[key] = self.spriteFactory.from_image(values[1])

        # TODO: Implement the path rendering code.
        # For demo purposes, we will load an external map texture.
        self.navMap = self.spriteFactory.from_image("images/demo_map.png")

        print("Done.")

    def _initialize_audio(self):
        """ Initialize audio and load wav files. Must be called after SDL has been initialized. """

        sdl2.sdlmixer.Mix_OpenAudio(22050, sdl2.sdlmixer.MIX_DEFAULT_FORMAT, 2, 4096)
        self.audioRequestControl = sdl2.sdlmixer.Mix_LoadMUS(b"audio/request.mp3")

    def _initialize_cv(self):
        """ Initialize the computer vision components. """

        cascadePath = "cv/haar_frontface_default.xml"
        videoDeviceIndex = 1

        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        self.videoCapture = cv2.VideoCapture(videoDeviceIndex)

        if not self.videoCapture.isOpened():
            print("Warning: Failed to open the video capture device.")

    def execute(self):
        """ The main execution loop of the program. """

        self._initialize_toc()
        self._initialize_sdl()
        self._initialize_textures()
        self._initialize_audio()
        self._initialize_cv()

        currentTime = sdl2.SDL_GetTicks()
        lastTime = currentTime

        self.running = True

        while self.running:
            # Handle all the SDL events.
            events = sdl2.ext.get_events()

            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    self.running = False

                self._check_keyboard(event)
                self._check_mouse(event)

            # Get the updated frame of the video and get the faces, assuming the object was
            # created successfully.
            if self.videoCapture is not None and self.videoCapture.isOpened():
                ret, frame = self.videoCapture.read()
                grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.faceCascade.detectMultiScale(grayscaleFrame, scaleFactor=1.1, minNeighbors=5,
                                                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                self.videoFaces = [len(faces) > 0] + self.videoFaces[:-1]

                #print(type(frame))
                #print(frame.dtype)
                #print(frame.shape)

                #surface = sdl2.SDL_GetWindowSurface(self.window.window)
                #print(type(surface))
                #rawPixels = sdl2.ext.pixels3d(surface)
                #sdl2.SDL_FreeSurface(surface)

                if self.videoOutputSDL is not None:
                    self.videoOutputCam.write(frame)

                if self.videoOutputCam is not None:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self.videoOutputCam.write(frame)

            # Perform the update but only every so often.
            currentTime = sdl2.SDL_GetTicks()

            if self.paused:
                # Do nothing if paused.
                pass
            elif self.navigating:
                # Update navigation otherwise.
                self._update_nav()
            elif self.initialDelayTime < self.initialDelay:
                # There is always an optional delay for demo purposes before initiating ToC.
                self.initialDelayTime += currentTime - lastTime
            else:
                # It only updates the POMDP following the updateRate variable.
                self.runTime += currentTime - lastTime

                if self.runTime > self.runTimeLastUpdate + self.updateRate:
                    self.runTimeLastUpdate += self.updateRate
                    self._update_toc()

            lastTime = currentTime

            # Render the frame to the window.
            self.renderer.color = sdl2.ext.Color(150, 150, 150)
            self.renderer.clear()
            self._render()
            self.renderer.present()

        self._uninitialize_cv()
        self._uninitialize_audio()
        self._uninitialize_textures()
        self._uninitialize_sdl()
        self._uninitialize_toc()

    def _uninitialize_cv(self):
        """ Uninitialize the OpenCV objects. """

        if self.videoOutputSDL is not None:
            self.videoOutputSDL.release()

        if self.videoOutputCam is not None:
            self.videoOutputCam.release()

        self.videoCapture.release()

    def _uninitialize_audio(self):
        """ Uninitialize the audio. """

        sdl2.sdlmixer.Mix_FreeMusic(self.audioRequestControl)
        sdl2.sdlmixer.Mix_CloseAudio()

    def _uninitialize_textures(self):
        """ Uninitialize the textures. """

        self.tocFontManager.close()
        self.navFontManager.close()

    def _uninitialize_sdl(self):
        """ Uninitialize SDL, the window, and the renderer. """

        sdl2.ext.quit()

    def _uninitialize_toc(self):
        """ Uninitialize the ToC POMDP and other variables. """

        pass

    def _update_nav(self):
        """ Update the navigation following the SSP. """

        pass

    def _update_toc(self):
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
                    self.b = np.array([1.0 * (s == self.observation) for s in self.tocpomdp.states])
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
            if sum(self.videoFaces) > len(self.videoFaces) / 2:
                self.observation = self.tocpomdp.observations[0] # "attentive"
            else:
                self.observation = self.tocpomdp.observations[1] # "distracted"

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
        """ Render the current POMDP belief, the last observation, and the message, or the navigation HUD. """

        # Based on how much time is remaining, render either the current message or
        # the final terminal state symbol. Only do this if not navigating.
        if self.navigating or self.initialDelayTime < self.initialDelay:
            self._render_nav()
        elif self.beliefFactorTimeRemaining is not None:
            self._render_message()
            self._render_attentiveness()
        elif self.beliefFactorEndResultState is not None:
            self._render_end_result()
            self._render_attentiveness()

    def _render_nav(self):
        """ Render the navigation component of the HUD. """

        offset = 20

        # Render the background for the map.
        sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                int(offset), int(offset),
                                int(self.width / 2 - offset), int(self.height - offset),
                                200, 200, 220, 255)
        sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                                int(offset), int(offset),
                                int(self.width / 2 - offset), int(self.height - offset),
                                220, 220, 240, 255)

        self.renderer.copy(self.navMap, dstrect=(int(offset), int(offset),
                                                int(self.width / 2 - offset * 2), int(self.height - offset * 2)))

        # Render the list of navigation directions to follow.
        navOffset = 190
        navVerticalTextOffset = 10
        navSymbolSize = 100

        for i, nav in enumerate(self.navDirections):
            self.renderer.copy(self.navSymbols[nav],
                            dstrect=(int(self.width / 2 + offset),
                                    int(offset + navOffset * i),
                                    navSymbolSize,
                                    navSymbolSize))
                                    #self.navSymbols[nav].size[0],
                                    #self.navSymbols[nav].size[1]))

            self.renderer.copy(self.navText[nav],
                            dstrect=(int(self.width / 2 + offset * 2 + navSymbolSize),
                                    int(offset + navOffset * i + navVerticalTextOffset),
                                    self.navText[nav].size[0],
                                    self.navText[nav].size[1]))

    def _render_message(self):
        """ Render the message as given by the variable 'previousAction'. """

        self._render_button_pressed()

        maximal = len(self.toc.T) + 1
        total = maximal - self.runTime / self.updateRate

        # Render the count down timer as a faint 'pie' shape.
        #if self.action == "nop":
        sdl2.sdlgfx.filledPieRGBA(self.renderer.renderer,
                                  int(self.width / 2), int(self.height / 2),
                                  int(min(self.width, self.height) / 3),
                                  0, int(360.0 * (total / maximal)),
                                  20, 20, 20, 255)
        sdl2.sdlgfx.pieRGBA(self.renderer.renderer,
                                  int(self.width / 2), int(self.height / 2),
                                  int(min(self.width, self.height) / 3),
                                  0, int(360.0 * (total / maximal)),
                                  100, 100, 100, 255)

        # Render the blinking light and text message.
        if self.action == "m0":
            if sdl2.sdlmixer.Mix_PlayingMusic() == 0:
                sdl2.sdlmixer.Mix_PlayMusic(self.audioRequestControl, 1)

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - self.messageText.size[0] / 2 - 30), int(self.height / 2 - self.messageText.size[1] / 2 - 30),
                                    int(self.width / 2 + self.messageText.size[0] / 2 + 30), int(self.height / 2 + self.messageText.size[1] / 2 + 30),
                                    20, 150, 20, 200)
            sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                                    int(self.width / 2 - self.messageText.size[0] / 2 - 30), int(self.height / 2 - self.messageText.size[1] / 2 - 30),
                                    int(self.width / 2 + self.messageText.size[0] / 2 + 30), int(self.height / 2 + self.messageText.size[1] / 2 + 30),
                                    80, 220, 80, 255)

            self.renderer.copy(self.messageText, dstrect=(int(self.width / 2 - self.messageText.size[0] / 2),
                                                         int(self.height / 2 - self.messageText.size[1] / 2),
                                                         self.messageText.size[0],
                                                         self.messageText.size[1]))

        elif self.action == "m1":
            if sdl2.sdlmixer.Mix_PlayingMusic() == 0:
                sdl2.sdlmixer.Mix_PlayMusic(self.audioRequestControl, 1)

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    0, 0,
                                    self.width, self.height,
                                    150, 20, 20, 200)
            sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                                    0, 0,
                                    self.width, self.height,
                                    220, 80, 80, 255)

            self.renderer.copy(self.messageText, dstrect=(int(self.width / 2 - self.messageText.size[0] / 2),
                                                         int(self.height / 2 - self.messageText.size[1] / 2),
                                                         self.messageText.size[0],
                                                         self.messageText.size[1]))

    def _render_end_result(self):
        """ Render the end result state: text plus the symbol of square, x-square, or diamond. """

        size = min(self.width, self.height) / 3

        if self.beliefFactorEndResultState == "success":
            self._render_button_pressed()

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    20, 150, 20, 200)
            sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    80, 220, 80, 255)

            self.renderer.copy(self.successText, dstrect=(int(self.width / 2 - self.successText.size[0] / 2),
                                                         int(self.height / 2 - self.successText.size[1] / 2),
                                                         self.successText.size[0],
                                                         self.successText.size[1]))

        elif self.beliefFactorEndResultState == "failure":
            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    150, 20, 20, 200)
            sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size), int(self.height / 2 - size),
                                    int(self.width / 2 + size), int(self.height / 2 + size),
                                    220, 80, 80, 255)

            sdl2.sdlgfx.thickLineRGBA(self.renderer.renderer,
                                    int(self.width / 2 - size - 50), int(self.height / 2 - size - 50),
                                    int(self.width / 2 + size + 50), int(self.height / 2 + size + 50),
                                    32,
                                    180, 30, 30, 255)
            sdl2.sdlgfx.thickLineRGBA(self.renderer.renderer,
                                    int(self.width / 2 + size + 50), int(self.height / 2 - size - 50),
                                    int(self.width / 2 - size - 50), int(self.height / 2 + size + 50),
                                    32,
                                    180, 30, 30, 255)

            self.renderer.copy(self.failureText, dstrect=(int(self.width / 2 - self.failureText.size[0] / 2),
                                                         int(self.height / 2 - self.failureText.size[1] / 2),
                                                         self.failureText.size[0],
                                                         self.failureText.size[1]))

        elif self.beliefFactorEndResultState == "aborted":
            array_4_short = ct.c_int16 * (4)
            x = array_4_short(*np.array([int(self.width / 2),
                                         int(self.width / 2 + size),
                                         int(self.width / 2),
                                         int(self.width / 2 - size)]))
            y = array_4_short(*np.array([int(self.height / 2 - size),
                                         int(self.height / 2),
                                         int(self.height / 2 + size),
                                         int(self.height / 2)]))

            sdl2.sdlgfx.filledPolygonRGBA(self.renderer.renderer,
                                    x, y, 4,
                                    20, 20, 150, 200)
            sdl2.sdlgfx.polygonRGBA(self.renderer.renderer,
                                    x, y, 4,
                                    80, 80, 220, 255)

            self.renderer.copy(self.abortedText, dstrect=(int(self.width / 2 - self.abortedText.size[0] / 2),
                                                         int(self.height / 2 - self.abortedText.size[1] / 2),
                                                         self.abortedText.size[0],
                                                         self.abortedText.size[1]))

    def _render_button_pressed(self):
        """ Render a light change in the background color to denote when the user successfully clicks the screen. """

        if self.buttonPressed:
            buttonPressedOffset = 20

            sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                                int(buttonPressedOffset), int(buttonPressedOffset),
                                int(self.width - buttonPressedOffset), int(self.height - buttonPressedOffset),
                                200, 200, 200, 255)

    def _render_attentiveness(self):
        """ Render the light bar at the bottom which scales between the driver being marked as attentive or not. """

        offset = 20
        height = 40

        sdl2.sdlgfx.boxRGBA(self.renderer.renderer,
                            int(offset),
                            int(self.height - offset - height),
                            int((self.width - offset * 2) * sum(self.videoFaces) / len(self.videoFaces) + offset),
                            int(self.height - offset),
                            80, 80, 80, 255)
        sdl2.sdlgfx.rectangleRGBA(self.renderer.renderer,
                            int(offset),
                            int(self.height - offset - height),
                            int((self.width - offset * 2) * sum(self.videoFaces) / len(self.videoFaces) + offset),
                            int(self.height - offset),
                            140, 140, 140, 255)

    def _check_keyboard(self, event):
        """ Check the keyboard input given the event.

            Parameters:
                event   --  The event provided.
        """

        if event.type == sdl2.SDL_KEYUP:
            if event.key.keysym.sym in [sdl2.SDLK_ESCAPE, sdl2.SDLK_q]:
                self.running = False

            if event.key.keysym.sym == sdl2.SDLK_RETURN:
                if self.navigating:
                    print("Executing Transfer of Control...")
                self.navigating = False

            if event.key.keysym.sym == sdl2.SDLK_SPACE:
                self.paused = not self.paused
                if self.paused:
                    print("Interact Experiment: Paused.")
                else:
                    print("Interact Experiment: Unpaused.")

            if event.key.keysym.sym == sdl2.SDLK_r:
                self._reset()
                print("Reset interact experiment.")

            if event.key.keysym.sym == sdl2.SDLK_s:
                if self.videoOutputSDL is None and self.videoOutputCam is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.videoOutputSDL = cv2.VideoWriter(self.videoOutputFilePrefix + "_sdl.avi", fourcc, 20.0, (640, 480))

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.videoOutputCam = cv2.VideoWriter(self.videoOutputFilePrefix + "_cam.avi", fourcc, 20.0, (640, 480))

                    print("Video Output: Enabled.")
                else:
                    if self.videoOutputSDL is not None:
                        self.videoOutputSDL.release()
                    self.videoOutputSDL = None

                    if self.videoOutputCam is not None:
                        self.videoOutputCam.release()
                    self.videoOutputCam = None

                    print("Video Output: Disabled.")

            navChoices = list()

            if event.key.keysym.sym == sdl2.SDLK_LEFT:
                navChoices = ["left_curve", "left_sharp", "left_turn"]

            if event.key.keysym.sym == sdl2.SDLK_RIGHT:
                navChoices = ["right_curve", "right_sharp", "right_turn"]

            if event.key.keysym.sym == sdl2.SDLK_UP:
                navChoices = ["straight"]

            if event.key.keysym.sym == sdl2.SDLK_DOWN:
                navChoices = ["int_cross", "int_t_left", "int_t_right"]

            if len(navChoices) > 0:
                self.navDirections = [rnd.choice(navChoices)] + self.navDirections[:-1]

    def _check_mouse(self, event):
        """ Check the mouse input given the event.

            Parameters:
                event   --  The event provided.
        """

        if event.type == sdl2.SDL_MOUSEBUTTONUP:
            if self.navigating:
                # Do nothing yet if navigating.
                pass
            else:
                self.buttonPressed = True


if __name__ == "__main__":
    print("Executing Interact Experiment...")

    interact = Interact()
    interact.execute()

    print("Done.")

