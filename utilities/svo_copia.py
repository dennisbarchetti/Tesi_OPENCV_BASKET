import cv2 as cv
import numpy as np
import imutils
import sys


class StandardVideoOperations:  # class for manage the operation on the video

    # instances of the KNN Background Substractor
    KNN_SX = cv.createBackgroundSubtractorKNN(history=200)
    KNN_DX = cv.createBackgroundSubtractorKNN(history=200)

    # when a new instance of StandardVideoOperations is created is initialized with (0, 0) as default values for all ROIs
    def __init__(self):
        self.upper_left_LEFT = (0, 0)
        self.bottom_right_LEFT = (0, 0)
        self.upper_left_RIGHT = (0, 0)
        self.bottom_right_RIGHT = (0, 0)

    # set the values for the left ROI
    def set_left(self, upper_left, bottom_right):
        if (len(upper_left) != 2 or len(bottom_right)) != 2:
            sys.exit("error: upper_left and bottom_right must be arrays with 2 items")
        if not all(isinstance(x, int) for x in upper_left) or not all(isinstance(x, int) for x in bottom_right):
            sys.exit("error: upper_left and bottom_right must contain only integers")
        self.upper_left_LEFT = upper_left
        self.bottom_right_LEFT = bottom_right

    # set the values for the right ROI
    def set_right(self, upper_left, bottom_right):
        if (len(upper_left) != 2 or len(bottom_right)) != 2:
            sys.exit("error: upper_left and bottom_right must be arrays with 2 items")
        if not all(isinstance(x, int) for x in upper_left) or not all(isinstance(x, int) for x in bottom_right):
            sys.exit("error: upper_left and bottom_right must contain only integers")
        self.upper_left_RIGHT = upper_left
        self.bottom_right_RIGHT = bottom_right

    # return a frame cutted cutted as specified in the parameters
    @staticmethod
    def video_cutter(frame, upper_left, bottom_right):
        if(len(upper_left) != 2 or len(bottom_right)) != 2:
            sys.exit("error: upper_left and bottom_right must be arrays with 2 items")
        if not all(isinstance(x, int) for x in upper_left) or not all(isinstance(x, int) for x in bottom_right):
            sys.exit("error: upper_left and bottom_right must contain only integers")
        if not isinstance(frame, np.ndarray):
            sys.exit("error: frame must be of type numpy.ndarray")
        if frame.ndim != 3:
            sys.exit("error: frame should be a matrix of three dimensions")
        rect_frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        return rect_frame

    # cut the frame with the ROI values setted with set_left method
    def cut_left(self, startingFrame):
        if not isinstance(startingFrame, np.ndarray):
            sys.exit("error: startingFrame must be of type numpy.ndarray")
        if startingFrame.ndim != 3:
            sys.exit("error: startingFrame should be a matrix of three dimensions")
        leftCut = StandardVideoOperations.video_cutter(startingFrame, self.upper_left_LEFT, self.bottom_right_LEFT)
        return leftCut

    # cut the frame in params with the ROI values setted with set_right method
    def cut_right(self, startingFrame):
        if not isinstance(startingFrame, np.ndarray):
            sys.exit("error: startingFrame must be of type numpy.ndarray")
        if startingFrame.ndim != 3:
            sys.exit("error: startingFrame should be a matrix of three dimensions")
        rightCut = StandardVideoOperations.video_cutter(startingFrame, self.upper_left_RIGHT, self.bottom_right_RIGHT)
        return rightCut
    
    # draw an empty rectangle with the specified color and position
    @staticmethod
    def draw_rectangle(frameBGR, upperLeft, bottomRight, color_string):
        if not isinstance(frameBGR, np.ndarray):
            sys.exit("error: frameBGR must be of type numpy.ndarray")
        if frameBGR.ndim != 3:
            sys.exit("error: frameBGR mask be a matrix of three dimensions")
        color = (255, 0, 0)
        if color_string == "red":
            color = (0, 0, 255)
        if color_string == "green":
            color = (0, 255, 0)
        return cv.rectangle(frameBGR, upperLeft, bottomRight, color, 1)

    # return the frame applying a mask to isolate the color of the ball
    @staticmethod
    def get_hsvmask_on_ball(frame_hsv):
        if not isinstance(frame_hsv, np.ndarray):
            sys.exit("error: frame_hsv must be of type numpy.ndarray")
        if frame_hsv.ndim != 3:
            sys.exit("error: frame_hsv should be a matrix of three dimensions")

        lower_red = np.array([160, 75, 85])
        upper_red = np.array([180, 255, 255])
        #lower_red = np.array([ 4, 53, 38])
        #upper_red = np.array([ 350, 54, 40])

        mask = cv.inRange(frame_hsv, lower_red, upper_red)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,(5, 5), iterations=1)
        mask = cv.dilate(mask, None, iterations=2)


        res = cv.bitwise_and(frame_hsv, frame_hsv, mask=mask )
        return res

    # apply the knn method on the left frame
    @staticmethod
    def get_knn_on_left_frame(frame):
        if not isinstance(frame, np.ndarray):
            sys.exit("error: frame must be of type numpy.ndarray")
        if frame.ndim != 3:
            sys.exit("error: frame mask be a matrix of three dimensions")
        frame_knn = StandardVideoOperations.KNN_SX.apply(frame)
        return frame_knn

    # apply the knn method on the left frame
    @staticmethod
    def get_knn_on_right_frame(frame):
        if not isinstance(frame, np.ndarray):
            sys.exit("error: frame must be of type numpy.ndarray")
        if frame.ndim != 3:
            sys.exit("error: frame mask be a matrix of three dimensions")
        frame_knn = StandardVideoOperations.KNN_DX.apply(frame)
        #history= StandardVideoOperations.KNN_DX.getHistory()
        #print("history",history)
        return frame_knn

    # find the circles from counturns 
    @staticmethod
    def find_circles(frame_to_scan, frame_to_design):
        if not isinstance(frame_to_scan, np.ndarray):
            sys.exit("error: frame_to_scan must be of type numpy.ndarray")
        if frame_to_scan.ndim != 3:
            sys.exit("error: frame_to_scan mask be a matrix of three dimensions")
        if not isinstance(frame_to_design, np.ndarray):
            sys.exit("error: frame_to_design must be of type numpy.ndarray")
        if frame_to_design.ndim != 3:
            sys.exit("error: frame_to_design mask be a matrix of three dimensions")
        cnts = cv.findContours(frame_to_scan.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            for i in cnts:
                ((x, y), radius) = cv.minEnclosingCircle(i)
                if 10 < radius < 25:
                    cv.circle(frame_to_design, (int(x), int(y)), int(radius), (255, 255, 255), -1)
        return frame_to_design

    # return true if the ball is spotted inside the specified rows and column range

    @staticmethod
    def countWhitePixels(rows, colRange, greyScaleFrame):
        if not all(isinstance(row, int) for row in rows) or not all(isinstance(col, int) for col in colRange):
            sys.exit("error: rows and colRange must contain only integers")
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        for row in rows:
            consecutiveWhitePixels = 0
            consecutiveBlackPixels = 0
            for col in colRange:
                if greyScaleFrame[row, col] < 255:
                    consecutiveBlackPixels += 1
                    if consecutiveBlackPixels == 2:
                        consecutiveWhitePixels = 0
                else:
                    consecutiveBlackPixels = 0
                    consecutiveWhitePixels += 1
                    if 15 < consecutiveWhitePixels:
                        return True
        return False

    # return true if the ball is spotted above the basket in the right frame
    @staticmethod
    def spotBallOnTop_right(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [50, 55, 60]
        return StandardVideoOperations.countWhitePixels(rows, range(90, 150), greyScaleFrame)

    # return true if the ball is spotted in the middle of the basket in the right frame
    @staticmethod
    def spotBallOnMedium_right(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [100, 105, 110]
        return StandardVideoOperations.countWhitePixels(rows, range(90, 150), greyScaleFrame)

    # return true if the ball is spotted below the basket in the right frame
    @staticmethod
    def spotBallOnBottom_right(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [160, 165, 170]
        return StandardVideoOperations.countWhitePixels(rows, range(75, 175), greyScaleFrame)

    # return true if the ball is spotted above the basket in the left frame
    @staticmethod
    def spotBallOnTop_left(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [85, 90, 95]
        return StandardVideoOperations.countWhitePixels(rows, range(80, 140), greyScaleFrame)

    # return true if the ball is spotted in the middle of the basket in the left frame
    @staticmethod
    def spotBallOnMedium_left(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [125, 130, 135]
        return StandardVideoOperations.countWhitePixels(rows, range(80, 140), greyScaleFrame)

    # return true if the ball is spotted below the basket in the left frame
    @staticmethod
    def spotBallOnBottom_left(greyScaleFrame):
        if greyScaleFrame.ndim != 2:
            sys.exit("error: greyScaleFrame must be a matrix of two dimensions")
        rows = [160, 165, 170]
        return StandardVideoOperations.countWhitePixels(rows, range(70, 160), greyScaleFrame)

