import cv2
# import serial
import numpy as np
# from struct import pack, unpack

cap = cv2.VideoCapture(0)


class Color:
    r = 0
    g = 0
    b = 0

    def __init__(self, r, g, b):
        self.r, self.g, self.b = r, g, b


def BGRtoRGB(color):
    return Color(color[2], color[1], color[0])


def RGBtoBGR(color):
    assert isinstance(color, Color)
    return [color.b, color.g, color.r]


def onC(a):
    pass


def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


cv2.namedWindow('captureFull')

cv2.createTrackbar('R1', 'captureFull', 0, 255, onC)
cv2.createTrackbar('G1', 'captureFull', 0, 255, onC)
cv2.createTrackbar('B1', 'captureFull', 0, 255, onC)
cv2.createTrackbar('R2', 'captureFull', 0, 255, onC)
cv2.createTrackbar('G2', 'captureFull', 0, 255, onC)
cv2.createTrackbar('B2', 'captureFull', 0, 255, onC)

cv2.setTrackbarPos('R1', 'captureFull', 60)
cv2.setTrackbarPos('G1', 'captureFull', 105)
cv2.setTrackbarPos('B1', 'captureFull', 112)
cv2.setTrackbarPos('R2', 'captureFull', 255)
cv2.setTrackbarPos('G2', 'captureFull', 255)
cv2.setTrackbarPos('B2', 'captureFull', 115)

# 初始化窗口通信
# ser = serial.Serial(port="/dev/ttyTHS2", baudrate=100000, stopbits=STOPBITS_ONE, bytesize=EIGHTBITS, parity=PARITY_EVEN)
# ser.open()

while (1):
    # get a frame
    ret, frame = cap.read()

    # show a frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    height, width = frame.shape[0], frame.shape[1]

    tgtColor = Color(0, 50, 220)

    outputFrame = frame

    r1 = cv2.getTrackbarPos('R1', 'captureFull')
    g1 = cv2.getTrackbarPos('G1', 'captureFull')
    b1 = cv2.getTrackbarPos('B1', 'captureFull')
    r2 = cv2.getTrackbarPos('R2', 'captureFull')
    g2 = cv2.getTrackbarPos('G2', 'captureFull')
    b2 = cv2.getTrackbarPos('B2', 'captureFull')

    HSV = cv2.cvtColor(outputFrame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    LowerBlue = np.array([b1, g1, r1])
    UpperBlue = np.array([b2, g2, r2])
    mask = cv2.inRange(HSV, LowerBlue, UpperBlue)
    BlueThings = cv2.bitwise_and(outputFrame, outputFrame, mask=mask)
    BlueThings = cv2.GaussianBlur(BlueThings, (3, 3), 8)

    gray = cv2.cvtColor(BlueThings, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    canny = cv2.Canny(gray, 30, 150)

    canny = np.uint8(np.absolute(canny))

    meanX, meanY = [], []
    thres2 = 30

    for y in range(height):
        xCount = 0
        xTime = 0
        for x in range(width):
            if gray[y, x] > thres2:
                xCount += x
                xTime += 1

        if not xTime == 0:
            meanX.append(int(xCount / xTime))
        else:
            meanX.append(0)

    for x in range(width):
        yCount = 0
        yTime = 0
        for y in range(height):
            if gray[y, x] > thres2:
                yCount += y
                yTime += 1
        if not yTime == 0:
            meanY.append(int(yCount / yTime))
        else:
            meanY.append(0)

    meanPointX = 0
    xC = 0
    for num in meanX:
        if not num == 0:
            meanPointX += num
            xC += 1
    if not xC == 0:
        meanPointX = int(meanPointX / xC)
    else:
        meanPointX = 0

    meanPointY = 0
    yC = 0
    for num in meanY:
        if not num == 0:
            meanPointY += num
            yC += 1
    if not yC == 0:
        meanPointY = int(meanPointY / yC)
    else:
        meanPointY = 0

    cv2.rectangle(frame, (meanPointX - 10, meanPointY - 10), (meanPointX + 10, meanPointY + 10), (0, 0, 255), 3)

    cv2.imshow("captureFull", frame)
    cv2.imshow("capture", canny)
    cv2.imshow("capture3", gray)

    # 计算偏差角度
    offestX = ((meanPointX / (width / 2)) - 1) * 150
    offestY = ((meanPointX / (height / 2)) - 1) * 80

    print("offest X: ", offestX)
    print("offest Y: ", offestY)

    # 发送数据到串口
    # ser.write(pack('f', offestX) + pack('f', offestY))

cap.release()
cv2.destroyAllWindows()
