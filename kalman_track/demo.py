import cv2
import numpy as np
from PIL import Image


# 空函数 用于填补cv2.createTrackbar的参数
def empty(a):
    pass


# 图片拼接函数
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# 根据每个轮廓点的坐标值获取链码
def get_chainCodes(contour):
    new_cnt = np.reshape(contour, (contour.shape[0], contour.shape[2]))
    res = ' '
    for index in range(0, new_cnt.shape[0] - 1):
        # 相邻轮廓点坐标值作差来判断移动方向
        dx = new_cnt[index + 1][0] - new_cnt[index][0]
        dy = new_cnt[index + 1][1] - new_cnt[index][1]
        if dy == 0 and dx > 0:
            res = res + str(0) + ' '  # 向右
        elif dy > 0 and dx > 0:
            res = res + str(1) + ' '  # 向右上
        elif dy > 0 and dx == 0:
            res = res + str(2) + ' '  # 向上
        elif dy > 0 and dx < 0:
            res = res + str(3) + ' '  # 向左上
        elif dy == 0 and dx < 0:
            res = res + str(4) + ' '  # 向左
        elif dy < 0 and dx < 0:
            res = res + str(5) + ' '  # 向左下
        elif dy < 0 and dx == 0:
            res = res + str(6) + ' '  # 向下
        elif dy < 0 and dx > 0:
            res = res + str(7) + ' '  # 向右下
    return res


def initWindow(video_path):
    # 初始化操作窗口名和大小
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 240)
    # 初始化变量的值域
    cv2.createTrackbar("Threshold1", "Parameters", 0, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 0, 255, empty)
    cv2.createTrackbar("AreaMin", "Parameters", 0, 30000, empty)
    cv2.createTrackbar("AreaMax", "Parameters", 0, 30000, empty)
    # 初始化变量的默认值
    cv2.setTrackbarPos("Threshold1", "Parameters", 50)
    cv2.setTrackbarPos("Threshold2", "Parameters", 75)
    cv2.setTrackbarPos("AreaMin", "Parameters", 1000)
    cv2.setTrackbarPos("AreaMax", "Parameters", 8000)
    # 初始化视频帧的大小
    frameWidth = 640
    frameHeight = 480
    v = cv2.VideoCapture(video_path)
    v.set(3, frameWidth)
    v.set(4, frameHeight)
    return v


def initKalman():
    # 初始化卡尔曼滤波器
    kf = cv2.KalmanFilter(6, 4, 0)
    # 初始化状态转移矩阵
    cv2.setIdentity(kf.transitionMatrix)
    # 初始化观测矩阵
    kf.measurementMatrix = np.zeros((4, 6), np.float32)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 4] = 1.0
    kf.measurementMatrix[3, 5] = 1.0
    # 初始化过程噪声协方差矩阵
    cv2.setIdentity(kf.processNoiseCov)
    kf.processNoiseCov[0, 0] = 1e-2
    kf.processNoiseCov[1, 1] = 1e-2
    kf.processNoiseCov[2, 2] = 5.0
    kf.processNoiseCov[3, 3] = 5.0
    kf.processNoiseCov[4, 4] = 1e-2
    kf.processNoiseCov[5, 5] = 1e-2
    # 测量噪声
    cv2.setIdentity(kf.measurementNoiseCov)
    return kf


# 进行轮廓检测 链码输出 目标追踪 轨迹绘制的函数 karf 卡尔曼滤波器 dt时间差 found 是否为首次检测 loc_list卡尔曼滤波器预测的物体中心位置
# center_list实际物体的位置 img经过预处理的图像帧 imgContour原图像帧
def detect(karf, dt, found, loc_list, center_list, img, imgContour):
    # 获取轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 对轮廓进行筛选
    for cnt in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("AreaMin", "Parameters")
        areaMax = cv2.getTrackbarPos("AreaMax", "Parameters")
        # 对多个封闭轮廓进行筛选
        if areaMin <= area <= areaMax:
            # 输出链码
            print('链码:', get_chainCodes(cnt))
            # 计算轮廓的长度
            peri = cv2.arcLength(cnt, True)
            # 求取轮廓的近似
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # 画轮廓
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            # 求取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(approx)
            # 加入物体中心点列表
            center_list.append([x, y, w, h])
    # 不是首次开始查找
    state = np.zeros(6, np.float32)  # [x,y,v_x,v_y,w,h],簇心位置，速度，高宽
    pre = np.zeros(4, np.float32)  # [z_x,z_y,z_w,z_h]
    if found:
        karf.transitionMatrix[0, 2] = dt
        karf.transitionMatrix[1, 3] = dt
        # 卡尔曼滤波器进行状态预测
        state = karf.predict()
        # 根据预测状态计算中心点和目标框的位置
        width = state[4]
        height = state[5]
        x_left = state[0] - width  # 左上角横坐标
        y_left = state[1] - height  # 左上角纵坐标
        x_right = state[0] + width  # 右下角横坐标
        y_right = state[1] + height  # 右下角纵坐标
        center_x = state[0]
        center_y = state[1]
        loc_list.append((int(center_x), int(center_y)))
        # 画边界框
        cv2.rectangle(imgContour, (x_left, y_left), (x_right, y_right), (0, 255, 0), 3)
    # 画运动轨迹
    for index in range(0, len(loc_list)):
        if loc_list[index - 1] is None or loc_list[index] is None:
            continue
        # 利用两点成线原理画直线
        cv2.line(imgContour, loc_list[index - 1], loc_list[index], (0, 255, 0), 1, cv2.LINE_AA)
    # 获取当前帧的实际状态值
    pre[0] = center_list[-1][0] + center_list[-1][2] / 2
    pre[1] = center_list[-1][1] + center_list[-1][3] / 2
    pre[2] = float(center_list[-1][2])
    pre[3] = float(center_list[-1][3])
    # 如果是首次检测，则利用第一帧的实际状态值来初始化卡尔曼滤波器的状态信息
    if not found:
        for i in range(len(karf.errorCovPre)):
            karf.errorCovPre[i, i] = 1
        state[0] = pre[0]
        state[1] = pre[1]
        state[2] = 0
        state[3] = 0
        state[4] = pre[2]
        state[5] = pre[3]
        karf.statePost = state
    else:
        # 如果不是首次预测，则利用当前状态值进行状态信息的修正
        karf.correct(pre)


if __name__ == '__main__':
    gif_begin = 20
    gif_end = 120
    gif_index = 0
    gif_frames = []
    # 初始化视频对象
    cap = initWindow(video_path='./video/demo_video.mp4')
    kalman_filter = initKalman()
    # 开始读取视频
    Found = False
    ticks = 0
    prePointCen = []  # 存储小球中心点位置
    centerList = []
    while cap.isOpened():
        # 获取每一帧的图像
        success, frame = cap.read()
        # 获取成功显示图像
        if success:
            # 修改图像大小
            frame = cv2.resize(frame, (512, 512))
            imgContour = frame.copy()
            # 进行高斯滤波
            imgBlur = cv2.GaussianBlur(frame, (7, 7), 1, 1)
            # 进行灰度转换
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
            # 利用Canny算子进行边缘检测
            imgCanny = cv2.Canny(imgGray, threshold1=threshold1, threshold2=threshold2)
            # 图片膨胀化处理
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            # 计算相邻两帧的时间差
            preTick = ticks
            ticks = float(cv2.getTickCount())
            dT = float((ticks - preTick) / cv2.getTickFrequency())
            # 进行轮廓检测 链码输出 目标追踪 轨迹绘制的函数
            detect(kalman_filter, dT, Found, prePointCen, centerList, imgDil, imgContour)
            Found = True
            # 显示实时视频
            imgStack = stackImages(0.8, ([frame, imgCanny],
                                         [imgDil, imgContour]))
            cv2.imshow("Result", imgStack)
            # 当视频播放完毕或输入空格后停止视频流的读取
            # 制作GIF
            if gif_begin <= gif_index <= gif_end:
                gif_frames.append(Image.fromarray(np.uint8(imgStack)))
            gif_index = gif_index + 1
            if cv2.waitKey(50) & 0xFF == ord(' '):
                break
        else:
            break
    gif_frames[0].save('./demo.gif', save_all=True, append_images=gif_frames[1:], transparency=0, duration=10, loop=0,disposal=2)
    # 释放视频对象
    cap.release()
    cv2.destroyAllWindows()
