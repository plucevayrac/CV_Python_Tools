from sys import argv
import cv2
import numpy as np

# ************* Functions *************


def bird_view(img_source, alpha=-88,
                     beta=0,
                     gamma=1,
                     focal=642, #784
                     dist=518, #534
                     Tx=0,
                     Ty=64, #59
                     mX=0,
                     mY=0):
    img = np.copy(img_source)
    (h, w) = img.shape[0:2]
    if mX==0:
        mX = w / 2
    if mY==0:
        mY = h / 2
    # Camera angles
    pitch = float(alpha) * np.pi / 180
    roll = float(beta) * np.pi / 180
    yaw = float(gamma) * np.pi / 180

    # Projection 2D-> 3D
    A1 = np.array([[1, 0, -mX], [0, 1, -mY], [0, 0, 0], [0, 0, 1]])

    def cos(angle):
        return np.cos(angle)

    def sin(angle):
        return np.sin(angle)

    # Rotation matrices
    RX = np.array([[1, 0, 0, 0], [0, cos(pitch), -sin(pitch), 0], [0, sin(pitch), cos(pitch), 0], [0, 0, 0, 1]])
    RY = np.array([[cos(roll), 0, -sin(roll), 0], [0, 1, 0, 0], [sin(roll), 0, cos(roll), 0], [0, 0, 0, 1]])
    RZ = np.array([[cos(yaw), -sin(yaw), 0, 0], [sin(yaw), cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Full rotation matrix
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix on Z axis, changing dist param will change height
    T = np.array([[1, 0, 0, Tx], [0, 1, 0, Ty], [0, 0, 1, dist], [0, 0, 0, 1]])

    # Projection 2D-> 3D
    A2 = np.array([[focal, 0, mX, 0], [0, focal, mY, 0], [0, 0, 1, 0]])

    # transformation matrix
    M = np.dot(A2, (np.dot(T, np.dot(R, A1))))

    # # start_time = time.time()
    warped_img = cv2.warpPerspective(img, M, (w, h),
                                     flags=cv2.INTER_LINEAR+ cv2.WARP_INVERSE_MAP)  # Image warping ~.001 s
    # # print("--- %s seconds ---" % (time.time() - start_time))
    return warped_img

def applyCLAHE(img, clipLimit, tileGridSize):
    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    # result = np.hstack((img, enhanced_img))
    return enhanced_img

def applyHSVFilter(img, low_H, low_S, low_V, high_H, high_S, high_V):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_filtered = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    return hsv_filtered


def updateImage(v):
    # get info from track bar and appy to result
    bird_perc = cv2.getTrackbarPos('bird_perc','result') / 100
    clipLimit = cv2.getTrackbarPos('clipLimit','result')
    tileH = cv2.getTrackbarPos('tileH','result')
    tileW = cv2.getTrackbarPos('tileW', 'result')
    tileGridSize = (tileW, tileH)
    vgridLines = cv2.getTrackbarPos('vgridLines', 'result')
    hgridLines = cv2.getTrackbarPos('hgridLines', 'result')
    blur = cv2.getTrackbarPos('blur', 'result')
    edgeDetect = cv2.getTrackbarPos('edgeDetect', 'result')
    ed_treshLow = cv2.getTrackbarPos('ed_treshLow', 'result')
    ed_treshHigh = cv2.getTrackbarPos('ed_treshHigh', 'result')
    video_frame = cv2.getTrackbarPos('video_frame', 'result')
    low_H = cv2.getTrackbarPos('low_H', 'result')
    low_S = cv2.getTrackbarPos('low_S', 'result')
    low_V = cv2.getTrackbarPos('low_V', 'result')
    high_H = cv2.getTrackbarPos('high_H', 'result')
    high_V  = cv2.getTrackbarPos('high_S', 'result')
    high_S  = cv2.getTrackbarPos('high_V', 'result')


    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ret, img = cap.read()
    if not ret:
        raise ValueError("Couldn't read frame: {}".format(video_frame))
    img = cv2.resize(img, (1024,750))

    # hsv = hsv_origin.copy()
    # ** BLUR
    if blur > 0:
        img_test = cv2.blur(img,(blur,blur))
    else:
        img_test = img

    # ** BIRDVIEW
    if bird_perc > 0:
        img_test = bird_view(img_test, -alpha*bird_perc, beta*bird_perc, -gamma*bird_perc,focal*bird_perc, dist*bird_perc,Tx*bird_perc,Ty*bird_perc)

    # ** CLAHE
    if (clipLimit > 0) & (tileH> 0) & (tileW > 0):
        img_test = applyCLAHE(img_test, clipLimit, tileGridSize)

    # edgeDetect
    if edgeDetect == 1:
        img_test = cv2.Canny(image=img_test, threshold1=ed_treshLow, threshold2=ed_treshHigh) 
    
    # HSV
    img_test = applyHSVFilter(img_test, low_H, low_S, low_V, high_H, high_S, high_V)

    # ** ADDING GRID LINES
    if vgridLines > 0:
        h, w, _ = img_test.shape
        split_width = w / (vgridLines+1)
        cv2.line(img_test, (int(w/2),h), (int(w/2),0), (0,255,0))
        for i in range(vgridLines):
            cv2.line(img_test, (int(split_width*i),h), (int(split_width*i),0), (0,255,0))
    if hgridLines > 0:
        h, w, _ = img_test.shape
        split_height = h / (hgridLines+1)
        cv2.line(img_test, (0,int(h/2)), (w,int(h/2)), (0,255,0))
        for i in range(1, hgridLines+1):
            cv2.line(img_test, (0,int(split_height*i)), (w,int(split_height*i)), (0,255,0))

    cv2.imshow('result',img_test)
    cv2.imshow('source',img)

def getImageAt(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    return cap.read()

#**************** MAIN ****************

# #converting to HSV
# hsv_origin = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

# Creating a window for later use
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1000, 1000)
cv2.namedWindow('source', cv2.WINDOW_NORMAL)
cv2.resizeWindow('source', 1000, 1000)

video_path = argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Can't open video: {}".format(video_path))


ret, img = cap.read()


img = cv2.resize(img, (1024,750))

#global variable
bird_perc = 0
clipLimit = 2 #0
tileH = 8
tileW = 8
vgridLines = 0
hgridLines = 0
blur = 0
edgeDetect = 0
ed_treshLow = 100 #183
ed_treshHigh = 200 #50
video_frame = 0
low_H = 15
low_S = 30
low_V = 30
high_H = 30
high_V = 255
high_S = 255

video_frame_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(video_frame_max)

alpha =  88
beta =  0
gamma =  2
focal =  613
dist =  571
Tx =  0
Ty =  100

#create trackbars for high,low H,S,V
cv2.createTrackbar('bird_perc', 'result', bird_perc, 100, updateImage)
cv2.createTrackbar('clipLimit', 'result', clipLimit, 100, updateImage)
cv2.createTrackbar('tileH', 'result', tileH, 100, updateImage)
cv2.createTrackbar('tileW', 'result', tileW, 100, updateImage)
cv2.createTrackbar('vgridLines', 'result', vgridLines, 100, updateImage)
cv2.createTrackbar('hgridLines', 'result', hgridLines, 100, updateImage)
cv2.createTrackbar('blur', 'result', blur, 100, updateImage)
cv2.createTrackbar('edgeDetect', 'result', edgeDetect, 1, updateImage)
cv2.createTrackbar('ed_treshLow', 'result', ed_treshLow, 1000, updateImage)
cv2.createTrackbar('ed_treshHigh', 'result', ed_treshHigh, 1000, updateImage)
cv2.createTrackbar("low_H", 'result' , low_H, 255, updateImage)
cv2.createTrackbar("high_H", 'result' , high_H, 255, updateImage)
cv2.createTrackbar("low_S", 'result' , low_S, 255, updateImage)
cv2.createTrackbar("high_S", 'result' , high_S, 255, updateImage)
cv2.createTrackbar("low_V", 'result' , low_V, 255, updateImage)
cv2.createTrackbar("high_V", 'result' , high_V, 255, updateImage)
cv2.createTrackbar('video_frame', 'result', video_frame, video_frame_max, updateImage)

while(1):

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print('bird_perc = ', bird_perc)
        print('clipLimit = ', clipLimit)
        print('tileH = ', tileH)
        print('tileW = ', tileW)
        print('vgridLines = ', vgridLines)
        print('hgridLines = ', hgridLines)
        print('blur = ', blur)
        print('edgeDetect = ', edgeDetect)
        print('ed_treshLow = ', ed_treshLow)
        print('ed_treshHigh = ', ed_treshHigh)
        break

# cap.release()
cv2.destroyAllWindows()