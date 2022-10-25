from sys import argv
import cv2
import numpy as np


# img = cv2.imread('/data/DragonFly/data/taxi/im_day/A350-59-Flight-367_FLIRC_2020-01-28-22-20-22_0_153.41690043530434.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m07s819.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m21s500.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m29s277.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m36s460.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m40s799.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h10m54s856.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h09m57s654.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/vlcsnap-2022-09-12-11h11m27s045.png')
# img = cv2.imread('/home/host_folders/Data/TaxiSlideData/TAXI_SNAP_11.png')

img = cv2.imread(argv[1])

scale = 0.25
img = cv2.resize(img, (0,0), fx=scale, fy=scale)

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

# Creating a window for later use
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1000, 1000)
cv2.namedWindow('result')

#global variable
alpha =  88
beta =  0
gamma =  2
focal =  611
dist =  518
Tx =  0
Ty =  48

def updateImage(v):
    # get info from track bar and appy to result
    alpha = cv2.getTrackbarPos('alpha','result')
    beta = cv2.getTrackbarPos('beta', 'result')
    gamma = cv2.getTrackbarPos('gamma','result')

    focal = cv2.getTrackbarPos('focal','result')
    dist = cv2.getTrackbarPos('dist', 'result')
    Tx = cv2.getTrackbarPos('Tx', 'result')
    Ty = cv2.getTrackbarPos('Ty', 'result')

    hsv = hsv_origin.copy()
    bird_col = bird_view(img, -alpha, beta, -gamma,focal, dist,Tx,Ty)

    h, w, _ = bird_col.shape
    nb_lines = 10
    split_width = w / 10
    cv2.line(bird_col, (int(w/2),h), (int(w/2),0), (0,255,0))
    for i in range(nb_lines):
        cv2.line(bird_col, (int(split_width*i),h), (int(split_width*i),0), (0,255,0))

    cv2.imshow('result',bird_col)

#create trackbars for high,low H,S,V
cv2.createTrackbar('alpha','result',alpha,100, updateImage)
cv2.createTrackbar('beta','result',beta,10, updateImage)
cv2.createTrackbar('gamma','result',gamma,10, updateImage)

cv2.createTrackbar('focal','result',focal,3000, updateImage)
cv2.createTrackbar('dist','result',dist,2000, updateImage)

cv2.createTrackbar('Tx','result',Tx,1000, updateImage)
cv2.createTrackbar('Ty','result',Ty,1000, updateImage)

#converting to HSV
hsv_origin = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)



while(1):

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print('alpha = ', alpha)
        print('beta = ', beta)
        print('gamma = ', gamma)
        print('focal = ', focal)
        print('dist = ', dist)
        print('Tx = ', Tx)
        print('Ty = ', Ty)
        break

# cap.release()
cv2.destroyAllWindows()