import cv2
import numpy as np
from scipy import signal

def medianBlur(image_way,winSize):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #图像的高、宽
    rows,cols = image.shape
    #窗口的高、宽，均为奇数
    winH,winW = winSize
    halfWinH = (winH-1)//2
    halfWinW = (winW-1)//2
    #中值滤波后的输出图像
    medianBlurImage = np.zeros(image.shape,image.dtype)
    for r in range(rows):
        for c in range(cols):
            #判断边界
            rTop = 0 if r-halfWinH < 0 else r-halfWinH
            rBottom = rows-1 if r+halfWinH > rows-1 else r+halfWinH
            cLeft = 0 if c-halfWinW < 0 else c-halfWinW
            cRight = cols-1 if c+halfWinW > cols-1 else c+halfWinW
            #取邻域
            region = image[rTop:rBottom+1,cLeft:cRight+1]
            #取中值
            medianBlurImage[r][c] = np.median(region)
    cv2.imshow('medianBlurImage',medianBlurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return medianBlurImage

def gaussBlur(image,sigma,H,W,_boundary = 'fill',_fillvalue = 0):
    image=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    #构建水平方向上的高斯卷积核
    gaussKenrnel_x=cv2.getGaussianKernel(sigma,W,cv2.CV_64F)
    #转置
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    #图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image,gaussKenrnel_x,
                                    mode='same',boundary = _boundary,fillvalue = _fillvalue)
    #构建垂直方向上的高斯卷积核
    gaussKenrnel_y=cv2.getGaussianKernel(sigma,H,cv2.CV_64F)
    #与垂直方向上的高斯核卷积
    gaussBlur_xy=signal.convolve2d(gaussBlur_x,gaussKenrnel_y,mode='same',
                                   boundary = _boundary,fillvalue = _fillvalue)
    blurImage=np.round(gaussBlur_xy)
    blurImage=blurImage.astype(np.uint8)
    cv2.imshow('GaussBlur',blurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blurImage

def fastMeanBlur(image,winSize,borderType = cv2.BORDER_DEFAULT):
    image=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    halfH = int((winSize[0]-1)/2)
    halfW = int((winSize[1]-1)/2)
    #边界扩充
    paddImage = cv2.copyMakeBorder(image,halfH,halfH,halfW,halfW,borderType)
    rows,cols=paddImage.shape
    #行积分运算
    inteImageC=np.zeros((rows,cols),np.float32)
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                inteImageC[r][c] = paddImage[r][c]
            else:
                inteImageC[r][c] = inteImageC[r][c-1]+paddImage[r][c]
    #列积分运算
    inteImage=np.zeros(paddImage.shape,np.float32)
    for c in range(cols):
        for r in range(rows):
            if r == 0:
                inteImage[r][c] = inteImageC[r][c]
            else:
                inteImage[r][c] = inteImage[r-1][c]+inteImageC[r][c]
    #上边和左边进行补零
    inteImage_0 = np.zeros((rows+1,cols+1),np.float32)
    inteImage_0[1:rows+1,1:cols+1] = inteImage
    ratio = 1.0/(winSize[0]*winSize[1])
    #图像积分
    paddIntegral = inteImage_0
    #原图像的高、宽
    rows_a,cols_a=image.shape
    #均值滤波后的结果
    meanBlurImage = np.zeros(image.shape,np.float32)
    r,c=0,0
    for h in range(halfH,halfH+rows_a,1):
        for w in range(halfW,halfW+cols_a,1):
            meanBlurImage[r][c] = (paddIntegral[h+halfH+1][w+halfW+1]+
                         paddIntegral[h-halfH][w-halfW]-paddIntegral[h+halfH+1][w-halfW]-
                         paddIntegral[h-halfH][w+halfW+1])*ratio
            c+=1
        r+=1
        c=0
    meanBlurImage = meanBlurImage.astype(np.uint8)
    cv2.imshow('fastMeanBlur',meanBlurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return meanBlurImage

