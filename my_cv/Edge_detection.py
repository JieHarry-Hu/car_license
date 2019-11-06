# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
from scipy import signal

# roberts 边缘检测
def roberts(image_way,threshold=255,_boundary='symm',_fillvalue=0):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #图像的高、宽
    H1,W1=image.shape[0:2]
    #卷积核的尺寸
    H2,W2=2,2
    #卷积核1及锚点的位置
    R1=np.array([[1,0],[0,-1]],np.float32)
    kr1,kc1=0,0
    #计算full卷积
    IconR1=signal.convolve2d(image,R1,mode='full',boundary=_boundary,fillvalue=_fillvalue)
    IconR1=IconR1[H2-kr1-1:H1+H2-kr1-1,W2-kc1-1:W1+W2-kc1-1]
    #卷积核2
    R2=np.array([[0,1],[- 1,0]],np.float32)
    #先计算full卷积
    IconR2=signal.convolve2d(image,R2,mode='full',boundary=_boundary,fillvalue=_fillvalue)
    #锚点的位置
    kr2,kc2=0,1
    #根据锚点的位置截取full卷积，从而得到same卷积
    IconR2=IconR2[H2-kr2-1:H1+H2-kr2-1,W2-kc2-1:W1+W2-kc2-1]
    #45°方向上的边缘强度的灰度级显示
    IconR1=np.abs(IconR1)
    #135°方向上的边缘强度
    IconR2=np.abs(IconR2)
    #用平方和的开方来衡量最后输出的边缘
    edge=np.sqrt(np.power(IconR1,2) + np.power(IconR2,2))
    edge = np.round(edge)
    edge[edge>threshold]=0
    edge.astype(np.uint8)
    #显示边缘
    cv2.imshow('edge',edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

# prewitt 边缘检测
def prewitt(image_way,_boundary='symm'):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #prewitt_x是可分离的，根据卷积运算的结合律，分两次小卷积核运算
    #1：垂直方向上的均值平滑
    ones_y = np.array([[1],[1],[1]],np.float32)
    i_conv_p_x = signal.convolve2d(image,ones_y,mode='same',boundary=_boundary)
    #2：水平方向上的差分
    diff_x = np.array([[1,0,-1]],np.float32)
    i_conv_p_x = signal.convolve2d(i_conv_p_x,diff_x,mode='same',boundary=_boundary)
    #prewitt_y是可分离的，根据卷积运算的结合律，分两次小卷积核运算
    #1：水平方向上的均值平滑
    ones_x = np.array([[1,1,1]],np.float32)
    i_conv_p_y = signal.convolve2d(image,ones_x,mode='same',boundary=_boundary)
    #2：垂直方向上的差分
    diff_y = np.array([[1],[0],[-1]],np.float32)
    i_conv_p_y = signal.convolve2d(i_conv_p_y,diff_y,mode='same',boundary=_boundary)
    #取绝对值，分别得到水平方向和垂直方向上的边缘强度
    abs_i_conv_pre_x = np.abs(i_conv_p_x)
    abs_i_conv_pre_y = np.abs(i_conv_p_y)
    #水平方向和垂直方向上的边缘强度的灰度级显示
    edge_x = abs_i_conv_pre_x.copy()
    edge_y = abs_i_conv_pre_y.copy()
    #将大于255的值截为255
    edge_x[edge_x>255] = 255
    edge_y[edge_y>255] = 255
    #利用abs_i_conv_pre_x和abs_i_conv_pre_y求最终的边缘强度
    #求边缘强度，有多种形式，这里使用的是插值法
    edge = 0.5*edge_x + 0.5*edge_y
    edge[edge>255] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('edge',edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

#二项式展开式的系数，即平滑系数
def pascalSmooth(n):
    pascalSmooth = np.zeros([1,n],np.float32)
    for i in range(n):
        pascalSmooth[0][i] = math.factorial(n -1)/(math.factorial(i)*math.factorial(n-1-i))
    return pascalSmooth
#计算差分
def pascalDiff(n):
    pascalDiff = np.zeros([1,n],np.float32)
    pascalSmooth_previous = pascalSmooth(n-1)
    for i in range(n):
        if i ==0:
            #恒等于 1
            pascalDiff[0][i] = pascalSmooth_previous[0][i]
        elif i == n-1:
            #恒等于 -1
            pascalDiff[0][i] = -pascalSmooth_previous[0][i-1]
        else:
            pascalDiff[0][i] = pascalSmooth_previous[0][i] - pascalSmooth_previous[0][i-1]
    return pascalDiff

# sobel 边缘检测
def sobel(image_way,winSize):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    rows,cols = image.shape
    pascalSmoothKernel = pascalSmooth(winSize)
    pascalDiffKernel = pascalDiff(winSize)
    # --- 与水平方向的卷积核卷积 ----
    image_sobel_x = np.zeros(image.shape,np.float32)
    #垂直方向上的平滑
    image_sobel_x = signal.convolve2d(image,pascalSmoothKernel.transpose(),mode='same')
    #水平方向上的差分
    image_sobel_x = signal.convolve2d(image_sobel_x,pascalDiffKernel,mode='same')
    # --- 与垂直方向上的卷积核卷积 --- 
    image_sobel_y = np.zeros(image.shape,np.float32)
    #水平方向上的平滑
    image_sobel_y = signal.convolve2d(image,pascalSmoothKernel,mode='same')
    #垂直方向上的差分
    image_sobel_y = signal.convolve2d(image_sobel_y,pascalDiffKernel.transpose(),mode='same')
    #卷积
    edge_x = np.abs(image_sobel_x)
    edge_x[ edge_x>255]=255
    edge_x=edge_x.astype(np.uint8)
    edge_y = np.abs(image_sobel_y)
    edge_y[ edge_y>255]=255
    edge_y=edge_y.astype(np.uint8)
    #边缘强度：两个卷积结果对应位置的平方和
    edge = np.sqrt(np.power(image_sobel_x,2.0) + np.power(image_sobel_y,2.0))
    #边缘强度的灰度级显示
    edge[edge>255] = 255
    edge = np.round(edge)
    edge = edge.astype(np.uint8)
    cv2.imshow('edge',edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

#非极大值抑制：插值比较
def non_maximum_suppression_Inter(dx,dy):
    #边缘强度
    edgeMag = np.sqrt(np.power(dx,2.0)+np.power(dy,2.0))
    #宽、高
    rows,cols = dx.shape
    #梯度方向
    gradientDirection = np.zeros(dx.shape)
    #边缘强度的非极大值抑制
    edgeMag_nonMaxSup = np.zeros(dx.shape)
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            if dy[r][c] ==0 and dx[r][c] == 0:
                continue
            #angle的范围 [0,180],[-180,0]
            angle = math.atan2(dy[r][c],dx[r][c])/math.pi*180
            gradientDirection[r][c] = angle
            #左上方和上方的插值 右下方和下方的插值
            if (angle > 45 and angle <=90) or (angle > -135 and angle <=-90):
                ratio = dx[r][c]/dy[r][c]
                leftTop_top = ratio*edgeMag[r-1][c-1]+(1-ratio)*edgeMag[r-1][c]
                rightBottom_bottom = (1-ratio)*edgeMag[r+1][c] + ratio*edgeMag[r+1][c+1]
                if edgeMag[r][c] >  leftTop_top and edgeMag[r][c] > rightBottom_bottom:
                    edgeMag_nonMaxSup[r][c]  = edgeMag[r][c]
            #右上方和上方的插值 左下方和下方的插值
            if (angle>90 and angle<=135) or (angle>-90 and angle <= -45):
                ratio = abs(dx[r][c]/dy[r][c])
                rightTop_top = ratio*edgeMag[r-1][c+1] + (1-ratio)*edgeMag[r-1][c]
                leftBottom_bottom = ratio*edgeMag[r+1][c-1] + (1-ratio)*edgeMag[r+1][c]
                if edgeMag[r][c] > rightTop_top and edgeMag[r][c] > leftBottom_bottom:
                    edgeMag_nonMaxSup[r][c]  = edgeMag[r][c]
            #左上方和左方的插值 右下方和右方的插值
            if (angle>=0 and angle <=45) or (angle>-180 and angle <= -135):
                ratio = dy[r][c]/dx[r][c]
                rightBottom_right = ratio*edgeMag[r+1][c+1]+(1-ratio)*edgeMag[r][c+1]
                leftTop_left = ratio*edgeMag[r-1][c-1]+(1-ratio)*edgeMag[r][c-1]
                if edgeMag[r][c] > rightBottom_right and edgeMag[r][c] > leftTop_left:
                    edgeMag_nonMaxSup[r][c]  = edgeMag[r][c]
            #右上方和右方的插值 左下方和左方的插值
            if(angle>135 and angle<=180) or (angle>-45 and angle <=0):
                ratio = abs(dy[r][c]/dx[r][c])
                rightTop_right = ratio*edgeMag[r-1][c+1]+(1-ratio)*edgeMag[r][c+1]
                leftBottom_left = ratio*edgeMag[r+1][c-1]+(1-ratio)*edgeMag[r][c-1]
                if edgeMag[r][c] > rightTop_right and edgeMag[r][c] > leftBottom_left:
                    edgeMag_nonMaxSup[r][c]  = edgeMag[r][c]
    return edgeMag_nonMaxSup
#判断一个点的坐标是否在图像范围内
def checkInRange(r,c,rows,cols):
    if r>=0 and r<rows and c>=0 and c<cols:
        return True
    else:
        return False
def trace(edgeMag_nonMaxSup,edge,lowerThresh,r,c,rows,cols):
    #大于阈值为确定边缘点
    if edge[r][c] == 0:
        edge[r][c]=255
        for i in range(-1,2):
            for j in range(-1,2):
                if checkInRange(r+i,c+j,rows,cols) and edgeMag_nonMaxSup[r+i][c+j] >= lowerThresh:
                    trace(edgeMag_nonMaxSup,edge,lowerThresh,r+i,c+j,rows,cols)
#滞后阈值
def hysteresisThreshold(edge_nonMaxSup,lowerThresh,upperThresh):
    #宽高
    rows,cols = edge_nonMaxSup.shape
    edge = np.zeros(edge_nonMaxSup.shape,np.uint8)
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            #大于高阈值，设置为确定边缘点，而且以该点为起始点延长边缘
            if edge_nonMaxSup[r][c] >= upperThresh:
                trace(edge_nonMaxSup,edge,lowerThresh,r,c,rows,cols)
            #小于低阈值，被剔除
            if edge_nonMaxSup[r][c]< lowerThresh:
                edge[r][c] = 0
    return edge

# canny 边缘检测
def canny(image_way,lowerThresh,upperThresh):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #第一步： 基于 sobel 核的卷积
    rows,cols = image.shape
    pascalSmoothKernel = pascalSmooth(3)
    pascalDiffKernel = pascalDiff(3)
    # --- 与水平方向的卷积核卷积 ----
    image_sobel_x = np.zeros(image.shape,np.float32)
    #垂直方向上的平滑
    image_sobel_x = signal.convolve2d(image,pascalSmoothKernel.transpose(),mode='same')
    #水平方向上的差分
    image_sobel_x = signal.convolve2d(image_sobel_x,pascalDiffKernel,mode='same')
    # --- 与垂直方向上的卷积核卷积 --- 
    image_sobel_y = np.zeros(image.shape,np.float32)
    #水平方向上的平滑
    image_sobel_y = signal.convolve2d(image,pascalSmoothKernel,mode='same')
    #垂直方向上的差分
    image_sobel_y = signal.convolve2d(image_sobel_y,pascalDiffKernel.transpose(),mode='same')
    #边缘强度：两个卷积结果对应位置的平方和
    edge = np.sqrt(np.power(image_sobel_x,2.0) + np.power(image_sobel_y,2.0))
    #边缘强度的灰度级显示
    edge[edge>255] = 255
    edge = edge.astype(np.uint8)
    #边缘强度：两个卷积结果对应位置的平方和
    edge = np.sqrt(np.power(image_sobel_x,2.0) + np.power(image_sobel_y,2.0))
    #边缘强度的灰度级显示
    edge[edge>255] = 255
    edge = edge.astype(np.uint8)
    #第二步：非极大值抑制
    edgeMag_nonMaxSup = non_maximum_suppression_Inter(image_sobel_x,image_sobel_y)
    edgeMag_nonMaxSup[edgeMag_nonMaxSup>255] =255
    edgeMag_nonMaxSup = edgeMag_nonMaxSup.astype(np.uint8)
    #第三步：双阈值滞后阈值处理，得到 canny 边缘
    #滞后阈值的目的就是最后决定处于高阈值和低阈值之间的是否为边缘点
    edge = hysteresisThreshold(edgeMag_nonMaxSup,lowerThresh,upperThresh)
    cv2.imshow("canny",edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

# 拉普拉斯laplacian边缘检测
def laplacian(image_way,_boundary='symm',_fillvalue=0):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #拉普拉斯卷积核
    #laplacianKernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float32)
    laplacianKernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32)
    #图像矩阵和拉普拉斯算子卷积
    i_conv_lap = signal.convolve2d(image,laplacianKernel,mode='same',
                                   boundary = _boundary,fillvalue=_fillvalue)

    #对卷积结果进行阈值化处理
    threshEdge = np.copy(i_conv_lap)
    threshEdge[threshEdge>0] = 255
    threshEdge[threshEdge<0] = 0
    threshEdge = threshEdge.astype(np.uint8)
    cv2.imshow('laplacian',threshEdge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return threshEdge

# 构建高斯拉普拉斯卷积核
def createLoGKernel(sigma,size):
    # LoG 算子的高和宽，且两者均为奇数
    H,W = size
    r,c = np.mgrid[0:H:1,0:W:1]
    r = r-(H-1)/2
    c = c-(W-1)/2
    #方差
    sigma2 = pow(sigma,2.0)
    # LoG 核
    norm2 = np.power(r,2.0)+np.power(c,2.0)
    #LoGKernel=1.0/sigma2*(norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))
    LoGKernel=(norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))
    return LoGKernel
# 高斯拉普拉斯LoG边缘检测
def LoG(image_way,sigma,size,_boundary='symm'):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #构建 LoG 卷积核
    loGKernel = createLoGKernel(sigma,size)
    #图像与 LoG 卷积核卷积
    img_conv_log = signal.convolve2d(image,loGKernel,'same',boundary =_boundary)
    #边缘的二值化显示
    edge_binary = np.copy(img_conv_log)
    edge_binary[edge_binary>0]=255
    edge_binary[edge_binary<=0]=0
    edge_binary = edge_binary.astype(np.uint8)
    cv2.imshow("LoG_edge",edge_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge_binary

# 实现非归一化的高斯卷积
def gaussConv(I,size,sigma):
    #卷积核的高和宽
    H,W = size
    #构造水平方向上非归一化的高斯卷积核
    xr,xc = np.mgrid[0:1,0:W]
    xc = xc-(W-1)/2
    xk = np.exp(-np.power(xc,2.0)/(2.0*pow(sigma,2)))
    # I 与 xk 卷积
    I_xk = signal.convolve2d(I,xk,'same','symm')
    #构造垂直方向上的非归一化的高斯卷积核
    yr,yc = np.mgrid[0:H,0:1]
    yr = yr-(H-1)/2
    yk = np.exp(-np.power(yr,2.0)/(2.0*pow(sigma,2.0)))
    # I_xk 与 yk 卷积
    I_xk_yk = signal.convolve2d(I_xk,yk,'same','symm')
    I_xk_yk *= 1.0/(2*np.pi*pow(sigma,2.0))
    return I_xk_yk
#  高斯差分DoG边缘检测
def DoG(image_way,sigma,size,k=1.1):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #标准差为 sigma 的非归一化的高斯卷积
    Is = gaussConv(image,size,sigma)
    #标准差为 k*sigma 的非归一化高斯卷积
    Isk = gaussConv(image,size,k*sigma)
    #两个高斯卷积的差分
    doG = Isk - Is
    doG /= (pow(sigma,2.0)*(k-1))
    imageDoG = doG
     #二值化边缘，对 imageDoG 阈值处理
    edge = np.copy(imageDoG)
    edge[edge>0] = 255
    edge[edge<=0] = 0
    edge = edge.astype(np.uint8)
    cv2.imshow("edge",edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

#零交叉点：方法1
def zero_cross_default(doG):
    zero_cross = np.zeros(doG.shape,np.uint8)
    rows,cols = doG.shape
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            # 左 / 右方向
            if doG[r][c-1]*doG[r][c+1] < 0:
                zero_cross[r][c] = 255
                continue
            #上 / 下方向
            if doG[r-1][c]*doG[r+1][c] < 0:
                zero_cross[r][c] = 255
                continue
            #左上 / 右下方向
            if doG[r-1][c-1]*doG[r+1][c+1] < 0:
                zero_cross[r][c] = 255
                continue
            #右上 / 左下方向
            if doG[r-1][c+1]*doG[r+1][c-1] < 0:
                zero_cross[r][c] = 255
                continue
    return zero_cross
#零交叉点：方法2
def zero_cross_mean(doG):
    zero_cross = np.zeros(doG.shape,np.uint8)
    #存储左上，右上，左下，右下方向
    fourMean = np.zeros(4,np.float32)
    rows,cols = doG.shape
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            #左上方的均值
            leftTopMean = np.mean(doG[r-1:r+1,c-1:c+1])
            fourMean[0] = leftTopMean
            #右上方的均值
            rightTopMean = np.mean(doG[r-1:r+1,c:c+2])
            fourMean[1] = rightTopMean
            #左下方的均值
            leftBottomMean = np.mean(doG[r:r+2,c-1:c+1])
            fourMean[2] = leftBottomMean
            #右下方的均值
            rightBottomMean = np.mean(doG[r:r+2,c:c+2])
            fourMean[3] = rightBottomMean
            if(np.min(fourMean)*np.max(fourMean)<0):
                zero_cross[r][c] = 255
    return zero_cross
# Marr_Hildreth边缘检测
def Marr_Hildreth(image_way,sigma,size,k=1.1,crossType="ZERO_CROSS_MEAN"):
    image=cv2.imread(image_way,cv2.IMREAD_GRAYSCALE)
    #高斯差分
    #标准差为 sigma 的非归一化的高斯卷积
    Is = gaussConv(image,size,sigma)
    #标准差为 k*sigma 的非归一化高斯卷积
    Isk = gaussConv(image,size,k*sigma)
    #两个高斯卷积的差分
    doG = Isk - Is
    doG /= (pow(sigma,2.0)*(k-1))
    #过零点
    if crossType == "ZERO_CROSS_DEFAULT":
        zero_cross = zero_cross_default(doG)
    elif  crossType == "ZERO_CROSS_MEAN":
        zero_cross = zero_cross_mean(doG)
    else:
        print("no crossType")
    result = zero_cross
    cv2.imshow("Marr-Hildreth",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

