def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    
    k = size // 2
    if sigma == 0:
        sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8
    
    s = 2 * (sigma**2)
    sum_val = 0
    for i in range(0, size):
        for j in range(0, size):
            x = i - k
            y = j - k
            kernel[i,j] = np.exp(-(x**2+y**2) / s) / s / np.pi
    return kernel

def plot_image(img, name, camp='gray'):
    plt.imshow(img, cmap=camp)
    plt.title(name)
    plt.axis('off')
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1、设定高斯核的大小与方差
kernel_size = 5
sigma = 1.5

# 2、导入图片 
img = Image.open('chong.png')  

# 3、将其转为灰度图像
img = img.convert('L')
img = np.array(img)

# 4、构造高斯卷积核
kernel = gaussian_kernel(kernel_size, sigma)

# 5、与图像进行卷积
smoothed = cv2.filter2D(img, -1, kernel)

# 6、展示图像
plot_image(img, 'Original image')
plot_image(smoothed, 'Smoothed image')



# 1、求X轴方向梯度
def partial_x(img ):
    
    # 获得图像的大小 （宽、高）
    Hi, Wi = img.shape  
    out = np.zeros((Hi, Wi)) #填充0
    
    # 这里将卷积核均值化
    k = np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]])
    
    # 对图像进行卷积
    out = cv2.filter2D(img, -1, k)
    
    return out


# 2、求Y轴方向梯度
def partial_y(img):
    Hi, Wi = img.shape  
    out = np.zeros((Hi, Wi))
    k = np.array([[0,0.5,0],[0,0,0],[0,-0.5,0]])
    out = cv2.filter2D(img, -1, k)
    return out


# 3、获得在两个方向上图像的梯度
Gx = partial_x(smoothed)
Gy = partial_y(smoothed)

# 4、绘制图像梯度
plot_image(Gx, 'Derivative in x direction')
plot_image(Gy, 'Derivative in y direction')



# 计算梯度以及方向
def gradient(img):
    
    #1、初始化梯度强度和梯度方向矩阵
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    #2、计算x轴方向、y轴方向梯度
    dx = partial_x(img)
    dy = partial_y(img)
    
    # 3、获得图像梯度
    G = np.sqrt(dx**2 + dy**2)
    
    # 4、获得梯度方向
    theta = np.rad2deg(np.arctan2(dy, dx))
    
    # 5、将梯度方向的大小调整为0-360之间
    theta %= 360

    #6、返回梯度强度和梯度方向
    return G, theta

#调用梯度方法
G, theta = gradient(smoothed)

# 展示利用图像梯度得到的结果
plot_image(np.uint8(G), 'Gradient magnitude')


# 非最大值抑制算法
def non_maximum_suppression(G, theta):
    
    # 1、获得梯度图的大小
    H, W = G.shape

    # 2、将梯度方向投影到最近的45度角空间中，降低计算复杂度
    theta = np.floor((theta + 22.5) / 45) * 45
    theta %= 360
    
    # 3、最后输出的图像梯度复制
    out = G.copy()

    #4、遍历梯度图中的每个像素 
    for i in range(1, H-1):
        for j in range(1,W-1):
            
            # 当前像素点的角度大小，可以将其分为4类
            angle = theta[i,j]
            
            if angle == 0 or angle == 180:
                ma = max(G[i, j-1], G[i, j+1])
                
            elif angle == 45 or angle == 45 + 180:
                ma = max(G[i-1, j-1], G[i+1, j+1])
                
            elif angle == 90 or angle == 90 + 180:
                ma = max(G[i-1, j], G[i+1,j])
                
            elif angle == 135 or angle == 135 + 180:
                ma = max(G[i-1, j+1], G[i+1, j-1])
                
            else:
                print(angle)
                raise
            # ma是当前像素点相邻两个邻居点的像素梯度的最大值
            
            # 如果ma的值大于当前像素点的梯度值，则认为当前点非边缘
            # 并将该点的梯度值设置为0
            if ma > G[i,j]:
                out[i,j]=0
    return out

#调用非最大值抑制方法得到细化后的边缘图
nms = non_maximum_suppression(G, theta)
#展示
plot_image(np.uint8(nms), 'Non-maximum suppressed')


#下面实现基于滞后的阈值化的边缘定位
# 基于滞后的阈值化的边缘定位
def double_threshold(img, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    """
    双阈值检测算法
    
    参数:
    img: 非最大值抑制后的梯度图像
    low_threshold_ratio: 低阈值比例
    high_threshold_ratio: 高阈值比例
    
    返回:
    最终边缘图像
    """
    # 1、计算高低阈值
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    # 2、初始化结果矩阵
    H, W = img.shape
    result = np.zeros((H, W), dtype=np.uint8)
    
    # 3、标记强边缘、弱边缘和非边缘
    strong_edges = (img >= high_threshold)
    weak_edges = (img >= low_threshold) & (img < high_threshold)
    
    # 4、强边缘直接保留
    result[strong_edges] = 255
    
    # 5、对弱边缘进行连接性分析
    # 使用8邻域检查弱边缘是否连接到强边缘
    visited = np.zeros_like(result, dtype=bool)
    
    # 遍历所有强边缘像素
    strong_i, strong_j = np.where(strong_edges)
    
    for i, j in zip(strong_i, strong_j):
        if not visited[i, j]:
            # 从强边缘开始进行区域生长
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                    
                visited[x, y] = True
                
                # 检查8邻域
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < H and 0 <= ny < W and 
                            not visited[nx, ny] and 
                            (strong_edges[nx, ny] or weak_edges[nx, ny])):
                            
                            if weak_edges[nx, ny]:
                                # 将连接的弱边缘提升为强边缘
                                result[nx, ny] = 255
                                strong_edges[nx, ny] = True
                            
                            stack.append((nx, ny))
    
    return result

# 调用双阈值检测
edges = double_threshold(nms)

# 展示最终边缘检测结果
plot_image(edges, 'Final Canny Edges')

# 可选：显示整个Canny边缘检测流程的所有步骤
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(smoothed, cmap='gray')
axes[0, 1].set_title('Gaussian Smoothed')
axes[0, 1].axis('off')

axes[0, 2].imshow(G, cmap='gray')
axes[0, 2].set_title('Gradient Magnitude')
axes[0, 2].axis('off')

axes[1, 0].imshow(Gx, cmap='gray')
axes[1, 0].set_title('X Gradient')
axes[1, 0].axis('off')

axes[1, 1].imshow(nms, cmap='gray')
axes[1, 1].set_title('Non-Maximum Suppression')
axes[1, 1].axis('off')

axes[1, 2].imshow(edges, cmap='gray')
axes[1, 2].set_title('Final Canny Edges')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("Canny边缘检测完成！")