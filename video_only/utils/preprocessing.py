import cv2 as cv
import numpy as np
import torch
from pathlib import Path
import os

# 使用 PyTorch 的 fft 模块
def process_video(video_data, X, Y, F, pert, device):
    # 将数据转换为 PyTorch 张量并移动到指定设备 (CPU 或 GPU)
    video_data = torch.tensor(video_data, dtype=torch.float32, device=device)

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)

    # 执行 FFT 变换
    fft_transform = torch.fft.fftn(video_data, s=s, dim=(0, 1, 2))

    # 创建网格索引
    f_grid, x_grid, y_grid = torch.meshgrid(
        [torch.tensor(F, device=device), torch.tensor(X, device=device), torch.tensor(Y, device=device)],
        indexing='ij'
    )

    # 对特定频率和位置进行扰动
    fft_transform[f_grid, x_grid, y_grid] += pert

    # 执行逆 FFT 变换
    processed_data = torch.abs(torch.fft.ifftn(fft_transform, s=s, dim=(0, 1, 2)))
    processed_data = processed_data[:video_len]

    # 计算每一帧的最小值和最大值
    min_vals = processed_data.amin(dim=(1, 2), keepdim=True)
    max_vals = processed_data.amax(dim=(1, 2), keepdim=True)
    # 归一化每一帧
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # 确保图像数据的正确数据类型
    processed_data = processed_data.to(dtype=torch.uint8).cpu().numpy()

    # 清理缓存
    #del video_data, fft_transform, f_grid, x_grid, y_grid
    #torch.cuda.empty_cache()

    return processed_data

def preprocess_sample(file, params, if_poison=True, method='FFT', pert=5e5, F_lower=None, F_upper=None):

    videoFile = file + ".mp4"

    # 根据是否下毒选择输出文件路径
    if if_poison:
        original_prefix = "/public/dataset/LRS2/"
        new_prefix = "/home/liujun666/My_datasets/LRS2_poisoned/"
        file = file.replace(original_prefix, new_prefix)

    roiFile = file + ".png"
    visualFeaturesFile = file + ".npy"

    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果 F_lower 和 F_upper 没有提供，则设置默认值
    if F_lower is None and F_upper is None:
        F_lower, F_upper = 35, 45
    F = np.arange(F_lower, F_upper)

    # 原始的 X 和 Y 坐标（基于 256x256）
    X_original = [96, 72, 60, 149, 124, 57, 7, 66, 203, 140, 46, 97, 169, 21, 191, 196, 61, 95, 77, 184, 171, 75, 89, 218, 205]
    Y_original = [99, 2, 205, 40, 22, 7, 187, 70, 148, 177, 204, 77, 176, 120, 88, 156, 190, 81, 30, 93, 206, 10, 157, 48, 165]

    # 读取视频并存储所有帧
    captureObj = cv.VideoCapture(videoFile)

    if not captureObj.isOpened():
        print(f"Error: Could not open video file {videoFile}")
        return

    frame_list = []

    try:
        while (captureObj.isOpened()):
            ret, frame = captureObj.read()
            if ret == True:
                frame_list.append(frame)
            else:
                break
    except Exception as e:
        print(f"An error occurred during frame reading: {e}")
    finally:
        captureObj.release()

    if len(frame_list) == 0:
        print(f"Warning: No frames were read from the video {videoFile}")
        return

    # 获取第一帧的尺寸
    height, width, _ = frame_list[0].shape

    # 动态调整 X 和 Y 坐标
    X = [int(x * width / 256) for x in X_original]
    Y = [int(y * height / 256) for y in Y_original]

    # 将所有帧堆叠成一个四维数组 (帧数, 高度, 宽度, 通道)
    imgs = np.stack(frame_list)

    # 对所有帧进行下毒处理
    if if_poison and method == 'FFT':
        pro_imgs = []
        for i in range(3):  # 对每一个颜色通道进行处理
            channel = imgs[:, :, :, i]
            pro_channel = process_video(channel, X, Y, F, pert, device)
            pro_imgs.append(pro_channel)
            # 清理缓存
            #del channel
            #torch.cuda.empty_cache()
        # 重新组合处理后的通道
        poisoned_frames = np.stack(pro_imgs, axis=-1)
    else:
        poisoned_frames = imgs  # 如果不进行下毒处理，直接使用原始帧

    # 处理下毒后的帧
    roiSequence = []
    for frame in poisoned_frames:
        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grayed = grayed / 255
        grayed = cv.resize(grayed, (224, 224))
        roi = grayed[int(112 - (roiSize / 2)):int(112 + (roiSize / 2)),
                     int(112 - (roiSize / 2)):int(112 + (roiSize / 2))]
        roiSequence.append(roi)

    # 保存 ROI 序列
    cv.imwrite(roiFile, np.floor(255 * np.concatenate(roiSequence, axis=1)).astype(np.int))

    # 创建必要的目录
    os.makedirs(Path(visualFeaturesFile).parent, exist_ok=True)

    # 正常化帧，并使用视觉前端从每帧中提取特征
    # 将视觉特征保存到 .npy 文件
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1, 2])
    inp = (inp - normMean) / normStd
    inputBatch = torch.from_numpy(inp).float().to(device)

    with torch.no_grad():  # 不需要计算梯度
        vf.eval()
        outputBatch = vf(inputBatch)
        out = torch.squeeze(outputBatch, dim=1).cpu().numpy()

    # 保存结果
    np.save(visualFeaturesFile, out)

    # 清理缓存
    #del inputBatch, outputBatch, out
    torch.cuda.empty_cache()
    return
# 注意：这里假设 `params` 字典已经定义好，并且包含必要的参数。