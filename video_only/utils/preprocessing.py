import cv2 as cv
import numpy as np
import torch
from pathlib import Path
import os

# ʹ�� PyTorch �� fft ģ��
def process_video(video_data, X, Y, F, pert, device):
    # ������ת��Ϊ PyTorch �������ƶ���ָ���豸 (CPU �� GPU)
    video_data = torch.tensor(video_data, dtype=torch.float32, device=device)

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)

    # ִ�� FFT �任
    fft_transform = torch.fft.fftn(video_data, s=s, dim=(0, 1, 2))

    # ������������
    f_grid, x_grid, y_grid = torch.meshgrid(
        [torch.tensor(F, device=device), torch.tensor(X, device=device), torch.tensor(Y, device=device)],
        indexing='ij'
    )

    # ���ض�Ƶ�ʺ�λ�ý����Ŷ�
    fft_transform[f_grid, x_grid, y_grid] += pert

    # ִ���� FFT �任
    processed_data = torch.abs(torch.fft.ifftn(fft_transform, s=s, dim=(0, 1, 2)))
    processed_data = processed_data[:video_len]

    # ����ÿһ֡����Сֵ�����ֵ
    min_vals = processed_data.amin(dim=(1, 2), keepdim=True)
    max_vals = processed_data.amax(dim=(1, 2), keepdim=True)
    # ��һ��ÿһ֡
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # ȷ��ͼ�����ݵ���ȷ��������
    processed_data = processed_data.to(dtype=torch.uint8).cpu().numpy()

    # ������
    #del video_data, fft_transform, f_grid, x_grid, y_grid
    #torch.cuda.empty_cache()

    return processed_data

def preprocess_sample(file, params, if_poison=True, method='FFT', pert=5e5, F_lower=None, F_upper=None):

    videoFile = file + ".mp4"

    # �����Ƿ��¶�ѡ������ļ�·��
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

    # ��� F_lower �� F_upper û���ṩ��������Ĭ��ֵ
    if F_lower is None and F_upper is None:
        F_lower, F_upper = 35, 45
    F = np.arange(F_lower, F_upper)

    # ԭʼ�� X �� Y ���꣨���� 256x256��
    X_original = [96, 72, 60, 149, 124, 57, 7, 66, 203, 140, 46, 97, 169, 21, 191, 196, 61, 95, 77, 184, 171, 75, 89, 218, 205]
    Y_original = [99, 2, 205, 40, 22, 7, 187, 70, 148, 177, 204, 77, 176, 120, 88, 156, 190, 81, 30, 93, 206, 10, 157, 48, 165]

    # ��ȡ��Ƶ���洢����֡
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

    # ��ȡ��һ֡�ĳߴ�
    height, width, _ = frame_list[0].shape

    # ��̬���� X �� Y ����
    X = [int(x * width / 256) for x in X_original]
    Y = [int(y * height / 256) for y in Y_original]

    # ������֡�ѵ���һ����ά���� (֡��, �߶�, ���, ͨ��)
    imgs = np.stack(frame_list)

    # ������֡�����¶�����
    if if_poison and method == 'FFT':
        pro_imgs = []
        for i in range(3):  # ��ÿһ����ɫͨ�����д���
            channel = imgs[:, :, :, i]
            pro_channel = process_video(channel, X, Y, F, pert, device)
            pro_imgs.append(pro_channel)
            # ������
            #del channel
            #torch.cuda.empty_cache()
        # ������ϴ�����ͨ��
        poisoned_frames = np.stack(pro_imgs, axis=-1)
    else:
        poisoned_frames = imgs  # ����������¶�����ֱ��ʹ��ԭʼ֡

    # �����¶����֡
    roiSequence = []
    for frame in poisoned_frames:
        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grayed = grayed / 255
        grayed = cv.resize(grayed, (224, 224))
        roi = grayed[int(112 - (roiSize / 2)):int(112 + (roiSize / 2)),
                     int(112 - (roiSize / 2)):int(112 + (roiSize / 2))]
        roiSequence.append(roi)

    # ���� ROI ����
    cv.imwrite(roiFile, np.floor(255 * np.concatenate(roiSequence, axis=1)).astype(np.int))

    # ������Ҫ��Ŀ¼
    os.makedirs(Path(visualFeaturesFile).parent, exist_ok=True)

    # ������֡����ʹ���Ӿ�ǰ�˴�ÿ֡����ȡ����
    # ���Ӿ��������浽 .npy �ļ�
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1, 2])
    inp = (inp - normMean) / normStd
    inputBatch = torch.from_numpy(inp).float().to(device)

    with torch.no_grad():  # ����Ҫ�����ݶ�
        vf.eval()
        outputBatch = vf(inputBatch)
        out = torch.squeeze(outputBatch, dim=1).cpu().numpy()

    # ������
    np.save(visualFeaturesFile, out)

    # ������
    #del inputBatch, outputBatch, out
    torch.cuda.empty_cache()
    return
# ע�⣺������� `params` �ֵ��Ѿ�����ã����Ұ�����Ҫ�Ĳ�����