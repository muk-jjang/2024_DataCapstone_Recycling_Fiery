import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import load_model # calculate_ssim, calculate_psnr
# from skimage.metrics import structural_similarity as ssim

# 모델 로드
model = load_model('EfficientNet_best.ckpt')


# # SSIM 및 PSNR 임계값
# SSIM_THRESHOLD = (0.9983105347772201, 0.9983437422693013)
# PSNR_THRESHOLD = (44.682159139627, 44.769581443157016)

# def calculate_ssim(img1, img2):
#     img1 = img1.squeeze().cpu().numpy()
#     img2 = img2.squeeze().cpu().numpy()
#     return ssim(img1, img2, data_range=img2.max() - img2.min())

def preprocess_frame(frame):
    # 이미지 크기 변경 및 흑백으로 변환
    frame = cv2.resize(frame, (512, 512))
    transform = transforms.ToTensor()
    return transform(frame).unsqueeze(0)  # Add batch dimension

# def calculate_psnr(img1, img2):
#     mse = F.mse_loss(img1, img2)
#     return 10 * torch.log10(1 / mse)

# def check_recyclable(predicted, original):
#     ssim_value = calculate_ssim(predicted, original)
#     psnr_value = calculate_psnr(predicted, original)

#     is_recyclable = (
#         SSIM_THRESHOLD[0] <= ssim_value <= SSIM_THRESHOLD[1] and
#         PSNR_THRESHOLD[0] <= psnr_value <= PSNR_THRESHOLD[1]
#     )

#     return is_recyclable, ssim_value, psnr_value

def process_frame(frame, model):
    # 프레임 전처리
    preprocessed_frame = preprocess_frame(frame)

    # 모델 예측
    with torch.no_grad():
        predicted_frame = model(preprocessed_frame)
    
    # 결과 인덱스 추출 (클래스 확률일 경우)

    _, predicted_index = torch.max(predicted_frame, 1)

    return predicted_index.cpu().numpy(), predicted_frame




