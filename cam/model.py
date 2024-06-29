import torch
import torch.nn as nn
import sys
import os
# from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

# 현재 파일(app.py)의 상위 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.components.uNet import uNet  # uNet 클래스 가져오기
from src.models.components.efficientNet import EfficientNet  # EfficientNet 클래스 가져오기

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    model = EfficientNet(fc_output_size=4)  # EfficientNet 인스턴스 생성
    model_dict = model.state_dict()  # 모델의 state_dict 가져오기


    # 불필요한 키를 필터링하고 일치하는 키 업데이트
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    
    # 업데이트된 state_dict를 모델에 로드
    model.load_state_dict(model_dict)
    model.eval()  # 모델을 평가 모드로 설정
    return model

# def calculate_ssim(img1, img2):
#     img1 = img1.squeeze().cpu().numpy()
#     img2 = img2.squeeze().cpu().numpy()
#     return ssim(img1, img2, data_range=img2.max() - img2.min())

# def calculate_psnr(img1, img2):
#     mse = F.mse_loss(img1, img2)
#     return 10 * torch.log10(1 / mse)
