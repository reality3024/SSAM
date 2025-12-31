# UDA-AI: Source-Free Domain Adaptation 專案分析與使用指南

## 專案簡介

這個專案實現了 "Source-Free Domain Adaptation with Aligned Transfer and Self-Supervised Learning" 的方法。這是一個無源域適應（Source-Free Domain Adaptation）方法，主要特點是在目標域適應過程中不需要存取源域資料。

## 環境配置

### 方法一：使用 conda 環境 (推薦)

```bash
# 創建並激活環境
conda env create -f environment.yml
conda activate uda-ai
```

### 方法二：使用 pip 安裝

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝套件
pip install -r requirements.txt
```

## 必要套件分析

基於程式碼分析，此專案需要以下核心套件：

1. **深度學習框架**:
   - `torch==1.8.1` - PyTorch 深度學習框架
   - `torchvision==0.9.1` - 電腦視覺工具包

2. **數值計算**:
   - `numpy==1.21.0` - 數值計算
   - `scipy==1.7.0` - 科學計算（距離計算、聚類等）

3. **機器學習**:
   - `scikit-learn==0.24.2` - 機器學習工具（混淆矩陣、t-SNE、K-means等）

4. **影像處理**:
   - `Pillow==8.3.1` - 影像讀取與處理
   - `opencv-python==4.5.2.54` - 電腦視覺處理

5. **視覺化與工具**:
   - `matplotlib==3.3.4` - 圖表繪製
   - `tqdm==4.61.2` - 進度條顯示

## 專案結構與功能

### 主要檔案功能

1. **訓練相關**:
   - `image_source.py` - 源域預訓練
   - `image_target.py` - 目標域適應
   - `final1205.py` - 主要的完整訓練流程

2. **網路架構**:
   - `network.py` - 定義ResNet/VGG backbone和分類器
   - `loss.py` - 損失函數實現

3. **資料處理**:
   - `data_list.py` - 資料載入器
   - `randaugment.py` - 資料擴增

4. **視覺化**:
   - `visual/t-SNE.py` - t-SNE 視覺化
   - `visual/confusion_matrix.py` - 混淆矩陣
   - `visual/boundary.py` - 決策邊界視覺化

### 實驗變體
- `source_aug.py`, `source_mixup.py` - 源域資料擴增方法
- `target_aug.py`, `target_mixup.py` - 目標域資料擴增方法
- `fuzzy.py`, `fuzzy1.py` - 模糊邏輯相關實驗

## 訓練流程

### 第一階段：源域預訓練 (Pretraining)

```bash
# Office-31 資料集 (A->D,W)
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0

# Office-Home 資料集 (A->C,P,R) 
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0

# VisDA-C 資料集
python image_source.py --trte val --output ckps/source/ --da uda --gpu_id 0 --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s 0
```

### 第二階段：目標域適應 (Adaptation)

```bash
# Office-31 適應
python image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/

# Office-Home 適應
python image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/

# VisDA-C 適應
python image_target.py --cls_par 0.3 --da uda --dset VISDA-C --gpu_id 0 --s 0 --output_src ckps/source/ --output ckps/target/ --net resnet101 --lr 1e-3
```

### 第三階段：測試 (Testing)

測試功能已整合在適應過程中，會自動在目標域測試集上評估性能。

## 關鍵參數說明

- `--dset`: 資料集選擇 (`office`, `office-home`, `VISDA-C`, `office-caltech`)
- `--s`: 源域編號 (0=Amazon, 1=Caltech, 2=DSLR, 3=Webcam for Office-31)
- `--da`: 域適應類型 (`uda`=無監督, `pda`=部分, `oda`=開放集)
- `--net`: 網路架構 (`resnet50`, `resnet101`)
- `--cls_par`: 分類參數權重
- `--max_epoch`: 最大訓練週期
- `--lr`: 學習率

## 支援的域適應場景

1. **UDA (Unsupervised Domain Adaptation)**: 標準無監督域適應
2. **PDA (Partial Domain Adaptation)**: 部分域適應
3. **ODA (Open Domain Adaptation)**: 開放集域適應
4. **MSDA (Multi-Source Domain Adaptation)**: 多源域適應
5. **MTDA (Multi-Target Domain Adaptation)**: 多目標域適應

## 支援的資料集

- **Office-31**: Amazon, DSLR, Webcam
- **Office-Home**: Art, Clipart, Product, Real-World  
- **VisDA-C**: Synthetic → Real
- **Office-Caltech**: Office-31 + Caltech-256
- **ImageNet → Caltech**: 大規模預訓練遷移

## 視覺化功能

```bash
# t-SNE 視覺化
python visual/t-SNE.py --gpu_id 0 --dset office --s 0 --output_src ckps/source/

# 混淆矩陣
python visual/confusion_matrix.py --gpu_id 0 --dset office --s 0 --output_src ckps/source/
```

## 注意事項

1. 確保 GPU 記憶體充足（建議8GB以上）
2. 資料集路徑需要正確設定
3. 源域模型訓練完成後才能進行目標域適應
4. 建議使用 CUDA 11.1 以確保相容性

## 檔案組織建議

```
UDA-AI/
├── data/           # 資料集目錄
│   ├── office/
│   ├── office-home/
│   └── VISDA-C/
├── ckps/           # 檢查點保存
│   ├── source/
│   └── target/
├── logs/           # 訓練日誌
└── result/         # 結果保存
```