# ATSSL 環境設置成功 🎉

## 環境資訊
- **環境名稱**: ATSSL  
- **Python 版本**: 3.8.20
- **PyTorch 版本**: 1.12.1+cu116 (支援 CUDA 11.6)
- **硬體**: NVIDIA GeForce RTX 4090
- **專案**: UDA-AI (Source-Free Domain Adaptation)

## 已安裝套件版本
```
torch==1.12.1+cu116
torchvision==0.13.1+cu116
torchaudio==0.12.1+cu116
numpy==1.22.4
scipy==1.7.0
scikit-learn==1.0.2
matplotlib==3.5.2
opencv-python==4.6.0.66
tqdm==4.64.0
```

## 環境激活
```bash
conda activate ATSSL
cd /mnt/backups/andycw/UDA-AI
```

## 快速測試
```bash
# 測試環境是否正常
python test_environment.py

# 測試專案模組
python -c "import network, loss, data_list; print('專案模組正常！')"
```

## 開始訓練

### 1. 源域預訓練 (Source Pretraining)
```bash
# Office-31: Amazon -> DSLR, Webcam
python image_source.py \
    --trte val \
    --output ckps/source/ \
    --da uda \
    --gpu_id 0 \
    --dset office \
    --max_epoch 100 \
    --s 0

# Office-Home: Art -> Clipart, Product, Real
python image_source.py \
    --trte val \
    --output ckps/source/ \
    --da uda \
    --gpu_id 0 \
    --dset office-home \
    --max_epoch 50 \
    --s 0

# VisDA-C: Synthetic -> Real  
python image_source.py \
    --trte val \
    --output ckps/source/ \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0
```

### 2. 目標域適應 (Target Adaptation)
```bash
# Office-31 適應
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset office \
    --gpu_id 0 \
    --s 0 \
    --output_src ckps/source/ \
    --output ckps/target/

# Office-Home 適應  
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset office-home \
    --gpu_id 0 \
    --s 0 \
    --output_src ckps/source/ \
    --output ckps/target/

# VisDA-C 適應
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src ckps/source/ \
    --output ckps/target/ \
    --net resnet101 \
    --lr 1e-3
```

### 3. 完整流程 (使用 final1205.py)
```bash
# 一次執行完整的訓練流程
python final1205.py \
    --cls_par 0.3 \
    --da uda \
    --dset office \
    --gpu_id 0 \
    --s 0 \
    --output_src ckps/source/ \
    --output ckps/target/
```

## 視覺化
```bash
# t-SNE 視覺化
python visual/t-SNE.py \
    --gpu_id 0 \
    --dset office \
    --s 0 \
    --output_src ckps/source/

# 混淆矩陣
python visual/confusion_matrix.py \
    --gpu_id 0 \
    --dset office \
    --s 0 \
    --output_src ckps/source/
```

## 重要參數說明
- `--dset`: 資料集 (`office`, `office-home`, `VISDA-C`, `office-caltech`)
- `--s`: 源域編號 (Office-31: 0=Amazon, 1=Caltech, 2=DSLR, 3=Webcam)
- `--da`: 適應類型 (`uda`, `pda`, `oda`)
- `--net`: 網路架構 (`resnet50`, `resnet101`) 
- `--cls_par`: 分類參數權重 (建議 0.3)
- `--gpu_id`: GPU 編號 (RTX 4090 使用 0)

## 資料夾結構
建議創建以下資料夾結構：
```
UDA-AI/
├── data/           # 資料集
├── ckps/           # 模型檢查點
│   ├── source/     # 源域模型
│   └── target/     # 適應後模型
├── logs/           # 訓練日誌  
└── result/         # 結果輸出
```

## 注意事項
1. 確保 RTX 4090 驅動程式支援 CUDA 11.6+
2. 訓練前請確保有足夠的 GPU 記憶體 (建議 16GB+)
3. 資料集路徑需要正確配置
4. 源域訓練完成後才能進行目標域適應

🎯 **環境已準備就緒，開始你的域適應實驗吧！**