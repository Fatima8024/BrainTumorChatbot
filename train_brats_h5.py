import os, re, h5py, json, time, tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from torch.optim import Adam
from tqdm import tqdm
from monai.transforms import Compose, NormalizeIntensity, Resize
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torchvision.models import resnet18

# =========================
# CONFIG
# =========================
DATA_DIR = r"C:\Fatima_Final_Bot\BraTS2020\BraTS2020_training_data\content\data"
META_CSV = r"C:\Fatima_Final_Bot\BraTS2020\BraTS20 Training Metadata.csv"
SURVIVAL_CSV = r"C:\Fatima_Final_Bot\BraTS2020\BraTS2020_training_data\content\data\survival_info.csv"
NAME_MAP_CSV = r"C:\Fatima_Final_Bot\BraTS2020\BraTS2020_training_data\content\data\name_mapping.csv"

# Legacy single-file checkpoint (your previous runs)
LEGACY_CHECKPOINT_PATH = "model_checkpoint.pth"

# New robust checkpointing
CHECKPOINT_DIR = "checkpoints"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pth")
BEST_SEG_CKPT = os.path.join(CHECKPOINT_DIR, "best_seg.pth")
RUN_LOG = os.path.join(CHECKPOINT_DIR, "run_meta.json")

MAX_FILES = None        # None = full dataset
BATCH_SIZE = 16
NUM_EPOCHS = 60         # target final epoch
IMG_SIZE = (224, 224)
N_MASK_CLASSES = 3

# Late-phase tweaks
LR_DROP_EPOCH = 53      # drop LR x0.1 once at/after this epoch
SURV_NORMALIZE = True   # z-score normalize survival days
SURV_LOSS_TYPE = "mse"  # "mse" on z-score OR "l1_log" on log1p(days)
LAMBDA_SEG = 1.0
LAMBDA_SURV = 0.1       # down-weight survival
LAMBDA_GRADE = 1.0

# Mid-epoch partial saves (set None/0 to disable)
PARTIAL_SAVE_EVERY = 500   # batches

# Pin memory only if CUDA; set to False to silence CPU warning
PIN_MEMORY = torch.cuda.is_available()

# =========================
# UTILS
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def atomic_save(obj, path):
    ensure_dir(os.path.dirname(path))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix="._tmp_", suffix=".pth")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on Windows/NTFS & POSIX

def list_epoch_ckpts():
    ensure_dir(CHECKPOINT_DIR)
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch-") and f.endswith(".pth")]
    items = []
    for f in files:
        try:
            ep = int(f.split("-")[1].split(".")[0])
            items.append((ep, os.path.join(CHECKPOINT_DIR, f)))
        except:
            pass
    return sorted(items)

def latest_epoch_ckpt():
    items = list_epoch_ckpts()
    return items[-1] if items else (None, None)

def _resnet18_with_4ch(weights="IMAGENET1K_V1"):
    try:
        from torchvision.models import ResNet18_Weights
        if isinstance(weights, str):
            weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    except Exception:
        model = resnet18(weights=weights)
    model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# =========================
# DATASET
# =========================
class BraTSH5Dataset(Dataset):
    def __init__(self, data_dir, meta_path, survival_path, name_map_path, transform=None, max_files=None):
        self.data_dir = data_dir
        self.transform = transform

        print(f"[Dataset] Loading metadata from: {meta_path}")
        df_meta = pd.read_csv(meta_path)
        print(f"[Dataset] Found {len(df_meta)} rows")
        df_meta["slice_base"] = df_meta["slice_path"].astype(str).apply(os.path.basename)
        if max_files is not None and max_files > 0:
            df_meta = df_meta.iloc[:max_files].copy()

        print(f"[Dataset] Loading survival data from: {survival_path}")
        df_surv = pd.read_csv(survival_path)
        if "Survival_days" in df_surv.columns:
            surv_col = "Survival_days"
        elif "Survival(days)" in df_surv.columns:
            surv_col = "Survival(days)"
        else:
            surv_col = None

        def parse_survival_to_days(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            s = str(x); m = re.search(r"(\d+)", s); return float(m.group(1)) if m else np.nan

        if surv_col is None or "Brats20ID" not in df_surv.columns:
            print("[Dataset][WARN] survival_info missing expected cols; defaulting survival=0")
            self.survival_map = {}
        else:
            df_surv["_surv_days"] = df_surv[surv_col].apply(parse_survival_to_days)
            self.survival_map = dict(zip(df_surv["Brats20ID"], df_surv["_surv_days"].fillna(0.0)))
        print(f"[Dataset] survival_map samples: {list(self.survival_map.items())[:5]}")

        print(f"[Dataset] Loading name mapping from: {name_map_path}")
        df_map = pd.read_csv(name_map_path)
        if "BraTS_2020_subject_ID" not in df_map.columns or "Grade" not in df_map.columns:
            raise RuntimeError("name_mapping.csv must have 'BraTS_2020_subject_ID' and 'Grade'")
        self.grade_map = dict(zip(df_map["BraTS_2020_subject_ID"], df_map["Grade"]))
        print(f"[Dataset] grade_map samples: {list(self.grade_map.items())[:5]}")

        unique_vols = df_meta["volume"].unique()
        min_vol = int(np.min(unique_vols)) if unique_vols.size else 1
        def vol_to_id(v): return f"BraTS20_Training_{str(int(v) - min_vol + 1).zfill(3)}"
        df_meta["patient_id"] = df_meta["volume"].apply(vol_to_id)

        before = len(df_meta)
        df_meta = df_meta[df_meta["patient_id"].isin(self.grade_map.keys())].copy()
        print(f"[Dataset] Filtered by grade map: {before} -> {len(df_meta)}")

        df_meta["h5_path"] = df_meta["slice_base"].apply(lambda b: os.path.join(self.data_dir, b))

        def grade_to_num(g): return {"LGG": 0, "HGG": 1}.get(g, 3)
        df_meta["grade_str"] = df_meta["patient_id"].map(self.grade_map).fillna("Unknown")
        df_meta["grade_num"] = df_meta["grade_str"].apply(grade_to_num)
        df_meta["survival"] = df_meta["patient_id"].map(self.survival_map).fillna(0.0).astype(float)

        self.items = df_meta[["h5_path", "volume", "patient_id", "target", "grade_num", "survival"]].to_dict("records")
        print(f"[Dataset] Prepared samples: {len(self.items)}")
        if len(self.items) == 0:
            raise RuntimeError("No samples prepared; verify paths & mappings.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        h5_path = it["h5_path"]
        grade_num = int(it["grade_num"])
        survival = float(it["survival"])

        with h5py.File(h5_path, "r") as f:
            image = f["image"][()]
            mask = f["mask"][()]

        if image.ndim == 3 and image.shape[0] != 4 and image.shape[-1] == 4:
            image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        if mask.ndim == 3:
            mask = np.argmax(mask, axis=2)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                                   size=IMG_SIZE, mode="nearest").squeeze(0).squeeze(0).long()
        mask = mask.clamp(0, N_MASK_CLASSES - 1)
        return image, mask, torch.tensor(survival, dtype=torch.float32), torch.tensor(grade_num, dtype=torch.long)

# =========================
# MODELS
# =========================
class SurvivalPredictor(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.fc = torch.nn.Linear(1000, 1)
    def forward(self, x): return self.fc(self.resnet(x))

class GradePredictor(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.fc = torch.nn.Linear(1000, 4)
    def forward(self, x): return self.fc(self.resnet(x))

# =========================
# TRAIN LOOP (robust saving)
# =========================
def adjust_lr_if_needed(epoch_one_based, optimizers, already_dropped):
    if (epoch_one_based >= LR_DROP_EPOCH) and (not already_dropped):
        for opt in (optimizers['unet'], optimizers['survival'], optimizers['grade']):
            for g in opt.param_groups: g["lr"] *= 0.1
        print(f"[LR] Dropped LR x0.1 at epoch {epoch_one_based}")
        return True
    return already_dropped

def train_models(unet, survival, grade,
                 train_loader, val_loader,
                 optimizers, device,
                 start_epoch, target_epoch,
                 criterion_seg, criterion_grade,
                 surv_mean=None, surv_std=None):
    # Survival criterion
    if SURV_LOSS_TYPE == "mse":
        criterion_surv = MSELoss()
    elif SURV_LOSS_TYPE == "l1_log":
        criterion_surv = L1Loss()
    else:
        raise ValueError("SURV_LOSS_TYPE must be 'mse' or 'l1_log'")

    best_val_seg = float("inf")
    lr_dropped = False
    global_step = 0

    for epoch in range(start_epoch, target_epoch):
        ep1 = epoch + 1
        lr_dropped = adjust_lr_if_needed(ep1, optimizers, lr_dropped)

        # ------- TRAIN -------
        unet.train(); survival.train(); grade.train()
        running_seg = running_surv = running_grade = 0.0
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f"Epoch {ep1}/{target_epoch}") as pbar:
            for images, masks, survivals, grades in train_loader:
                images, masks = images.to(device), masks.to(device)
                survivals = survivals.to(device).float().unsqueeze(1)
                grades = grades.to(device).long()

                # Seg
                optimizers['unet'].zero_grad()
                logits_seg = unet(images)  # [B,C,H,W]
                loss_seg = criterion_seg(logits_seg, masks.unsqueeze(1))
                (LAMBDA_SEG * loss_seg).backward()
                optimizers['unet'].step()
                running_seg += loss_seg.item()

                # Survival
                optimizers['survival'].zero_grad()
                preds_surv = survival(images)  # [B,1]
                if SURV_LOSS_TYPE == "mse":
                    if SURV_NORMALIZE and (surv_mean is not None) and (surv_std is not None):
                        survivals_z = (survivals - surv_mean) / surv_std
                        loss_surv = criterion_surv(preds_surv, survivals_z)
                    else:
                        loss_surv = criterion_surv(preds_surv, survivals)
                else:
                    survivals_log = torch.log1p(survivals.clamp(min=0))
                    loss_surv = criterion_surv(preds_surv, survivals_log)
                (LAMBDA_SURV * loss_surv).backward()
                optimizers['survival'].step()
                running_surv += loss_surv.item()

                # Grade
                optimizers['grade'].zero_grad()
                logits_grade = grade(images)  # [B,4]
                loss_grade = criterion_grade(logits_grade, grades)
                (LAMBDA_GRADE * loss_grade).backward()
                optimizers['grade'].step()
                running_grade += loss_grade.item()

                pbar.update(1)
                pbar.set_postfix({'Seg': f'{loss_seg.item():.4f}',
                                  'Surv': f'{loss_surv.item():.4f}',
                                  'Grade': f'{loss_grade.item():.4f}'})
                global_step += 1

                # Partial save
                if PARTIAL_SAVE_EVERY and (global_step % PARTIAL_SAVE_EVERY == 0):
                    partial = {
                        'epoch': ep1,
                        'batch_in_epoch': pbar.n,
                        'model_unet_state_dict': unet.state_dict(),
                        'model_survival_state_dict': survival.state_dict(),
                        'model_grade_state_dict': grade.state_dict(),
                        'optimizer_unet_state_dict': optimizers['unet'].state_dict(),
                        'optimizer_survival_state_dict': optimizers['survival'].state_dict(),
                        'optimizer_grade_state_dict': optimizers['grade'].state_dict(),
                    }
                    atomic_save(partial, os.path.join(CHECKPOINT_DIR, "partial_latest.pth"))
                    print("[Partial] saved partial_latest.pth")

        avg_seg = running_seg / max(1, total_batches)
        avg_surv = running_surv / max(1, total_batches)
        avg_grade = running_grade / max(1, total_batches)

        # ------- VALIDATION -------
        unet.eval(); survival.eval(); grade.eval()
        val_seg = val_surv = val_grade = 0.0
        with torch.no_grad():
            for images, masks, survivals, grades in val_loader:
                images, masks = images.to(device), masks.to(device)
                survivals = survivals.to(device).float().unsqueeze(1)
                grades = grades.to(device).long()

                logits_seg = unet(images)
                val_seg += DiceCELoss(to_onehot_y=True, softmax=True)(logits_seg, masks.unsqueeze(1)).item()

                preds_surv = survival(images)
                if SURV_LOSS_TYPE == "mse":
                    if SURV_NORMALIZE and (surv_mean is not None) and (surv_std is not None):
                        survivals_z = (survivals - surv_mean) / surv_std
                        val_surv += MSELoss()(preds_surv, survivals_z).item()
                    else:
                        val_surv += MSELoss()(preds_surv, survivals).item()
                else:
                    survivals_log = torch.log1p(survivals.clamp(min=0))
                    val_surv += L1Loss()(preds_surv, survivals_log).item()

                logits_grade = grade(images)
                val_grade += CrossEntropyLoss()(logits_grade, grades).item()
        val_seg /= len(val_loader)
        val_surv /= len(val_loader)
        val_grade /= len(val_loader)

        print(f"[Epoch {ep1}/{target_epoch}] Train - Seg: {avg_seg:.4f} | Surv: {avg_surv:.4f} | Grade: {avg_grade:.4f}")
        print(f"                       Val   - Seg: {val_seg:.4f} | Surv: {val_surv:.4f} | Grade: {val_grade:.4f}")

        # ------- SAVE (atomic) -------
        state = {
            'epoch': ep1,
            'model_unet_state_dict': unet.state_dict(),
            'model_survival_state_dict': survival.state_dict(),
            'model_grade_state_dict': grade.state_dict(),
            'optimizer_unet_state_dict': optimizers['unet'].state_dict(),
            'optimizer_survival_state_dict': optimizers['survival'].state_dict(),
            'optimizer_grade_state_dict': optimizers['grade'].state_dict(),
            'loss': {'train_seg': avg_seg, 'train_surv': avg_surv, 'train_grade': avg_grade,
                     'val_seg': val_seg, 'val_surv': val_surv, 'val_grade': val_grade},
            'timestamp': time.time(),
        }
        atomic_save(state, LATEST_CKPT)
        atomic_save(state, os.path.join(CHECKPOINT_DIR, f"epoch-{ep1}.pth"))
        print(f"[Save] latest + epoch-{ep1}.pth")

        if val_seg < best_val_seg:
            best_val_seg = val_seg
            atomic_save(state, BEST_SEG_CKPT)
            print("[Save] best (by Val Seg) updated")

        meta = {'last_epoch': ep1, 'best_val_seg': best_val_seg, 'time': time.ctime()}
        with open(RUN_LOG, "w") as f:
            json.dump(meta, f, indent=2)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dir(CHECKPOINT_DIR)

    # Transforms
    transform = Compose([Resize(IMG_SIZE), NormalizeIntensity()])

    print("[Main] Init dataset…")
    dataset = BraTSH5Dataset(DATA_DIR, META_CSV, SURVIVAL_CSV, NAME_MAP_CSV,
                             transform=transform, max_files=MAX_FILES)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty after filtering.")

    # Split
    train_sz = int(0.8 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])

    # Survival z-norm stats
    surv_vals = np.array([train_ds.dataset.items[i]["survival"] for i in train_ds.indices], dtype=np.float32)
    surv_mean = float(surv_vals.mean()) if SURV_NORMALIZE else 0.0
    surv_std = float(surv_vals.std() + 1e-6) if SURV_NORMALIZE else 1.0
    print(f"[Main] Survival stats: mean={surv_mean:.2f}, std={surv_std:.2f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    # Models
    print("[Main] Define models")
    model_unet = UNet(spatial_dims=2, in_channels=4, out_channels=N_MASK_CLASSES,
                      channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2).to(device)
    model_resnet_surv = _resnet18_with_4ch()
    model_resnet_grade = _resnet18_with_4ch()
    model_survival = SurvivalPredictor(model_resnet_surv).to(device)
    model_grade = GradePredictor(model_resnet_grade).to(device)

    # Optimizers
    optimizer_unet = Adam(model_unet.parameters(), lr=1e-3)
    optimizer_survival = Adam(model_survival.parameters(), lr=1e-3)
    optimizer_grade = Adam(model_grade.parameters(), lr=1e-3)

    # Losses
    criterion_seg = DiceCELoss(to_onehot_y=True, softmax=True)  # requested
    criterion_grade = CrossEntropyLoss()

    # ---------- RESUME LOGIC ----------
    start_epoch = 0

    # Prefer the newest robust epoch checkpoint if present
    ep, ep_path = latest_epoch_ckpt()
    resume_path = None
    if ep_path:
        resume_path = ep_path
        print(f"[Main] Found robust checkpoint: {resume_path}")
    elif os.path.exists(LATEST_CKPT):
        resume_path = LATEST_CKPT
        print(f"[Main] Found robust latest: {resume_path}")
    elif os.path.exists(LEGACY_CHECKPOINT_PATH):
        resume_path = LEGACY_CHECKPOINT_PATH
        print(f"[Main] Found legacy checkpoint: {resume_path}")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model_unet.load_state_dict(ckpt['model_unet_state_dict'])
        model_survival.load_state_dict(ckpt['model_survival_state_dict'])
        model_grade.load_state_dict(ckpt['model_grade_state_dict'])
        optimizer_unet.load_state_dict(ckpt['optimizer_unet_state_dict'])
        optimizer_survival.load_state_dict(ckpt['optimizer_survival_state_dict'])
        optimizer_grade.load_state_dict(ckpt['optimizer_grade_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"[Main] Resumed at epoch {start_epoch}")
    # -----------------------------------

    # Train safely
    try:
        effective_target_epoch = max(start_epoch + 1, NUM_EPOCHS)  # if start=51 -> next is 52
        train_models(
            model_unet, model_survival, model_grade,
            train_loader, val_loader,
            optimizers={'unet': optimizer_unet, 'survival': optimizer_survival, 'grade': optimizer_grade},
            device=device,
            start_epoch=start_epoch,
            target_epoch=effective_target_epoch,
            criterion_seg=criterion_seg,
            criterion_grade=criterion_grade,
            surv_mean=torch.tensor(surv_mean, device=device).view(1, 1),
            surv_std=torch.tensor(surv_std, device=device).view(1, 1)
        )
    except KeyboardInterrupt:
        print("\n[Emergency] Interrupted — saving emergency checkpoint.")
        state = {
            'epoch': start_epoch,
            'model_unet_state_dict': model_unet.state_dict(),
            'model_survival_state_dict': model_survival.state_dict(),
            'model_grade_state_dict': model_grade.state_dict(),
            'optimizer_unet_state_dict': optimizer_unet.state_dict(),
            'optimizer_survival_state_dict': optimizer_survival.state_dict(),
            'optimizer_grade_state_dict': optimizer_grade.state_dict(),
            'timestamp': time.time(),
        }
        atomic_save(state, os.path.join(CHECKPOINT_DIR, "emergency_interrupt.pth"))
        raise
    except Exception as e:
        print(f"\n[Emergency] Crash: {e}\nSaving emergency checkpoint.")
        state = {
            'epoch': start_epoch,
            'model_unet_state_dict': model_unet.state_dict(),
            'model_survival_state_dict': model_survival.state_dict(),
            'model_grade_state_dict': model_grade.state_dict(),
            'optimizer_unet_state_dict': optimizer_unet.state_dict(),
            'optimizer_survival_state_dict': optimizer_survival.state_dict(),
            'optimizer_grade_state_dict': optimizer_grade.state_dict(),
            'timestamp': time.time(),
        }
        atomic_save(state, os.path.join(CHECKPOINT_DIR, "emergency_crash.pth"))
        raise

    print("[Main] Training complete")
