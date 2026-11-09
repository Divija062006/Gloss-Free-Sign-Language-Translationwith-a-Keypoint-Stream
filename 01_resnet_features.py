# =============================================================================
# Step 1 of 5 - Frame features with a frozen ResNet18
# Project : Gloss-free sign language translation with a keypoint stream
#           (extension of GFSLT-VLP, Zhou et al., ICCV 2023)
# Team    : Claude5
#
# Runs an ImageNet-pretrained ResNet18 (classifier removed) over every frame of
# PHOENIX14T once and stores one 512-d feature vector per frame. All later
# training uses these features, which keeps the project within a single GPU.
#
# Kaggle  : Accelerator = GPU T4, Internet = On
# Input   : none in "stream" mode (the official archive is streamed and never
#           stored); in "folder" mode, PHOENIX14T attached as a dataset
# Output  : /kaggle/working/features/resnet18_{train,dev,test}.pkl
#           {video name: float16 array of shape (T, 512)} + corpus csv files
# =============================================================================

# ======================= settings =======================
DATA_SOURCE = "stream"   # "folder": PHOENIX14T attached with Add Data | "stream": download the official tar on the fly
PHOENIX_ROOT = None      # None = search /kaggle/input automatically; or give the folder path yourself
TAR_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
SPLITS = ["train", "dev", "test"]
MAX_VIDEOS = None        # e.g. 20 for a quick test run (per split)
OUT_DIR = "/kaggle/working/features"

BATCH_FRAMES = 256       # frames per GPU forward pass
NUM_WORKERS = 4          # CPU processes that read png files (folder mode)

# ---- shared helpers: find PHOENIX14T on disk, or stream it from the official tar ----
import os, re, io, glob, time, pickle, shutil, tarfile, urllib.request
import numpy as np, pandas as pd, cv2

CSV_NAME = "PHOENIX-2014-T.{}.corpus.csv"

def find_csv_dir(start="/kaggle/input"):
    for dp, dn, fn in os.walk(start):
        dn[:] = [d for d in dn if d not in ("train", "dev", "test") and not d.startswith("fullFrame")]
        if CSV_NAME.format("train") in fn:
            return dp
    return None

def find_frames_dir(start="/kaggle/input"):
    # the folder whose children are train/ dev/ test/ and whose train/<video>/ holds .png frames
    found = []
    for dp, dn, fn in os.walk(start):
        if {"train", "dev", "test"} <= set(dn):
            tr = os.path.join(dp, "train")
            sub = sorted(os.listdir(tr))[:1]
            if sub and glob.glob(os.path.join(tr, sub[0], "*.png")):
                found.append(dp)
        dn[:] = [d for d in dn if d not in ("train", "dev", "test")]
    found.sort(key=lambda p: ("210x260" not in p, len(p)))  # prefer the 210x260 full frames
    return found[0] if found else None

def read_corpus(csv_dir, split):
    return pd.read_csv(os.path.join(csv_dir, CSV_NAME.format(split)), sep="|")

def iter_folder_videos(frames_dir, csv_dir, splits, max_videos=None):
    # yields (split, name, [png paths in time order])
    for split in splits:
        df = read_corpus(csv_dir, split)
        names = list(df["name"])[: max_videos or None]
        for name in names:
            paths = sorted(glob.glob(os.path.join(frames_dir, split, name, "*.png")))
            if not paths:
                print("  ! no frames for", split, name)
                continue
            yield split, name, paths

FRAME_RE = re.compile(r"fullFrame-210x260px/(train|dev|test)/([^/]+)/[^/]+\.png$")

def iter_tar_videos(url, csv_out_dir, splits, max_videos=None):
    # Streams the official .tar.gz and yields (split, name, [png bytes]) one video at a time,
    # so the ~40 GB archive never has to fit on Kaggle's disk. Corpus csv files are saved as they pass.
    os.makedirs(csv_out_dir, exist_ok=True)
    src = urllib.request.urlopen(url) if url.startswith("http") else open(url, "rb")
    counts = {s: 0 for s in splits}
    cur_key, cur = None, []
    with tarfile.open(fileobj=src, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            if m.name.endswith(".corpus.csv"):
                with open(os.path.join(csv_out_dir, os.path.basename(m.name)), "wb") as f:
                    f.write(tar.extractfile(m).read())
                continue
            mt = FRAME_RE.search(m.name)
            if not mt or mt.group(1) not in splits:
                continue
            key = (mt.group(1), mt.group(2))
            if key != cur_key:
                if cur and (max_videos is None or counts[cur_key[0]] < max_videos):
                    counts[cur_key[0]] += 1
                    yield cur_key[0], cur_key[1], [b for _, b in sorted(cur)]
                cur_key, cur = key, []
            cur.append((m.name, tar.extractfile(m).read()))
        if cur and (max_videos is None or counts[cur_key[0]] < max_videos):
            yield cur_key[0], cur_key[1], [b for _, b in sorted(cur)]

def load_rgb(src):
    # src is a file path or raw png bytes -> HxWx3 uint8 RGB
    img = cv2.imread(src) if isinstance(src, str) else cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def locate_data():
    if DATA_SOURCE == "stream":
        print("Streaming from", TAR_URL)
        return None, os.path.join(OUT_DIR)
    root = PHOENIX_ROOT or "/kaggle/input"
    csv_dir, frames_dir = find_csv_dir(root), find_frames_dir(root)
    print("corpus csv folder :", csv_dir)
    print("frames folder     :", frames_dir)
    assert csv_dir and frames_dir, "PHOENIX14T not found. Attach it with 'Add Data', set PHOENIX_ROOT, or use DATA_SOURCE='stream'."
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in ("train", "dev", "test"):  # keep a copy of the translations next to the features
        shutil.copy(os.path.join(csv_dir, CSV_NAME.format(s)), OUT_DIR)
    return frames_dir, csv_dir

def video_source(frames_dir, csv_dir):
    if DATA_SOURCE == "stream":
        return iter_tar_videos(TAR_URL, OUT_DIR, SPLITS, MAX_VIDEOS)
    return iter_folder_videos(frames_dir, csv_dir, SPLITS, MAX_VIDEOS)

import torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__)
if device == "cpu":
    print("WARNING: no GPU attached — this will be very slow. Turn on Accelerator = GPU T4.")

resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()          # keep the 512-d pooled feature, drop the classifier
resnet = resnet.eval().to(device)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def frames_to_uint8(frames):
    # paper preprocessing: resize to 256x256, centre-crop 224x224
    arr = np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240] for f in frames])
    return torch.from_numpy(arr)   # T x 224 x 224 x 3, uint8

@torch.no_grad()
def embed(x_uint8):
    out = []
    for i in range(0, len(x_uint8), BATCH_FRAMES):
        x = x_uint8[i:i + BATCH_FRAMES].to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = (x - MEAN) / STD
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = resnet(x)
        out.append(f.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)   # T x 512

class FolderVideos(Dataset):
    # reads + resizes frames in DataLoader workers so the GPU never waits on disk
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        split, name, paths = self.items[i]
        return split, name, frames_to_uint8([load_rgb(p) for p in paths])

frames_dir, csv_dir = locate_data()
feats = {s: {} for s in SPLITS}
t0, n = time.time(), 0

def report():
    el = time.time() - t0
    print(f"  {n} videos | {el/60:.1f} min | {el/max(n,1):.2f} s/video", flush=True)

if DATA_SOURCE == "folder":
    items = list(video_source(frames_dir, csv_dir))
    print("videos to process:", len(items))
    loader = DataLoader(FolderVideos(items), batch_size=None, num_workers=NUM_WORKERS, prefetch_factor=4)
    for split, name, x in loader:
        feats[split][name] = embed(x)
        n += 1
        if n % 250 == 0: report()
else:
    for split, name, pngs in video_source(frames_dir, csv_dir):
        feats[split][name] = embed(frames_to_uint8([load_rgb(b) for b in pngs]))
        n += 1
        if n % 250 == 0: report()
report()

os.makedirs(OUT_DIR, exist_ok=True)
for s in SPLITS:
    with open(os.path.join(OUT_DIR, f"resnet18_{s}.pkl"), "wb") as f:
        pickle.dump(feats[s], f, protocol=4)
    print(f"saved resnet18_{s}.pkl  ({len(feats[s])} videos)")

# ---- sanity check ----
for s in SPLITS:
    with open(os.path.join(OUT_DIR, f"resnet18_{s}.pkl"), "rb") as f:
        d = pickle.load(f)
    lens = [v.shape[0] for v in d.values()]
    k = next(iter(d))
    print(f"{s:5s}: {len(d):5d} videos | frames min/mean/max = {min(lens)}/{np.mean(lens):.0f}/{max(lens)} | "
          f"example {k}: {d[k].shape}, mean value {float(d[k].astype(np.float32).mean()):.3f}")
print(sorted(os.listdir(OUT_DIR)))
print("Expected about 7096 / 519 / 642 videos for train / dev / test.")