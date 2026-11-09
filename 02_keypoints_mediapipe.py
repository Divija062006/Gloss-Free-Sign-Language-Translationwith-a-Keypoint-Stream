# =============================================================================
# Step 2 of 5 - Hand, face and body keypoints with MediaPipe
# Project : Gloss-free sign language translation with a keypoint stream
# Team    : Claude5
#
# Extracts 75 landmarks per frame (11 upper-body, 2 x 21 hand, 22 face) with the
# MediaPipe Tasks API (Pose, Hand and Face Landmarkers). MediaPipe runs on every
# FRAME_STEP-th frame; frames in between are linearly interpolated.
# The run checks the first video before starting, saves progress periodically,
# stops and saves before Kaggle's 12-hour limit, reconnects if the download
# drops, and resumes from an earlier run's output attached as input.
#
# Kaggle  : Accelerator = None (CPU only), Internet = On
# Input   : none (the official archive is streamed)
# Output  : /kaggle/working/keypoints/kp_{train,dev,test}.pkl
#           {video name: float16 array (T, 75, 3) = pixel x, pixel y, detected}
# =============================================================================

# ---------------- settings ----------------
TAR_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
SPLITS = ["train", "dev", "test"]
OUT_DIR = "/kaggle/working/keypoints"
FRAME_STEP = 2            # run MediaPipe on every 2nd frame, interpolate the ones in between
TIME_BUDGET_HOURS = 10.5  # stop and save before Kaggle's 12-hour limit
SAVE_EVERY = 500          # videos between progress saves
VIDEO_TIMEOUT_S = 300     # a single video taking longer than this counts as failed

import os, subprocess, sys
os.environ["GLOG_minloglevel"] = "2"        # silence MediaPipe's per-video log spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe"], check=False)
_r = subprocess.run([sys.executable, "-c",
                     "import mediapipe as mp; from mediapipe.tasks.python import vision; "
                     "print(mp.__version__, hasattr(vision, 'PoseLandmarker'), hasattr(vision, 'HandLandmarker'), hasattr(vision, 'FaceLandmarker'))"],
                    capture_output=True, text=True)
print("mediapipe check:", _r.stdout.strip(), _r.stderr.strip()[-300:])
assert _r.stdout.strip().endswith("True True True"), "MediaPipe Tasks API not available; check the installed mediapipe version."

import re, time, glob, pickle, tarfile, socket, urllib.request, multiprocessing as mproc
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np, cv2

# ---------------- MediaPipe model files (downloaded once, kept out of the output folder) ----------------
MODEL_DIR = "/tmp/mp_models"
MODELS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
os.makedirs(MODEL_DIR, exist_ok=True)
for k, url in MODELS.items():
    p = os.path.join(MODEL_DIR, k + ".task")
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    print(f"model {k}: {os.path.getsize(p)/1e6:.1f} MB")

# ---------------- selected landmarks: 75 per frame, in the layout the training scripts expect ----------------
POSE_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]   # nose, eyes, ears, shoulders, elbows, wrists
FACE_IDX = [33, 133, 159, 145, 362, 263, 386, 374, 70, 105, 107, 336, 334, 300, 61, 291, 0, 17, 13, 14, 78, 308]
N_POINTS = 75   # 0-10 body, 11-31 first hand, 32-52 second hand, 53-74 face

def _fill(dst, start, lms, idx, w, h, use_visibility=False):
    for k, i in enumerate(idx):
        p = lms[i]
        vis = getattr(p, "visibility", None)
        ok = (vis is None or vis > 0.3) if use_visibility else True
        dst[start + k] = (p.x * w, p.y * h, 1.0 if ok else 0.0)

def load_rgb(png_bytes):
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _init_worker():
    cv2.setNumThreads(1)

def process_video(task):
    # task = (split, name, [png bytes]) -> (split, name, T x 75 x 3 float16 in pixel coordinates)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    split, name, frames = task
    T = len(frames)
    out = np.zeros((T, N_POINTS, 3), np.float32)
    VIDEO = vision.RunningMode.VIDEO
    opt = lambda k: BaseOptions(model_asset_path=os.path.join(MODEL_DIR, k + ".task"))
    with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=opt("pose"), running_mode=VIDEO, num_poses=1)) as pose, \
         vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=opt("hand"), running_mode=VIDEO, num_hands=2)) as hands, \
         vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=opt("face"), running_mode=VIDEO, num_faces=1)) as face:
        done = [t for t in range(T) if t % FRAME_STEP == 0 or t == T - 1]
        for t in done:
            rgb = load_rgb(frames[t])
            h, w = rgb.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = t * 40                                  # 25 fps, strictly increasing
            r = pose.detect_for_video(img, ts)
            if r.pose_landmarks:
                _fill(out[t], 0, r.pose_landmarks[0], POSE_IDX, w, h, use_visibility=True)
            r = hands.detect_for_video(img, ts)
            used = set()
            for lm, hd in zip(r.hand_landmarks, r.handedness):
                slot = 11 if hd[0].category_name == "Left" else 32
                if slot in used:
                    slot = 32 if slot == 11 else 11
                used.add(slot)
                _fill(out[t], slot, lm, range(21), w, h)
            r = face.detect_for_video(img, ts)
            if r.face_landmarks:
                _fill(out[t], 53, r.face_landmarks[0], FACE_IDX, w, h)
    for a, b in zip(done[:-1], done[1:]):           # fill skipped frames from their two neighbours
        for t in range(a + 1, b):
            wgt = (t - a) / (b - a)
            both = np.minimum(out[a, :, 2], out[b, :, 2])
            out[t, :, :2] = ((1 - wgt) * out[a, :, :2] + wgt * out[b, :, :2]) * both[:, None]
            out[t, :, 2] = both
    return split, name, out.astype(np.float16)

# ---------------- streaming the official archive, reconnecting if it drops ----------------
FRAME_RE = re.compile(r"fullFrame-210x260px/(train|dev|test)/([^/]+)/[^/]+\.png$")

def iter_tar_videos(url):
    src = urllib.request.urlopen(url, timeout=180) if url.startswith("http") else open(url, "rb")
    cur_key, cur = None, []
    with tarfile.open(fileobj=src, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            if m.name.endswith(".corpus.csv"):
                with open(os.path.join(OUT_DIR, os.path.basename(m.name)), "wb") as f:
                    f.write(tar.extractfile(m).read())
                continue
            mt = FRAME_RE.search(m.name)
            if not mt or mt.group(1) not in SPLITS:
                continue
            key = (mt.group(1), mt.group(2))
            if key != cur_key:
                if cur:
                    yield cur_key[0], cur_key[1], [b for _, b in sorted(cur)]
                cur_key, cur = key, []
            cur.append((m.name, tar.extractfile(m).read()))
        if cur:
            yield cur_key[0], cur_key[1], [b for _, b in sorted(cur)]

def robust_stream(is_done, max_retries=6):
    for attempt in range(max_retries + 1):
        try:
            for task in iter_tar_videos(TAR_URL):
                if not is_done(task[0], task[1]):
                    yield task
            return
        except Exception as e:
            print(f"  ! download problem ({type(e).__name__}: {e}); reconnecting, attempt {attempt + 1}", flush=True)
            time.sleep(30)
    raise RuntimeError("download kept failing")

# ---------------- resume from any earlier output attached with Add Input ----------------
os.makedirs(OUT_DIR, exist_ok=True)
kps = {s: {} for s in SPLITS}
for s in SPLITS:
    for p in glob.glob(f"/kaggle/input/**/kp_{s}*.pkl", recursive=True):
        with open(p, "rb") as f:
            kps[s].update(pickle.load(f))
print("already done from earlier runs:", {s: len(kps[s]) for s in SPLITS})

def save(final=False):
    for s in SPLITS:
        with open(os.path.join(OUT_DIR, f"kp_{s}.pkl"), "wb") as f:
            pickle.dump(kps[s], f, protocol=4)
    if final:
        print("saved:", {s: len(kps[s]) for s in SPLITS}, flush=True)

# ---------------- main loop: download and process at the same time ----------------
WORKERS = os.cpu_count() or 4
MAX_IN_FLIGHT = WORKERS * 3
t0, n, failed, since_save = time.time(), 0, [], 0
ctx = mproc.get_context("fork")
stream = robust_stream(lambda s, nm: nm in kps[s])
print(f"workers: {WORKERS} | streaming from {TAR_URL}", flush=True)

def collect(fut, task_of, block_timeout=None):
    global n, since_save
    split, name = task_of.pop(fut)
    try:
        s, nm, arr = fut.result(timeout=block_timeout)
        kps[s][nm] = arr
    except Exception as e:
        failed.append((split, name, f"{type(e).__name__}: {e}"))
        if len(failed) <= 5:
            print(f"  ! failed on {split}/{name}: {type(e).__name__}: {e}", flush=True)
    n += 1; since_save += 1
    if n % 250 == 0:
        el = time.time() - t0
        print(f"  {n} videos | {el/60:.1f} min | {el/n:.2f} s/video | failed so far {len(failed)}", flush=True)

with ProcessPoolExecutor(WORKERS, mp_context=ctx, initializer=_init_worker) as ex:
    # 1) fail fast: run the very first video on its own and check the result
    first = next(stream, None)
    if first is None:
        print("nothing left to process")
    else:
        s, nm, arr = ex.submit(process_video, first).result(timeout=VIDEO_TIMEOUT_S)
        kps[s][nm] = arr; n += 1
        det = arr[..., 2].astype(np.float32)
        print(f"FIRST VIDEO OK: {s}/{nm} {arr.shape} | body {det[:, 5:7].min(1).mean():.0%} | "
              f"hand A {det[:, 11].mean():.0%} | hand B {det[:, 32].mean():.0%} | face {det[:, 53].mean():.0%}", flush=True)
        assert det.sum() > 0, "MediaPipe detected no landmarks in the first video; check the input frames."

    # 2) everything else, with a bounded number of videos in flight
    task_of, out_of_time = {}, False
    for task in stream:
        task_of[ex.submit(process_video, task)] = (task[0], task[1])
        while len(task_of) >= MAX_IN_FLIGHT:
            done, _ = wait(list(task_of), timeout=VIDEO_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done:
                raise RuntimeError("worker processes stopped responding; aborting.")
            for f in done:
                collect(f, task_of)
        if since_save >= SAVE_EVERY:
            save(); since_save = 0
        if time.time() - t0 > TIME_BUDGET_HOURS * 3600:
            out_of_time = True
            print("TIME BUDGET REACHED - saving what is done so far", flush=True)
            break
    for f in list(task_of):
        collect(f, task_of, block_timeout=VIDEO_TIMEOUT_S)

save(final=True)
total = {"train": 7096, "dev": 519, "test": 642}
complete = all(len(kps[s]) >= total[s] for s in SPLITS)
print(f"\nfinished in {(time.time() - t0)/60:.0f} min | failed videos: {len(failed)}")
if failed:
    print("first failures:", failed[:3])
for s in SPLITS:
    allk = np.concatenate([a[..., 2].astype(np.float32) for a in kps[s].values()])
    print(f"{s:5s}  videos={len(kps[s]):5d}/{total[s]} | body {allk[:, 5:7].min(1).mean():.1%} | "
          f"hand A {allk[:, 11].mean():.1%} | hand B {allk[:, 32].mean():.1%} | face {allk[:, 53].mean():.1%}")
print("\nALL VIDEOS DONE" if complete else "\nINCOMPLETE - run again with this notebook's output added as input to continue")
