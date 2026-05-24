# Required: 
#           pip install ultralytics opencv-python numpy matplotlib
#           pip install git+https://github.com/JonathonLuiten/TrackEval.git

"""
Multi-Object Tracking Pipeline
===============================
- Detection & Tracking : YOLOv11n + BotSORT (ultralytics)
- Evaluation           : Official TrackEval Library
- Dataset              : MOT17-02-FRCNN
"""

import os
import glob
import warnings
import numpy as np
import cv2

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

DATASET_DIR   = "./MOT17-02-FRCNN"
IMG_DIR       = f"{DATASET_DIR}/img1"
GT_PATH       = f"{DATASET_DIR}/gt/gt.txt"
OUTPUT_VIDEO  = "./output_tracking.mp4"
RESULTS_TXT   = "./tracking_results.txt"
IDSW_LOG_TXT  = "./idsw_log.txt"
FPS           = 30
IMG_W, IMG_H  = 1920, 1080

# Tracking hyper-parameters
CONF_THRESH   = 0.15
IOU_THRESH    = 0.6
TRACKER_CFG   = "custom_botsort.yaml"

# Visualization settings
COLOR_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.6
FONT_THICK    = 2
PROGRESS_STEP = 50

# Evaluation settings
IOU_MATCH_THRESH = 0.5
GT_ACTIVE_COL    = 6   # column index for active_flag
GT_CLASS_COL     = 7   # column index for class_id
PEDESTRIAN_CLASS = 1


# ============================================================
# SECTION 2: TRACKING
# ============================================================

def get_color_for_id(track_id):
    """Return a consistent BGR color for a given track ID."""
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

def get_sorted_frame_paths(img_dir):
    """Return sorted list of .jpg frame paths in img_dir."""
    patterns = [os.path.join(img_dir, "*.jpg"), os.path.join(img_dir, "*.png")]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths = sorted(paths)
    return paths

def run_tracking(frame_paths):
    """Run YOLOv11n + BotSORT on each frame; return per-frame results list."""
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (IMG_W, IMG_H))

    all_results = []
    total_frames = len(frame_paths)

    for idx, fpath in enumerate(frame_paths):
        frame_id = idx + 1 

        frame = cv2.imread(fpath)
        if frame is None:
            continue

        results = model.track(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            tracker=TRACKER_CFG,
            persist=True,
            imgsz=1280,
            verbose=False,
            classes=[0],
        )

        det = results[0]
        if det.boxes is not None and det.boxes.id is not None:
            boxes_xyxy = det.boxes.xyxy.cpu().numpy()
            track_ids  = det.boxes.id.cpu().numpy().astype(int)
            confs      = det.boxes.conf.cpu().numpy()

            for i in range(len(track_ids)):
                tid = int(track_ids[i])
                if tid <= 0:
                    continue

                x1, y1, x2, y2 = boxes_xyxy[i]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                conf = float(confs[i])

                all_results.append((frame_id, tid, x, y, w, h, conf))

                color = get_color_for_id(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, FONT_THICK)
                label = f"ID{tid} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 8), FONT, FONT_SCALE, color, FONT_THICK)

        cv2.putText(frame, f"Frame: {frame_id}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        writer.write(frame)

        if frame_id % PROGRESS_STEP == 0 or frame_id == total_frames:
            print(f"Processing frame {frame_id}/{total_frames}...")

    writer.release()
    print(f"[INFO] Video saved to {OUTPUT_VIDEO}")
    return all_results


# ============================================================
# SECTION 3: SAVE TRACKING RESULTS
# ============================================================

def save_results(all_results):
    """Write tracking results in MOT Challenge format."""
    with open(RESULTS_TXT, "w") as f:
        for (frame_id, tid, x, y, w, h, conf) in all_results:
            f.write(f"{frame_id},{tid},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{conf:.4f},-1,-1,-1\n")
    print(f"[INFO] Tracking results saved to {RESULTS_TXT}")


# ============================================================
# SECTION 4: EVALUATION & ANALYSIS
# ============================================================

def bb_iou(boxA, boxB):
    """Calculate IoU between two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def log_id_switches(gt_path, res_path, out_log):
    """Compare GT and Tracking results frame-by-frame to log ID Switches."""
    gt_dict = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                fid, gid = int(parts[0]), int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                active, cls = int(parts[6]), int(parts[7])
                if active == 1 and cls == 1: # Only active pedestrians
                    if fid not in gt_dict: gt_dict[fid] = {}
                    gt_dict[fid][gid] = (x, y, w, h)

    trk_dict = {}
    with open(res_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                fid, tid = int(float(parts[0])), int(float(parts[1]))
                x, y, w, h = map(float, parts[2:6])
                if fid not in trk_dict: trk_dict[fid] = {}
                trk_dict[fid][tid] = (x, y, w, h)

    last_matched_trk = {}
    switches = []
    
    all_frames = sorted(list(set(gt_dict.keys()) | set(trk_dict.keys())))

    for f in all_frames:
        gts = gt_dict.get(f, {})
        trks = trk_dict.get(f, {})
        
        # Greedy matching by IoU
        ious = []
        for gid, gbox in gts.items():
            for tid, tbox in trks.items():
                iou = bb_iou(gbox, tbox)
                if iou >= IOU_MATCH_THRESH:
                    ious.append((iou, gid, tid))
        
        ious.sort(reverse=True, key=lambda x: x[0])
        matched_g, matched_t = set(), set()
        matches = {}
        
        for iou, gid, tid in ious:
            if gid not in matched_g and tid not in matched_t:
                matches[gid] = tid
                matched_g.add(gid)
                matched_t.add(tid)
                
        # Detect switches
        for gid, tid in matches.items():
            if gid in last_matched_trk:
                prev_tid = last_matched_trk[gid]
                if prev_tid != tid:
                    switches.append(f"Frame {f:04d} | GT ID: {gid:>3} switched from Trk ID: {prev_tid:>3} -> {tid:>3}")
            last_matched_trk[gid] = tid

    # Save to file
    with open(out_log, "w") as f:
        f.write(f"ID SWITCH LOG (Total: {len(switches)})\n")
        f.write("="*50 + "\n")
        for s in switches:
            f.write(s + "\n")
            
    print(f"[INFO] Analyzed {len(switches)} ID switches. Log saved to {out_log}")


def run_trackeval(gt_path, results_path, num_frames):
    """Run official TrackEval pipeline with dummy folder structure."""
    import tempfile
    import shutil
    try:
        if not hasattr(np, "float"):
            np.float = float
        import trackeval
    except ImportError:
        print("[ERROR] TrackEval is not installed.")
        return None

    seq_name = "MOT17-02"
    tracker_name = "YOLO_BotSORT"

    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_dir = os.path.join(tmp_dir, "gt", seq_name, "gt")
        os.makedirs(gt_dir)
        shutil.copy(gt_path, os.path.join(gt_dir, "gt.txt"))

        seqinfo_path = os.path.join(tmp_dir, "gt", seq_name, "seqinfo.ini")
        with open(seqinfo_path, "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={seq_name}\n")
            f.write(f"seqLength={num_frames}\n")

        tracker_dir = os.path.join(tmp_dir, "trackers", tracker_name, "data")
        os.makedirs(tracker_dir)
        shutil.copy(results_path, os.path.join(tracker_dir, f"{seq_name}.txt"))

        seqmap_dir = os.path.join(tmp_dir, "seqmaps")
        os.makedirs(seqmap_dir)
        seqmap_file = os.path.join(seqmap_dir, "MOT17-test.txt")
        with open(seqmap_file, "w") as f:
            f.write("name\n")
            f.write(f"{seq_name}\n")

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update({'DISPLAY_LESS_PROGRESS': False, 'PRINT_RESULTS': False, 'PRINT_CONFIG': False, 'TIME_PROGRESS': False})
        
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            'GT_FOLDER': os.path.join(tmp_dir, "gt"),
            'TRACKERS_FOLDER': os.path.join(tmp_dir, "trackers"),
            'SEQMAP_FILE': seqmap_file,
            'SEQMAP_FOLDER': seqmap_dir,
            'TRACKERS_TO_EVAL': [tracker_name],
            'CLASSES_TO_EVAL': ['pedestrian'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'test',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': False,
            'DO_PREPROC': False,
            'TRACKER_SUB_FOLDER': 'data',
            'OUTPUT_SUB_FOLDER': '',
            'SKIP_SPLIT_FOL': True
        })

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
            
        output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
        
        res = output_res
        try:
            res = res[list(res.keys())[0]][tracker_name][list(res[list(res.keys())[0]][tracker_name].keys())[0]]
            res = res.get('COMBINED_SEQ', res[list(res.keys())[0]])
        except Exception:
            return None
        
        metrics = {}
        metrics['HOTA'] = float(np.mean(res.get('HOTA', {}).get('HOTA', 0))) * 100
        metrics['DetA'] = float(np.mean(res.get('HOTA', {}).get('DetA', 0))) * 100
        metrics['AssA'] = float(np.mean(res.get('HOTA', {}).get('AssA', 0))) * 100
        
        metrics['MOTA']   = float(res.get('CLEAR', {}).get('MOTA', 0)) * 100
        metrics['MOTP']   = float(res.get('CLEAR', {}).get('MOTP', 0)) * 100
        metrics['CLR_Re'] = float(res.get('CLEAR', {}).get('CLR_Re', 0)) * 100
        metrics['CLR_Pr'] = float(res.get('CLEAR', {}).get('CLR_Pr', 0)) * 100
        
        metrics['FP']   = int(res.get('CLEAR', {}).get('CLR_FP', res.get('CLEAR', {}).get('FP', 0)))
        metrics['FN']   = int(res.get('CLEAR', {}).get('CLR_FN', res.get('CLEAR', {}).get('FN', 0)))
        metrics['IDSW'] = int(res.get('CLEAR', {}).get('IDSW', 0))
        
        metrics['IDF1'] = float(res.get('Identity', {}).get('IDF1', 0)) * 100
        metrics['IDR']  = float(res.get('Identity', {}).get('IDR', 0)) * 100
        metrics['IDP']  = float(res.get('Identity', {}).get('IDP', 0)) * 100
        
        return metrics


def print_and_plot_results(metrics):
    """Print results in ASCII table and plot separated charts."""
    if not metrics:
        return
        
    print()
    print("╔══════════════════════════════════════════╗")
    print("║            TRACKEVAL RESULTS             ║")
    print("╠══════════════════════════════════════════╣")
    print("║ PERCENTAGE METRICS                       ║")
    print(f"║  HOTA   : {metrics['HOTA']:>6.2f} %                        ║")
    print(f"║  DetA   : {metrics['DetA']:>6.2f} %                        ║")
    print(f"║  AssA   : {metrics['AssA']:>6.2f} %                        ║")
    print(f"║  MOTA   : {metrics['MOTA']:>6.2f} %                        ║")
    print(f"║  MOTP   : {metrics['MOTP']:>6.2f} %                        ║")
    print(f"║  CLR_Re : {metrics['CLR_Re']:>6.2f} %                        ║")
    print(f"║  CLR_Pr : {metrics['CLR_Pr']:>6.2f} %                        ║")
    print(f"║  IDF1   : {metrics['IDF1']:>6.2f} %                        ║")
    print(f"║  IDR    : {metrics['IDR']:>6.2f} %                        ║")
    print(f"║  IDP    : {metrics['IDP']:>6.2f} %                        ║")
    print("╠══════════════════════════════════════════╣")
    print("║ ERROR COUNTS                             ║")
    print(f"║  FP     : {int(metrics['FP']):>6d}                          ║")
    print(f"║  FN     : {int(metrics['FN']):>6d}                          ║")
    print(f"║  IDSW   : {int(metrics['IDSW']):>6d}                          ║")
    print("╚══════════════════════════════════════════╝")
    print()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Chart 1: Percentage Metrics
    plt.figure(figsize=(10, 7))
    pct_labels = ['HOTA', 'DetA', 'AssA', 'MOTA', 'MOTP', 'IDF1', 'IDR', 'IDP', 'CLR_Re', 'CLR_Pr']
    pct_values = [metrics[k] for k in pct_labels]
    y_pos = np.arange(len(pct_labels))
    plt.barh(y_pos, pct_values, align='center', color='skyblue', edgecolor='black')
    plt.yticks(y_pos, labels=pct_labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Score (%)')
    plt.title('Tracking Percentage Metrics')
    plt.xlim(0, 100)
    for i, v in enumerate(pct_values):
        plt.text(v + 1, i, f"{v:.1f}", va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_percentage_metrics.png', dpi=150)
    plt.close()

    # Chart 2: Absolute Error Counts
    plt.figure(figsize=(8, 6))
    cnt_labels = ['FP', 'FN', 'IDSW']
    cnt_values = [metrics[k] for k in cnt_labels]
    x_pos = np.arange(len(cnt_labels))
    plt.bar(x_pos, cnt_values, align='center', color=['lightcoral', 'lightsalmon', 'crimson'], edgecolor='black')
    plt.xticks(x_pos, labels=cnt_labels)
    plt.ylabel('Count')
    plt.title('Absolute Error Counts')
    for i, v in enumerate(cnt_values):
        plt.text(i, v + (max(cnt_values)*0.02 if max(cnt_values) > 0 else 0.5), str(int(v)), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_error_counts.png', dpi=150)
    plt.close()

    print("[INFO] Charts successfully separated and saved as:")
    print("       - chart_percentage_metrics.png")
    print("       - chart_error_counts.png")


# ============================================================
# MAIN
# ============================================================

def main():
    frame_paths = get_sorted_frame_paths(IMG_DIR)
    num_frames  = len(frame_paths)
    print(f"[INFO] Found {num_frames} frames in {IMG_DIR}")

    if num_frames == 0:
        print("[ERROR] No frames found. Check DATASET_DIR / IMG_DIR paths.")
        return

    # 1. Tracking
    all_results = run_tracking(frame_paths)
    save_results(all_results)

    # 2. Log ID Switches against GT
    if os.path.exists(GT_PATH):
        log_id_switches(GT_PATH, RESULTS_TXT, IDSW_LOG_TXT)

    # 3. TrackEval Metrics & Separate Charts
    metrics = run_trackeval(GT_PATH, RESULTS_TXT, num_frames)
    print_and_plot_results(metrics)

if __name__ == "__main__":
    main()