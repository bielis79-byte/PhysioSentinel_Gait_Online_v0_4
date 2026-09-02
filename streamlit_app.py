
import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile
import uuid
import json
import math
import traceback
import re
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import toml
from scipy.signal import find_peaks

APP_VERSION = "0.4.0-online"

st.set_page_config(
    page_title="PhysioSentinel Gait",
    page_icon="🚶",
    layout="wide",
)

st.title("PhysioSentinel Gait")
st.caption(f"Versión {APP_VERSION} · análisis 2D online + ángulos dinámicos · sesión temporal")
st.info("🌐 Versión online de prueba: los vídeos y resultados se guardan solo de forma temporal en el servidor. No uses datos identificativos del paciente en esta fase.")

BASE_DIR = Path(tempfile.gettempdir()) / "physiosentinel_gait_online"
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

HALPE26 = {
    "Nose": 0,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "Head": 17,
    "Neck": 18,
    "Hip": 19,
    "LBigToe": 20,
    "RBigToe": 21,
    "LSmallToe": 22,
    "RSmallToe": 23,
    "LHeel": 24,
    "RHeel": 25,
}

LOWER_BODY_NAMES = [
    "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle",
    "LHeel","RHeel","LBigToe","RBigToe"
]

ANGLE_BODY_NAMES = [
    "LShoulder","RShoulder","LElbow","RElbow",
    "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle",
    "LBigToe","RBigToe"
]

SKELETON_EDGES = [
    ("LShoulder","RShoulder"), ("LShoulder","LHip"), ("RShoulder","RHip"),
    ("LHip","RHip"), ("LShoulder","LElbow"), ("RShoulder","RElbow"),
    ("LElbow","LWrist"), ("RElbow","RWrist"),
    ("LHip","LKnee"), ("RHip","RKnee"), ("LKnee","LAnkle"), ("RKnee","RAnkle"),
    ("LAnkle","LHeel"), ("RAnkle","RHeel"), ("LAnkle","LBigToe"), ("RAnkle","RBigToe"),
    ("LHeel","LBigToe"), ("RHeel","RBigToe"),
]


# -------------------------------------------------
# Persistencia online
# -------------------------------------------------
# Esta versión no conserva una base de datos clínica.
# Todo se mantiene en la sesión temporal del servidor Streamlit.

# -------------------------------------------------
# Vídeo y sesión
# -------------------------------------------------
def safe_name(text):
    text = (text or "sesion").strip()
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)

def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": (frames / fps) if fps > 0 else 0,
        "orientation": "Vertical" if height > width else "Horizontal",
    }

def save_upload(uploaded, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

def create_session(patient, record):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = SESSIONS_DIR / f"{safe_name(patient)}_{safe_name(record)}_{stamp}_{uuid.uuid4().hex[:8]}"
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    return folder

def prepare_config(session_dir):
    """Crea una configuración mínima; Pose2Sim completa el resto con sus valores por defecto."""
    cfg = {
        "project": {
            "project_dir": str(session_dir),
            "multi_person": False,
            "participant_height": "auto",
            "participant_mass": 70.0,
            "frame_rate": "auto",
            "frame_range": "auto",
        },
        "pose": {
            "pose_model": "Body_with_feet",
            "mode": "balanced",
            "det_frequency": 4,
            "device": "auto",
            "backend": "auto",
            "display_detection": False,
            "overwrite_pose": True,
            "save_video": "to_video",
            "output_format": "openpose",
            "tracking_mode": "sports2d",
        },
    }
    # Se guarda también para diagnóstico/descarga, pero el motor recibe el diccionario
    # para que Pose2Sim fusione automáticamente esta configuración con sus defaults.
    config_path = Path(session_dir) / "Config.toml"
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    return cfg

def run_pose2sim(config):
    from Pose2Sim import Pose2Sim
    Pose2Sim.poseEstimation(config)

def find_pose_json_dir(session_dir):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    candidates = sorted([p for p in pose_dir.rglob("*_json") if p.is_dir()])
    if candidates:
        return candidates[0]
    return None

def find_annotated_video(session_dir):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(pose_dir.rglob(ext))
    videos = [p for p in videos if p.is_file()]
    return sorted(videos)[0] if videos else None


def get_ffmpeg_exe():
    """Localiza FFmpeg: imageio-ffmpeg o PATH del sistema."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")

def transcode_for_web(src, dst):
    """Convierte a H.264/yuv420p/faststart para Chrome y Streamlit."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return None, "No existe el vídeo de origen."
    exe = get_ffmpeg_exe()
    if not exe:
        return None, (
            "No encuentro FFmpeg. Instala imageio-ffmpeg en el entorno Pose2Sim "
            "o FFmpeg en Windows para generar el vídeo compatible con navegador."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size < 1024:
            return None, (proc.stderr or "FFmpeg no pudo convertir el vídeo.")[-1200:]
        return dst, None
    except Exception as e:
        return None, str(e)

def ensure_web_video(src, suffix="_web"):
    src = Path(src)
    dst = src.with_name(src.stem + suffix + ".mp4")
    if dst.exists() and dst.stat().st_size > 1024 and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst, None
    return transcode_for_web(src, dst)

def show_browser_video(path, caption=None):
    path = Path(path)
    web_path, err = ensure_web_video(path)
    if caption:
        st.caption(caption)
    if web_path:
        st.video(str(web_path))
        return web_path
    st.warning("No pude crear la copia web H.264. Muestro el archivo original como alternativa.")
    if err:
        with st.expander("Detalle de conversión de vídeo"):
            st.code(err)
    st.video(str(path))
    return None


# -------------------------------------------------
# Lectura HALPE26
# -------------------------------------------------
def load_pose_dataframe(json_dir):
    files = sorted(json_dir.glob("*.json"))
    rows = []
    for fallback_i, path in enumerate(files):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                continue
            pts = people[0].get("pose_keypoints_2d", [])
            if len(pts) < 26 * 3:
                continue
            nums = re.findall(r"(\d+)", path.stem)
            frame_no = int(nums[-1]) if nums else fallback_i
            row = {"frame": frame_no}
            for name, idx in HALPE26.items():
                base = idx * 3
                row[f"{name}_x"] = float(pts[base])
                row[f"{name}_y"] = float(pts[base + 1])
                row[f"{name}_score"] = float(pts[base + 2])
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("frame").drop_duplicates("frame").reset_index(drop=True)

def point_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    c = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def joint_angle_from_row(r, a, b, c, transform="raw", min_score=0.35):
    for name in (a, b, c):
        sc = r.get(f"{name}_score", np.nan)
        if not np.isfinite(sc) or sc < min_score:
            return np.nan
    ang = point_angle(
        r[f"{a}_x"], r[f"{a}_y"], r[f"{b}_x"], r[f"{b}_y"], r[f"{c}_x"], r[f"{c}_y"]
    )
    if not np.isfinite(ang):
        return np.nan
    return 180.0 - ang if transform == "flex" else ang

def add_projected_angles(seg):
    """Ángulos 2D proyectados por frame; no son cinemática anatómica 3D."""
    for side in ("L", "R"):
        seg[f"{side}_hip_angle"] = [joint_angle_from_row(r, f"{side}Shoulder", f"{side}Hip", f"{side}Knee", "flex") for _, r in seg.iterrows()]
        seg[f"{side}_knee_flex"] = [joint_angle_from_row(r, f"{side}Hip", f"{side}Knee", f"{side}Ankle", "flex") for _, r in seg.iterrows()]
        seg[f"{side}_ankle_angle"] = [joint_angle_from_row(r, f"{side}Knee", f"{side}Ankle", f"{side}BigToe", "raw") for _, r in seg.iterrows()]
        seg[f"{side}_shoulder_angle"] = [joint_angle_from_row(r, f"{side}Hip", f"{side}Shoulder", f"{side}Elbow", "flex") for _, r in seg.iterrows()]
    return seg

def robust_p5_p95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    p5, p95 = np.percentile(arr, [5, 95])
    return float(p5), float(p95), float(p95 - p5)

def angle_metric_block(metrics, seg, joint_key, label, note):
    vals = {}
    for side, side_label in (("L", "izquierda"), ("R", "derecha")):
        p5, p95, rom = robust_p5_p95(seg[f"{side}_{joint_key}"])
        vals[side] = {"p95": p95, "rom": rom}
        metrics.extend([
            {"key":f"{joint_key}_{side.lower()}_p95","label":f"{label} {side_label} 2D (P95)","value":p95,"unit":"°","quality":"2D proyectado","notes":note+" Percentil 95 robusto del segmento."},
            {"key":f"{joint_key}_{side.lower()}_rom","label":f"ROM {label.lower()} {side_label} 2D","value":rom,"unit":"°","quality":"2D proyectado","notes":note+" ROM robusto=P95-P5; no equivale a ROM anatómico 3D."},
        ])
    diff = abs(vals["L"]["p95"]-vals["R"]["p95"]) if np.isfinite(vals["L"]["p95"]) and np.isfinite(vals["R"]["p95"]) else np.nan
    metrics.append({"key":f"{joint_key}_p95_diff","label":f"Diferencia D/I {label.lower()} 2D (P95)","value":diff,"unit":"°","quality":"2D proyectado","notes":note+" Diferencia absoluta D/I; interpretar según la vista."})

def rolling_smooth(arr, window=7):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()

def zero_crossings(signal):
    s = np.asarray(signal, dtype=float)
    finite = np.isfinite(s)
    idx = []
    for i in range(1, len(s)):
        if not (finite[i-1] and finite[i]):
            continue
        if (s[i-1] <= 0 < s[i]) or (s[i-1] >= 0 > s[i]):
            idx.append(i)
    return np.array(idx, dtype=int)

def quality_label(score):
    if score >= 0.80:
        return "Alta"
    if score >= 0.65:
        return "Moderada"
    return "Baja"


# -------------------------------------------------
# Vídeo con esqueleto + ángulos dinámicos
# -------------------------------------------------
def _pt_from_row(r, name, min_score=0.25):
    try:
        sc = float(r[f"{name}_score"]); x = float(r[f"{name}_x"]); y = float(r[f"{name}_y"])
        if sc < min_score or not np.isfinite(x) or not np.isfinite(y):
            return None
        return int(round(x)), int(round(y))
    except Exception:
        return None

def _draw_label(frame, xy, text, color):
    if xy is None:
        return
    x, y = xy
    cv2.circle(frame, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

def generate_angle_video(source_video, df, session_dir, force=False):
    source_video = Path(source_video)
    pose_dir = Path(session_dir) / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    raw_out = pose_dir / "cam01_angles_raw.mp4"
    web_out = pose_dir / "cam01_angles_web.mp4"
    if web_out.exists() and web_out.stat().st_size > 1024 and not force:
        return web_out, None
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        return None, "No se pudo abrir el vídeo original para dibujar ángulos."
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release(); return None, "No se pudo crear el vídeo temporal con ángulos."
    lookup = df.set_index("frame", drop=False).to_dict("index")
    left_color=(60,220,60); right_color=(255,170,40); neutral=(255,255,255)
    frame_no=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        r = lookup.get(frame_no)
        if r is not None:
            for a,b in SKELETON_EDGES:
                pa=_pt_from_row(r,a); pb=_pt_from_row(r,b)
                if pa and pb:
                    col = left_color if a.startswith("L") and b.startswith("L") else right_color if a.startswith("R") and b.startswith("R") else neutral
                    cv2.line(frame, pa, pb, col, 2, cv2.LINE_AA)
            for side, side_txt, color in (("L","I",left_color),("R","D",right_color)):
                hip=joint_angle_from_row(r,f"{side}Shoulder",f"{side}Hip",f"{side}Knee","flex")
                knee=joint_angle_from_row(r,f"{side}Hip",f"{side}Knee",f"{side}Ankle","flex")
                ankle=joint_angle_from_row(r,f"{side}Knee",f"{side}Ankle",f"{side}BigToe","raw")
                shoulder=joint_angle_from_row(r,f"{side}Hip",f"{side}Shoulder",f"{side}Elbow","flex")
                if np.isfinite(hip): _draw_label(frame,_pt_from_row(r,f"{side}Hip"),f"Cad {side_txt}: {hip:.0f} deg",color)
                if np.isfinite(knee): _draw_label(frame,_pt_from_row(r,f"{side}Knee"),f"Rod {side_txt}: {knee:.0f} deg",color)
                if np.isfinite(ankle): _draw_label(frame,_pt_from_row(r,f"{side}Ankle"),f"Tob {side_txt}: {ankle:.0f} deg",color)
                if np.isfinite(shoulder): _draw_label(frame,_pt_from_row(r,f"{side}Shoulder"),f"Hom {side_txt}: {shoulder:.0f} deg",color)
            cv2.rectangle(frame,(8,8),(min(width-8,430),42),(0,0,0),-1)
            cv2.putText(frame,"Angulos 2D proyectados - no 3D anatomico",(16,31),cv2.FONT_HERSHEY_SIMPLEX,0.52,neutral,1,cv2.LINE_AA)
        writer.write(frame); frame_no += 1
    cap.release(); writer.release()
    converted, err = transcode_for_web(raw_out, web_out)
    if converted:
        try: raw_out.unlink(missing_ok=True)
        except Exception: pass
        return converted, None
    return raw_out if raw_out.exists() else None, err

# -------------------------------------------------
# Métricas 2D
# -------------------------------------------------
def compute_metrics(df, fps, start_frame, end_frame):
    seg = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].copy()
    if len(seg) < max(30, int(fps * 2)):
        raise ValueError("El segmento seleccionado es demasiado corto.")
    score_cols=[f"{name}_score" for name in LOWER_BODY_NAMES if f"{name}_score" in seg]
    mean_tracking=float(seg[score_cols].mean(axis=1).mean())
    good_frames=float((seg[score_cols].min(axis=1)>=0.5).mean()*100.0)
    tracking_quality=quality_label(mean_tracking)
    seg=add_projected_angles(seg)
    metrics=[
        {"key":"tracking_mean","label":"Confianza media del tracking","value":mean_tracking,"unit":"","quality":tracking_quality,"notes":"Media de puntuaciones HALPE26 del tren inferior."},
        {"key":"good_frames_pct","label":"Frames con tracking ≥0,50","value":good_frames,"unit":"%","quality":tracking_quality,"notes":"Todos los puntos principales del tren inferior ≥0,50."},
    ]
    angle_metric_block(metrics,seg,"hip_angle","Cadera","Proxy angular 2D hombro-cadera-rodilla; depende de la vista.")
    angle_metric_block(metrics,seg,"knee_flex","Flexión rodilla","Flexión geométrica 2D cadera-rodilla-tobillo.")
    angle_metric_block(metrics,seg,"ankle_angle","Ángulo tobillo","Ángulo geométrico 2D rodilla-tobillo-antepié; no es dorsiflexión anatómica aislada.")
    angle_metric_block(metrics,seg,"shoulder_angle","Hombro","Proxy angular 2D tronco-hombro-brazo; depende de la vista.")
    l_knee_p95=metric_value(metrics,"knee_flex_l_p95"); r_knee_p95=metric_value(metrics,"knee_flex_r_p95"); knee_diff=metric_value(metrics,"knee_flex_p95_diff")
    metrics.extend([
        {"key":"knee_flex_left_p95","label":"Flexión rodilla izquierda 2D (P95)","value":l_knee_p95,"unit":"°","quality":tracking_quality,"notes":"Clave compatible con v0.3; ángulo 2D proyectado."},
        {"key":"knee_flex_right_p95","label":"Flexión rodilla derecha 2D (P95)","value":r_knee_p95,"unit":"°","quality":tracking_quality,"notes":"Clave compatible con v0.3; ángulo 2D proyectado."},
        {"key":"knee_flex_diff","label":"Diferencia flexión rodillas 2D","value":knee_diff,"unit":"°","quality":tracking_quality,"notes":"Clave compatible con v0.3; diferencia absoluta P95 D/I."},
    ])
    ly=(seg["LAnkle_y"].to_numpy()+seg["LHeel_y"].to_numpy()+seg["LBigToe_y"].to_numpy())/3.0
    ry=(seg["RAnkle_y"].to_numpy()+seg["RHeel_y"].to_numpy()+seg["RBigToe_y"].to_numpy())/3.0
    diff=rolling_smooth(ry-ly,7); crossings=zero_crossings(diff)
    if len(crossings)>1:
        kept=[crossings[0]]; min_gap=max(1,int(round(0.25*fps)))
        for c in crossings[1:]:
            if c-kept[-1]>=min_gap: kept.append(c)
        crossings=np.array(kept,dtype=int)
    intervals=np.diff(crossings)/fps if len(crossings)>=2 else np.array([])
    cadence=mean_alt=cv_alt=asym=np.nan
    if len(intervals)>=3:
        mean_alt=float(np.mean(intervals)); cadence=float(60.0/mean_alt); cv_alt=float(np.std(intervals,ddof=1)/np.mean(intervals)*100.0)
        type1,type2=intervals[0::2],intervals[1::2]
        if len(type1)>=2 and len(type2)>=2:
            m1,m2=float(np.mean(type1)),float(np.mean(type2)); asym=float(abs(m1-m2)/((m1+m2)/2.0)*100.0)
    metrics.extend([
        {"key":"cadence_exp","label":"Cadencia estimada","value":cadence,"unit":"pasos/min","quality":"Experimental" if np.isfinite(cadence) else "No calculable","notes":"Derivada de alternancias relativas D/I. No equivale todavía a heel-strike validado."},
        {"key":"alternation_interval","label":"Intervalo medio de alternancia","value":mean_alt,"unit":"s","quality":"Experimental" if np.isfinite(mean_alt) else "No calculable","notes":"Intervalo entre cruces consecutivos de la señal distal D-I."},
        {"key":"regularity_cv","label":"Variabilidad temporal de alternancia","value":cv_alt,"unit":"%","quality":"Experimental" if np.isfinite(cv_alt) else "No calculable","notes":"CV de los intervalos de alternancia."},
        {"key":"temporal_asymmetry_exp","label":"Asimetría temporal experimental","value":asym,"unit":"%","quality":"Experimental" if np.isfinite(asym) else "No calculable","notes":"Alternancias impares vs pares; todavía no etiquetadas anatómicamente como paso D/I."},
    ])
    chart=pd.DataFrame({
        "frame":seg["frame"].to_numpy(),"time_s":seg["frame"].to_numpy()/fps,
        "Cadera izquierda":seg["L_hip_angle"].to_numpy(),"Cadera derecha":seg["R_hip_angle"].to_numpy(),
        "Rodilla izquierda":seg["L_knee_flex"].to_numpy(),"Rodilla derecha":seg["R_knee_flex"].to_numpy(),
        "Tobillo izquierdo":seg["L_ankle_angle"].to_numpy(),"Tobillo derecho":seg["R_ankle_angle"].to_numpy(),
        "Hombro izquierdo":seg["L_shoulder_angle"].to_numpy(),"Hombro derecho":seg["R_shoulder_angle"].to_numpy(),
        "Alternancia D-I":diff,
    })
    return metrics,chart,crossings,seg


def metric_value(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m["value"]
    return None

def fmt(value, decimals=1):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


# -------------------------------------------------
# App
# -------------------------------------------------
with st.sidebar:
    st.header("Sesión")
    patient = st.text_input("Código temporal", value="Prueba")
    record = st.text_input("Nombre del registro", value="Marcha")

    st.divider()
    mode = st.radio(
        "Modo de análisis",
        ["1 cámara · 2D", "2 cámaras · frontal/posterior + lateral"],
        index=0,
    )

    if mode.startswith("1 cámara"):
        view = st.radio("Vista", ["Frontal/posterior", "Lateral"], index=0)
    else:
        view = None

    st.divider()
    st.markdown("**Motor interno**")
    st.write("Pose2Sim + RTMPose")
    st.write("Body_with_feet / HALPE26")
    st.caption("Procesamiento en el servidor Streamlit. Datos no persistentes.")

tabs = st.tabs([
    "1 · Vídeos",
    "2 · Calidad",
    "3 · Analizar marcha",
    "4 · Resultados 2D",
    "5 · 3D futuro",
])

with tabs[0]:
    st.subheader("Carga de vídeos")

    if mode.startswith("1 cámara"):
        st.info(f"Modo actual: **1 cámara · {view}**")
        single = st.file_uploader(
            f"Vídeo {view.lower()}",
            type=["mp4","mov","avi","mkv"],
            key="single_video",
        )
        if single:
            st.caption(f"Seleccionado: {single.name}. La app generará una copia H.264 compatible al crear la sesión.")

        if st.button("Crear sesión temporal", type="primary", use_container_width=True):
            if single is None:
                st.error("Selecciona primero un vídeo.")
            else:
                session = create_session(patient, record)
                suffix = Path(single.name).suffix.lower() or ".mp4"
                p = session / "videos" / f"cam01{suffix}"
                save_upload(single, p)
                meta = video_metadata(p)

                st.session_state.session_dir = str(session)
                st.session_state.video1 = str(p)
                st.session_state.meta1 = meta
                st.session_state.mode = mode
                st.session_state.view = view
                st.session_state.pose_done = False
                st.session_state.metrics_done = False

                st.success("Sesión temporal creada correctamente.")
                with st.spinner("Preparando vídeo compatible con navegador..."):
                    web_preview, web_err = ensure_web_video(p, suffix="_input_web")
                if web_preview:
                    st.session_state.video1_web = str(web_preview)
                    st.video(str(web_preview))
                elif web_err:
                    st.warning("El vídeo temporal se cargó, pero no se pudo crear todavía la copia web H.264.")

    else:
        st.info("Modo actual: **2 cámaras · frontal/posterior + lateral**")
        c1, c2 = st.columns(2)
        with c1:
            front = st.file_uploader(
                "Vídeo frontal/posterior",
                type=["mp4","mov","avi","mkv"],
                key="front_video",
            )
            if front:
                st.caption(f"Seleccionado: {front.name}")
        with c2:
            side = st.file_uploader(
                "Vídeo lateral",
                type=["mp4","mov","avi","mkv"],
                key="side_video",
            )
            if side:
                st.caption(f"Seleccionado: {side.name}")

        st.warning(
            "En v0.4 online se procesa la pose de ambas cámaras, "
            "pero las métricas integradas de dos vistas se añadirán después."
        )

        if st.button("Crear sesión temporal con dos vídeos", type="primary", use_container_width=True):
            if front is None or side is None:
                st.error("Selecciona los dos vídeos.")
            else:
                session = create_session(patient, record)
                s1 = Path(front.name).suffix.lower() or ".mp4"
                s2 = Path(side.name).suffix.lower() or ".mp4"
                p1 = session / "videos" / f"cam01{s1}"
                p2 = session / "videos" / f"cam02{s2}"
                save_upload(front, p1)
                save_upload(side, p2)
                meta1 = video_metadata(p1)
                meta2 = video_metadata(p2)

                st.session_state.session_dir = str(session)
                st.session_state.video1 = str(p1)
                st.session_state.video2 = str(p2)
                st.session_state.meta1 = meta1
                st.session_state.meta2 = meta2
                st.session_state.mode = mode
                st.session_state.view = "Frontal+Lateral"
                st.session_state.pose_done = False
                st.session_state.metrics_done = False

                st.success("Sesión temporal creada correctamente.")

with tabs[1]:
    st.subheader("Control de calidad de adquisición")
    if "session_dir" not in st.session_state:
        st.warning("Primero crea una sesión.")
    else:
        metas = [("Cámara 1", st.session_state.get("meta1"))]
        if st.session_state.get("mode", "").startswith("2 cámaras"):
            metas.append(("Cámara 2", st.session_state.get("meta2")))

        for title, meta in metas:
            st.markdown(f"### {title}")
            if not meta:
                st.error("No se pudo leer el vídeo.")
                continue

            a,b,c,d = st.columns(4)
            a.metric("FPS", f'{meta["fps"]:.1f}')
            b.metric("Duración", f'{meta["duration"]:.1f} s')
            c.metric("Resolución", f'{meta["width"]} × {meta["height"]}')
            d.metric("Orientación", meta["orientation"])

            if meta["fps"] >= 29:
                st.success("Frecuencia de imagen adecuada para esta fase.")
            else:
                st.warning("FPS bajo: interpretar parámetros temporales con cautela.")

with tabs[2]:
    st.subheader("Analizar marcha")

    if "session_dir" not in st.session_state:
        st.warning("Primero crea una sesión.")
    else:
        session_dir = Path(st.session_state.session_dir)

        st.write(f"Sesión: `{session_dir}`")
        st.write("Motor: **Pose2Sim + RTMPose · Body_with_feet (HALPE26)**")

        st.success("Configuración Pose2Sim online preparada (HALPE26 / RTMPose).")
        st.caption("La primera ejecución puede tardar más porque el servidor puede necesitar descargar el modelo de pose.")

        if st.button("▶ Analizar marcha", type="primary", use_container_width=True):
            try:
                with st.spinner("Detectando pose con Pose2Sim/RTMPose..."):
                    cfg = prepare_config(session_dir)
                    run_pose2sim(cfg)

                json_dir = find_pose_json_dir(session_dir)
                if json_dir is None:
                    raise RuntimeError("Pose2Sim terminó, pero no encuentro la carpeta de JSON.")

                df = load_pose_dataframe(json_dir)
                if df.empty:
                    raise RuntimeError("No se pudieron leer keypoints HALPE26.")

                st.session_state.pose_done = True
                st.session_state.pose_json_dir = str(json_dir)
                st.session_state.pose_frames = len(df)

                angle_video = None
                angle_err = None
                source_video = st.session_state.get("video1")
                if source_video:
                    with st.spinner("Generando vídeo con esqueleto y ángulos 2D..."):
                        angle_video, angle_err = generate_angle_video(source_video, df, session_dir)
                    if angle_video:
                        st.session_state.angle_video = str(angle_video)

                st.success(f"Análisis de pose completado: {len(df)} frames útiles.")
                if angle_video:
                    st.success("Vídeo con ángulos dinámicos preparado.")
                elif angle_err:
                    st.warning("La pose está calculada, pero no pude preparar el vídeo angular: " + str(angle_err)[:300])
                st.info("Ahora abre **4 · Resultados 2D** para calcular y visualizar las métricas.")
            except Exception as e:
                st.session_state.pose_done = False
                st.error(f"Error durante el análisis: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())

with tabs[3]:
    st.subheader("Resultados 2D")

    if not st.session_state.get("pose_done"):
        st.info("Ejecuta primero **Analizar marcha**.")
    else:
        session_dir = Path(st.session_state.session_dir)
        json_dir = Path(st.session_state.pose_json_dir)
        df = load_pose_dataframe(json_dir)
        meta = st.session_state.get("meta1")

        if df.empty or not meta:
            st.error("No puedo cargar los datos del análisis.")
        else:
            fps = float(meta["fps"])
            n_frames = int(df["frame"].max()) + 1
            duration = n_frames / fps

            st.markdown("### Segmento que quieres analizar")
            st.caption(
                "En esta versión puedes excluir manualmente giros, zooms o partes no válidas. "
                "Más adelante la detección será automática."
            )

            start_s, end_s = st.slider(
                "Intervalo válido (segundos)",
                min_value=0.0,
                max_value=float(round(duration, 2)),
                value=(0.0, float(round(duration, 2))),
                step=max(0.01, round(1.0 / fps, 2)),
            )

            start_frame = int(round(start_s * fps))
            end_frame = min(n_frames - 1, int(round(end_s * fps)))

            if st.button("Calcular resultados del segmento", type="primary", use_container_width=True):
                try:
                    metrics, chart, crossings, seg = compute_metrics(
                        df, fps, start_frame, end_frame
                    )
                    st.session_state.metrics = metrics
                    st.session_state.chart = chart
                    st.session_state.crossings = crossings.tolist()
                    st.session_state.metrics_done = True

                    st.success("Resultados calculados. Permanecerán disponibles mientras esta sesión online siga activa.")
                except Exception as e:
                    st.error(str(e))

            if st.session_state.get("metrics_done"):
                metrics = st.session_state.metrics
                chart = st.session_state.chart

                st.markdown("### Resumen clínico 2D")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric(
                    "Cadencia estimada",
                    fmt(metric_value(metrics, "cadence_exp"), 1) + " pasos/min"
                    if metric_value(metrics, "cadence_exp") is not None and np.isfinite(metric_value(metrics, "cadence_exp"))
                    else "—"
                )
                c2.metric(
                    "Regularidad temporal",
                    fmt(metric_value(metrics, "regularity_cv"), 1) + " % CV"
                    if metric_value(metrics, "regularity_cv") is not None and np.isfinite(metric_value(metrics, "regularity_cv"))
                    else "—"
                )
                c3.metric(
                    "Tracking válido",
                    fmt(metric_value(metrics, "good_frames_pct"), 1) + " %"
                )
                c4.metric(
                    "Asimetría temporal",
                    fmt(metric_value(metrics, "temporal_asymmetry_exp"), 1) + " %"
                    if metric_value(metrics, "temporal_asymmetry_exp") is not None and np.isfinite(metric_value(metrics, "temporal_asymmetry_exp"))
                    else "—"
                )

                st.caption(
                    "Cadencia, regularidad y asimetría temporal se consideran experimentales "
                    "hasta validar los eventos de contacto del pie."
                )

                st.markdown("### Cinemática 2D proyectada · cadera, rodilla, tobillo y hombro")
                st.caption(
                    "Los valores son proyecciones 2D del plano de cámara. En lateral son especialmente útiles para patrones sagitales; "
                    "en frontal/posterior describen proyección/alineación y no son ángulos anatómicos 3D."
                )
                joint_tabs=st.tabs(["Cadera","Rodilla","Tobillo","Hombro"])
                joint_defs=[
                    ("Cadera","Cadera izquierda","Cadera derecha","hip_angle_l_p95","hip_angle_r_p95","hip_angle_p95_diff"),
                    ("Rodilla","Rodilla izquierda","Rodilla derecha","knee_flex_l_p95","knee_flex_r_p95","knee_flex_p95_diff"),
                    ("Tobillo","Tobillo izquierdo","Tobillo derecho","ankle_angle_l_p95","ankle_angle_r_p95","ankle_angle_p95_diff"),
                    ("Hombro","Hombro izquierdo","Hombro derecho","shoulder_angle_l_p95","shoulder_angle_r_p95","shoulder_angle_p95_diff"),
                ]
                for jt,(title,left_col,right_col,kl,kr,kd) in zip(joint_tabs,joint_defs):
                    with jt:
                        a,b,c=st.columns(3)
                        a.metric(f"{title} I · P95",fmt(metric_value(metrics,kl),1)+"°")
                        b.metric(f"{title} D · P95",fmt(metric_value(metrics,kr),1)+"°")
                        c.metric("Diferencia D/I",fmt(metric_value(metrics,kd),1)+"°")
                        st.line_chart(chart.set_index("time_s")[[left_col,right_col]],x_label="Tiempo (s)",y_label="Ángulo 2D (°)")
                        if title=="Tobillo":
                            st.caption("Tobillo = ángulo geométrico rodilla-tobillo-antepié; no equivale por sí solo a dorsiflexión anatómica.")

                st.markdown("### Alternancia distal derecha/izquierda")
                st.line_chart(chart.set_index("time_s")[["Alternancia D-I"]],x_label="Tiempo (s)",y_label="Diferencia vertical relativa (px)")

                with st.expander("Todas las métricas y calidad"):
                    table=pd.DataFrame(metrics)
                    st.dataframe(table[["label","value","unit","quality","notes"]],use_container_width=True,hide_index=True)

                st.markdown("### Vídeo procesado · esqueleto + ángulos dinámicos")
                angle_video=st.session_state.get("angle_video")
                if angle_video and Path(angle_video).exists():
                    st.video(angle_video)
                    st.caption("Cad=cadera · Rod=rodilla · Tob=tobillo · Hom=hombro. I/D = lado anatómico izquierdo/derecho.")
                else:
                    source_video=st.session_state.get("video1")
                    if source_video and st.button("Generar vídeo con ángulos",use_container_width=True):
                        with st.spinner("Dibujando esqueleto, ángulos y preparando H.264..."):
                            av,ae=generate_angle_video(source_video,df,session_dir,force=True)
                        if av:
                            st.session_state.angle_video=str(av); st.rerun()
                        else:
                            st.error(ae or "No se pudo generar el vídeo con ángulos.")
                    annotated=find_annotated_video(session_dir)
                    if annotated and annotated.exists():
                        show_browser_video(annotated,caption="Vídeo Pose2Sim convertido para Chrome/Streamlit")
                    else:
                        st.info("Los keypoints están calculados, pero todavía no encuentro un vídeo anotado.")

with tabs[4]:
    st.subheader("Modo 3D · futuro")
    st.write(
        "El modo de dos cámaras queda preparado como base para calibración, "
        "sincronización y triangulación Pose2Sim."
    )
    st.warning(
        "La v0.4 online todavía no interpreta dos vídeos 2D como biomecánica 3D. "
        "El 3D se activará únicamente tras calibrar correctamente las cámaras."
    )

st.divider()
st.caption(
    f"PhysioSentinel Gait · v{APP_VERSION} · sesión temporal online · "
    "sin almacenamiento clínico persistente"
)
