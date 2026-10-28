"""
camera_live_v2.py — version optimisée de camera_live_final.py

Changements principaux (repérés par [MODIF] dans le code) :
  1. UNE seule connexion SSH ouverte au démarrage et réutilisée (paramiko),
     au lieu d'un nouveau scp + un nouveau ssh à chaque image (2 poignées de main SSH ~1-2 s sur le Cortex-A9).
  2. Bug corrigé : le temps total était multiplié par 3000 au lieu de 1000.
  3. Prétraitement identique à l'entraînement : découpe centrale 2,6:1 + resize bilinéaire (INTER_LINEAR).
  4. Affichage du temps détaillé (transfert / HPS / total), de l'âge du résultat et de l'image réellement analysée.
  5. Classe OK réglable en ligne de commande (--ok-class), et enregistrement facultatif pour l'interface Streamlit.

Installation :  python -m pip install paramiko opencv-python numpy tensorflow
Lancement   :  python camera_live_v2.py --host 192.168.1.50 --password <mdp>     (ou clé SSH : sans --password)
"""
import argparse
import io
import re
import threading
import time

import cv2
import numpy as np
import paramiko                       # [MODIF] connexion SSH persistante
import tensorflow as tf

try:                                  # [MODIF] facultatif : alimente l'interface Streamlit (journal.py)
    from journal import enregistrer_resultat
except ImportError:
    enregistrer_resultat = None

INTERVAL = 1.0                        # une classification par seconde (si le traitement le permet)
RATIO = 2.6                           # découpe hauteur/largeur, comme les photos du dataset

GREEN = (0, 200, 0)
RED = (0, 0, 255)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)


def load_model_input(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    info = interpreter.get_input_details()[0]
    if tuple(info["shape"]) != (1, 128, 128, 3):
        raise ValueError("Expected input 1x128x128x3")
    if info["dtype"] != np.int8:
        raise ValueError("Expected INT8 model input")
    scale, zero_point = info["quantization"]
    if scale <= 0:
        raise ValueError("Invalid input quantization")
    print("Input scale:", scale, "| zero point:", zero_point)
    return scale, zero_point


def crop_box(frame):
    """[MODIF] Rectangle central 2,6:1 (même découpe que les scripts de capture du dataset)."""
    H, W = frame.shape[:2]
    h = int(H * 0.9)
    w = int(h / RATIO)
    x0, y0 = (W - w) // 2, (H - h) // 2
    return x0, y0, w, h


def preprocess(frame, scale, zero_point, mode):
    x0, y0, w, h = crop_box(frame)
    rgb = cv2.cvtColor(frame[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)   # [MODIF] bilinéaire, comme l'entraînement
    values = rgb.astype(np.float32)
    if mode == "unit":
        values /= 255.0
    elif mode == "minus_one_one":
        values = values / 127.5 - 1.0
    quantized = np.clip(np.rint(values / scale + zero_point), -128, 127).astype(np.int8)
    return quantized, rgb


class HpsClient:
    """[MODIF] Connexion SSH ouverte UNE fois ; chaque image = 1 envoi SFTP + 1 commande sur la même connexion."""

    def __init__(self, host, user, password, remote_dir):
        self.remote_dir = remote_dir
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        t0 = time.monotonic()
        self.ssh.connect(host, username=user, password=password, timeout=10,
                         look_for_keys=password is None, allow_agent=password is None)
        self.ssh.get_transport().set_keepalive(15)
        self.sftp = self.ssh.open_sftp()
        print(f"Connecté à {user}@{host} en {(time.monotonic() - t0) * 1000:.0f} ms (connexion gardée ouverte)")

    def classify(self, quantized):
        remote_file = f"{self.remote_dir}/camera_input_int8.bin"
        t0 = time.monotonic()
        self.sftp.putfo(io.BytesIO(quantized.tobytes()), remote_file)          # envoi de l'image (49 152 octets)
        t1 = time.monotonic()
        command = f"cd {self.remote_dir} && ./bottle_ai_live_robuste.sh camera_input_int8.bin"
        _, stdout, stderr = self.ssh.exec_command(command, timeout=150)
        out = stdout.read().decode(errors="replace")
        status = stdout.channel.recv_exit_status()
        t2 = time.monotonic()
        if status != 0:
            raise RuntimeError(f"script HPS en erreur ({status}) : {stderr.read().decode(errors='replace')}")
        match = re.search(r"RESULT class=(\d+) fpga_ms=([\d.]+) arm_ms=([\d.]+) total_ms=([\d.]+)", out)
        if not match:
            raise RuntimeError("No valid result from HPS:\n" + out)
        return {
            "class": int(match.group(1)),
            "fpga_ms": float(match.group(2)),
            "arm_ms": float(match.group(3)),
            "hps_ms": float(match.group(4)),
            "transfer_ms": (t1 - t0) * 1000.0,
            "script_ms": (t2 - t1) * 1000.0,      # temps de la commande vue du PC (inclut le lancement du script)
            "roundtrip_ms": (t2 - t0) * 1000.0,   # [MODIF] x1000 (et plus x3000)
        }

    def close(self):
        self.sftp.close()
        self.ssh.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bottle_cnn_int8.tflite")
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="root")
    p.add_argument("--password", default=None, help="mot de passe SSH (sinon clé SSH)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mode", choices=["raw", "unit", "minus_one_one"], default="raw")
    p.add_argument("--remote-dir", default="/root/ai")
    p.add_argument("--ok-class", type=int, default=0,
                   help="numéro de la classe OK renvoyé par le FPGA (vérifier avec une bouteille connue)")
    args = p.parse_args()
    OK_CLASS = args.ok_class

    scale, zero_point = load_model_input(args.model)
    hps = HpsClient(args.host, args.user, args.password, args.remote_dir)

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera")

    state = {"status": "WAITING", "busy": False, "last_start": 0.0, "done_at": 0.0,
             "res": None, "thumb": None, "error": ""}
    lock = threading.Lock()

    def worker(frame):
        try:
            quantized, thumb = preprocess(frame, scale, zero_point, args.mode)
            res = hps.classify(quantized)
            status = "OK" if res["class"] == OK_CLASS else "DEFECT"
            with lock:
                state.update(status=status, res=res, thumb=thumb, error="", done_at=time.monotonic())
            print(f"{status:6s} transfert {res['transfer_ms']:.0f} ms | commande {res['script_ms']:.0f} ms "
                  f"(HPS {res['hps_ms']:.0f}, FPGA {res['fpga_ms']:.0f}, ARM {res['arm_ms']:.0f}) "
                  f"| total {res['roundtrip_ms']:.0f} ms")
            if enregistrer_resultat:
                enregistrer_resultat(status, latence_ms=res["roundtrip_ms"])
        except Exception as exc:
            with lock:
                state.update(status="ERROR", error=str(exc))
            print("HPS error:", exc)
        finally:
            with lock:
                state["busy"] = False

    print("Camera running. Q pour quitter.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera read failed")
                break
            now = time.monotonic()
            with lock:
                should_start = not state["busy"] and now - state["last_start"] >= INTERVAL
                if should_start:
                    state["busy"] = True
                    state["last_start"] = now
                snap = dict(state)
            if should_start:
                threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()

            # ---------------- affichage
            height, width = frame.shape[:2]
            panel = 280
            canvas = np.zeros((height, width + panel, 3), dtype=np.uint8)
            vue = frame.copy()
            x0, y0, w, h = crop_box(frame)                          # [MODIF] zone réellement analysée
            cv2.rectangle(vue, (x0, y0), (x0 + w, y0 + h), YELLOW, 2)
            canvas[:, :width] = vue

            status = snap["status"]
            color = GREEN if status == "OK" else RED if status == "DEFECT" else GRAY
            cv2.rectangle(canvas, (width + 20, 20), (width + panel - 20, 120), color, -1)
            ts = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
            cv2.putText(canvas, status, (width + (panel - ts[0]) // 2, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 3)

            if snap["thumb"] is not None:                           # [MODIF] image analysée (128x128)
                th = cv2.cvtColor(snap["thumb"], cv2.COLOR_RGB2BGR)
                canvas[135:263, width + 20:width + 148] = th
                age = now - snap["done_at"]
                cv2.putText(canvas, "image analysee", (width + 155, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
                cv2.putText(canvas, f"il y a {age:.1f} s", (width + 155, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

            r = snap["res"]
            if r:
                lignes = [f"Transfert : {r['transfer_ms']:.0f} ms",
                          f"HPS       : {r['hps_ms']:.0f} ms",
                          f"  FPGA {r['fpga_ms']:.0f} / ARM {r['arm_ms']:.0f} ms",
                          f"Total     : {r['roundtrip_ms']:.0f} ms"]
                for i, t in enumerate(lignes):
                    cv2.putText(canvas, t, (width + 15, height - 110 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            if snap["busy"]:
                cv2.putText(canvas, "PROCESSING...", (width + 15, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

            cv2.imshow("Bottle AI - Cyclone V FPGA + HPS", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        hps.close()


if __name__ == "__main__":
    main()
