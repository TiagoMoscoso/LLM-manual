import os
import sys
import tempfile
import subprocess
import shutil
import requests
import wave
import struct
import signal

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QMessageBox, QProgressBar
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
STT_URL      = os.getenv("STT_URL",    "http://localhost:9000/inference")
TTS_BASE     = (os.getenv("TTS_BASE") or os.getenv("TTS_URL") or "http://localhost:5000").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
LANG         = os.getenv("LANG", "pt")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_MANUAL_MD = os.path.join(APP_DIR, "manual.md")
SUBJECT_NAME = os.getenv("SUBJECT_NAME", "Virtus").strip()
MAX_ANSWER_CHARS = 600


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def load_markdown(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def call_ollama_generate(base_url: str, model: str, prompt: str, timeout=180) -> str:
    url = f"{base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    ans = r.json().get("response", "").strip()
    return ans[:MAX_ANSWER_CHARS] if MAX_ANSWER_CHARS > 0 else ans


def call_stt_inference(stt_url: str, wav_path: str, language: str = LANG, timeout=180) -> str:
    with open(wav_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"language": language, "response_format": "text"}
        r = requests.post(stt_url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text.strip()


def call_tts_any(tts_base: str, text: str, timeout=180) -> bytes:
    candidates = [
        ("/tts",     "POST"),
        ("/api/tts", "POST"),
        ("/speak",   "POST"),
        ("/",        "GET"),
    ]
    for path, method in candidates:
        url = f"{tts_base.rstrip('/')}{path}"
        try:
            if method == "POST":
                r = requests.post(url, json={"text": text}, timeout=timeout)
            else:
                r = requests.get(url, params={"text": text}, timeout=timeout)
            if r.status_code in (200, 201) and r.content:
                return r.content
        except Exception:
            continue
    raise RuntimeError(f"TTS falhou. Base={tts_base}")


def open_with_system(path: str):
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        os.startfile(path)  


def _spawn_pcm_source(rate: int, channels: int):
    if which("ffmpeg"):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "pulse", "-i", "default",
            "-ac", str(channels), "-ar", str(rate),
            "-f", "s16le", "-"
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    elif which("arecord"):
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(rate), "-c", str(channels), "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raise RuntimeError("Instale 'ffmpeg' OU 'alsa-utils' (arecord).")


def record_wav_until_silence(path_out: str,
                             rate=16000, channels=1,
                             frame_ms=30,
                             start_threshold=900,
                             silence_threshold=700,
                             silence_seconds=2.0,
                             max_seconds=30.0):
    bytes_per_sample = 2
    frame_bytes = int(rate * frame_ms / 1000) * channels * bytes_per_sample
    max_frames = int((max_seconds * 1000) / frame_ms)
    needed_silence_frames = int((silence_seconds * 1000) / frame_ms)

    proc = _spawn_pcm_source(rate, channels)
    if proc.stdout is None:
        raise RuntimeError("Falha ao capturar áudio.")

    frames = []
    started = False
    silence_run = 0

    try:
        for _ in range(max_frames):
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            frames.append(chunk)
            samples = struct.unpack("<%dh" % (len(chunk)//2), chunk)
            peak = max(abs(s) for s in samples) if samples else 0

            if not started and peak >= start_threshold:
                started = True
                silence_run = 0
            elif started:
                if peak < silence_threshold:
                    silence_run += 1
                    if silence_run >= needed_silence_frames:
                        break
                else:
                    silence_run = 0
    finally:
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=1)
        except Exception:
            try: proc.kill()
            except Exception: pass

    with wave.open(path_out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bytes_per_sample)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))


class LLMWorker(QThread):
    """Executa a chamada ao LLM em background."""
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, prompt: str, parent=None):
        super().__init__(parent)
        self.prompt = prompt

    def run(self):
        try:
            ans = call_ollama_generate(OLLAMA_URL, OLLAMA_MODEL, self.prompt)
            self.done.emit(ans)
        except Exception as ex:
            self.error.emit(str(ex))


class TTSWorker(QThread):
    """Executa TTS em background e salva WAV temporário."""
    done = pyqtSignal(str)   
    error = pyqtSignal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text

    def run(self):
        try:
            wav_bytes = call_tts_any(TTS_BASE, self.text)
            fd, out = tempfile.mkstemp(prefix="tts_", suffix=".wav")
            os.close(fd)
            with open(out, "wb") as f:
                f.write(wav_bytes)
            self.done.emit(out)
        except Exception as ex:
            self.error.emit(str(ex))


class VoicePipelineWorker(QThread):
    """Grava -> STT -> LLM (tudo em background)."""
    text_ready = pyqtSignal(str)
    answer_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, manual_text: str, parent=None):
        super().__init__(parent)
        self.manual_text = manual_text or ""

    def build_prompt(self, question: str) -> str:
        return (
            "ATENÇÃO (INSTRUÇÕES OBRIGATÓRIAS) :\n"
            f"- Responda SOMENTE se a pergunta for sobre {SUBJECT_NAME} e estritamente com base no manual abaixo.\n"
            "- Se a pergunta não for sobre esse assunto ou não houver base suficiente no manual, responda exatamente:\n"
            "\"Só posso responder sobre {subject} com base no manual fornecido. Reformule sua pergunta.\".\n"
            "- Não invente, não cite fontes externas, não complemente com conhecimento geral.\n"
            "- Seja curto e direto (português do Brasil).\n\n"
            "--- MANUAL ---\n"
            f"{self.manual_text}\n"
            "---------------\n\n"
            f"Pergunta do usuário: {question}\n"
            "{subject} = " + SUBJECT_NAME
        )

    def run(self):
        fd, tmpwav = tempfile.mkstemp(prefix="ask_", suffix=".wav")
        os.close(fd)
        try:
            record_wav_until_silence(tmpwav)
            text = call_stt_inference(STT_URL, tmpwav)
            if not text:
                self.error.emit("Transcrição vazia.")
                return
            self.text_ready.emit(text)
            ans = call_ollama_generate(OLLAMA_URL, OLLAMA_MODEL, self.build_prompt(text))
            self.answer_ready.emit(ans)
        except Exception as ex:
            self.error.emit(str(ex))
        finally:
            try: os.remove(tmpwav)
            except: pass


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agente do Manual")
        self.resize(900, 700)

        self.manual_text = ""
        self.last_answer = ""
        self.last_audio_path = ""

        root = QWidget()
        grid = QGridLayout(root)
        row = 0

        self.manual_status = QLabel("Sem manual")
        grid.addWidget(self.manual_status, row, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        row += 1

        grid.addWidget(QLabel("Pergunta:"), row, 0, 1, 2)
        row += 1
        self.question = QTextEdit()
        self.question.setPlaceholderText("Escreva sua pergunta…")
        grid.addWidget(self.question, row, 0, 1, 2)
        row += 1

        self.ask_btn = QPushButton("Perguntar (texto → LLM)")
        grid.addWidget(self.ask_btn, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("Resposta:"), row, 0, 1, 2)
        row += 1
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        grid.addWidget(self.answer, row, 0, 1, 2)
        row += 1

        bbar = QHBoxLayout()
        self.rec_and_ask_btn = QPushButton("🎙️ Falar (para após 2s de silêncio)")
        self.save_tts_btn = QPushButton("Salvar WAV…")
        bbar.addWidget(self.rec_and_ask_btn)
        bbar.addWidget(self.save_tts_btn)
        grid.addLayout(bbar, row, 0, 1, 2)
        row += 1
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(1.0)  # 0.0–1.0

        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.setCentralWidget(root)
        self.loading = QProgressBar()
        self.loading.setRange(0, 0)              
        self.loading.setTextVisible(False)       
        self.loading.hide()
        grid.addWidget(self.loading, row, 0, 1, 2)
        row += 1

        self.ask_btn.clicked.connect(self.on_ask_text_async)
        self.rec_and_ask_btn.clicked.connect(self.on_record_and_ask_async)
        self.save_tts_btn.clicked.connect(self.on_save_tts)

        self.load_manual_md()


    def load_manual_md(self):
        try:
            self.manual_text = load_markdown(AUTO_MANUAL_MD)
            if self.manual_text:
                self.manual_status.setText(f"Manual: manual.md ({len(self.manual_text)} chars)")
            else:
                self.manual_status.setText("Sem manual.md encontrado")
        except Exception as ex:
            self.manual_status.setText(f"Erro ao ler manual.md: {ex}")


        
    def set_busy(self, busy: bool):
        self.ask_btn.setDisabled(busy)
        self.rec_and_ask_btn.setDisabled(busy)
        self.save_tts_btn.setDisabled(busy)

        if busy:
            self.loading.show()                       
            self.setCursor(Qt.CursorShape.BusyCursor)
            QApplication.processEvents()              
        else:
            self.loading.hide()                       
            self.setCursor(Qt.CursorShape.ArrowCursor)



    def prompt_from_question(self, q: str) -> str:
        return (
            "ATENÇÃO (INSTRUÇÕES OBRIGATÓRIAS):\n"
            f"- Responda SOMENTE se a pergunta for sobre {SUBJECT_NAME} e estritamente com base no manual abaixo.\n"
            "- Se a pergunta não for sobre esse assunto ou não houver base suficiente no manual, responda exatamente:\n"
            "\"Só posso responder sobre {subject} com base no manual fornecido. Reformule sua pergunta.\".\n"
            "- Não invente, não cite fontes externas, não complemente com conhecimento geral.\n"
            "- Seja curto e direto (português do Brasil).\n\n"
            "--- MANUAL ---\n"
            f"{self.manual_text}\n"
            "---------------\n\n"
            f"Pergunta do usuário: {q}\n"
            "{subject} = " + SUBJECT_NAME
        )

    def on_ask_text_async(self):
        q = self.question.toPlainText().strip()
        if not q:
            QMessageBox.information(self, "Aviso", "Digite uma pergunta.")
            return

        self.set_busy(True)
        self.answer.clear()

        self.llm_worker = LLMWorker(self.prompt_from_question(q), parent=self)
        self.llm_worker.done.connect(self._on_llm_answer_then_tts)
        self.llm_worker.error.connect(self._on_worker_error)
        self.llm_worker.finished.connect(lambda: None)  
        self.llm_worker.start()

    def _on_llm_answer_then_tts(self, ans: str):
        self.last_answer = ans
        self.answer.setPlainText(ans)

        self.tts_worker = TTSWorker(ans, parent=self)
        self.tts_worker.done.connect(self._on_tts_ready)
        self.tts_worker.error.connect(self._on_worker_error)
        self.tts_worker.finished.connect(lambda: self.set_busy(False))
        self.tts_worker.start()

    def on_record_and_ask_async(self):
        self.set_busy(True)
        self.answer.clear()

        self.voice_worker = VoicePipelineWorker(self.manual_text, parent=self)
        self.voice_worker.text_ready.connect(self._on_voice_text_ready)
        self.voice_worker.answer_ready.connect(self._on_voice_answer_then_tts)
        self.voice_worker.error.connect(self._on_worker_error)
        self.voice_worker.finished.connect(lambda: None)  
        self.voice_worker.start()

    def _on_voice_text_ready(self, text: str):
        self.question.setPlainText(text)

    def _on_voice_answer_then_tts(self, ans: str):
        self.last_answer = ans
        self.answer.setPlainText(ans)

        self.tts_worker = TTSWorker(ans, parent=self)
        self.tts_worker.done.connect(self._on_tts_ready)
        self.tts_worker.error.connect(self._on_worker_error)
        self.tts_worker.finished.connect(lambda: self.set_busy(False))
        self.tts_worker.start()

    def _on_tts_ready(self, wav_path: str):
        self.last_audio_path = wav_path
        try:
            self.player.stop()  # garante restart limpo
        except Exception:
            pass
        self.player.setSource(QUrl.fromLocalFile(wav_path))
        self.player.play()

    def _on_worker_error(self, msg: str):
        self.set_busy(False)
        QMessageBox.critical(self, "Erro", msg)

    def on_save_tts(self):
        if not self.last_audio_path or not os.path.exists(self.last_audio_path):
            QMessageBox.information(self, "Salvar WAV", "Nenhum áudio gerado ainda.")
            return
        QMessageBox.information(self, "Caminho do WAV", self.last_audio_path)


def main():
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
