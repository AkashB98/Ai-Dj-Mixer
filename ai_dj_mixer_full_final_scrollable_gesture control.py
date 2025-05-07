
import os
import json
import re
import threading
from tkinter import *
from tkinter import filedialog, ttk
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import simpleaudio as sa

# Optional Gesture Imports
gesture_enabled = True
try:
    import cv2
    import mediapipe as mp
except ImportError:
    gesture_enabled = False

import google.generativeai as genai
api_key = "AIzaSyD2_mfrP7NQf_ftITUFkZMih6M2YcEh6-A"

song_paths = []
cue_points = [{}, {}]
volume_levels = [1.0, 1.0]
preview_audio = [None, None]
labels = []
waveform_canvases = []
scroll_frames = []

root = Tk()
root.title("Rap Song Mixer")

main_canvas = Canvas(root, width=1200, height=900)
scroll_y = Scrollbar(root, orient="vertical", command=main_canvas.yview)
main_frame = Frame(main_canvas)

main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
main_canvas.configure(yscrollcommand=scroll_y.set)

main_canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

Label(main_frame, text="Select 2 rap songs to mix").pack(pady=10)
for i in range(2):
    Button(main_frame, text=f"Load Song {i+1}", command=lambda i=i: load_song(i)).pack()
    label = Label(main_frame, text="No file selected")
    label.pack()
    labels.append(label)
    canvas_frame = Frame(main_frame)
    canvas_frame.pack()
    waveform_canvases.append(canvas_frame)
    scroll_frame = Frame(main_frame)
    scroll_frame.pack(fill="both", expand=True)
    scroll_frames.append(scroll_frame)
    Label(main_frame, text=f"Track {i+1} Volume").pack()
    Scale(main_frame, from_=0, to=100, orient=HORIZONTAL, command=lambda val, idx=i: update_volume(idx, val)).pack()

mix_style = StringVar()
mix_style.set("crossfade")
Label(main_frame, text="Select mix style:").pack(pady=(10, 0))
Radiobutton(main_frame, text="Overlay", variable=mix_style, value="overlay").pack()
Radiobutton(main_frame, text="Crossfade", variable=mix_style, value="crossfade").pack()
Radiobutton(main_frame, text="AI Suggest (basic)", variable=mix_style, value="ai").pack()

Button(main_frame, text="🎻 Let AI Suggest Mix", command=lambda: run_ai_suggestion()).pack(pady=5)
Button(main_frame, text="Mix & Export", command=lambda: mix_and_export()).pack(pady=5)
Button(main_frame, text="🌿 Auto Mix with AI", command=lambda: auto_mix_from_ai("./songs", api_key)).pack(pady=5)

status_label = Label(main_frame, text="")
status_label.pack()

def load_song(index):
    path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if path:
        if len(song_paths) > index:
            song_paths[index] = path
        else:
            song_paths.append(path)
        labels[index].config(text=os.path.basename(path))
        display_waveform(path, waveform_canvases[index])
        display_scrollable_waveform(path, scroll_frames[index], index)

def display_waveform(path, frame):
    y, sr = librosa.load(path)
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="mediumblue")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def display_scrollable_waveform(path, frame, track_index):
    y, sr = librosa.load(path)
    fig, ax = plt.subplots(figsize=(12, 2), dpi=100)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="crimson")
    ax.set_title(f"Full waveform: {os.path.basename(path)}")
    fig.tight_layout()
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    Button(frame, text="Set Cue Point Here", command=lambda: set_cue_point(track_index)).pack(pady=2)
    Button(frame, text="Loop 4s from Cue", command=lambda: loop_from_cue(track_index)).pack(pady=2)
    Button(frame, text="Preview From Cue", command=lambda: preview_from_cue(track_index)).pack(pady=2)

def set_cue_point(index):
    cue_points[index]['start_ms'] = 5000
    status_label.config(text=f"Cue set for Track {index+1} at 5s")

def loop_from_cue(index):
    if 'start_ms' not in cue_points[index]:
        status_label.config(text="Set cue point first")
        return
    seg = AudioSegment.from_file(song_paths[index])
    loop = seg[cue_points[index]['start_ms']:cue_points[index]['start_ms'] + 4000] * 3
    out = filedialog.asksaveasfilename(defaultextension=".mp3")
    if out:
        loop.export(out, format="mp3")

def preview_from_cue(index):
    if 'start_ms' not in cue_points[index]:
        return
    clip = AudioSegment.from_file(song_paths[index])[cue_points[index]['start_ms']:cue_points[index]['start_ms'] + 4000]
    preview_audio[index] = sa.play_buffer(clip.raw_data, num_channels=clip.channels,
                                          bytes_per_sample=clip.sample_width, sample_rate=clip.frame_rate)

def update_volume(index, val):
    volume_levels[index] = float(val) / 100

def suggest_mix_with_ai(files, key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        prompt = f"Suggest 2 good rap songs to mix from: {', '.join(files)}. Return JSON with song1, song2, mix_style"
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        return f"AI Error: {e}"

def extract_json_from_response(resp):
    try:
        match = re.search(r'\{.*?\}', resp, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("JSON error:", e)
    return None

def run_ai_suggestion():
    folder = "./songs"
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    except:
        status_label.config(text="❌ Folder error")
        return
    if len(files) < 2:
        status_label.config(text="❌ Add at least 2 MP3s")
        return
    result = suggest_mix_with_ai(files, api_key)
    print("🎧 AI Suggestion:
", result)
    status_label.config(text="✅ Check terminal for suggestion")

def mix_and_export():
    if len(song_paths) < 2:
        status_label.config(text="⚠️ Select 2 songs")
        return
    s1 = AudioSegment.from_file(song_paths[0]) - (1 - volume_levels[0]) * 60
    s2 = AudioSegment.from_file(song_paths[1]) - (1 - volume_levels[1]) * 60
    s1, s2 = s1.fade_in(2000), s2.fade_out(2000)
    m = mix_style.get()
    mixed = s1.overlay(s2) if m == "overlay" else s1.append(s2, crossfade=3000)
    path = filedialog.asksaveasfilename(defaultextension=".mp3")
    if path:
        mixed.export(path, format="mp3")
        status_label.config(text="✅ Mix exported!")

def auto_mix_from_ai(folder, key):
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".mp3")]
        result = suggest_mix_with_ai(files, key)
        print("🎷 AI:", result)
        info = extract_json_from_response(result)
        if not info:
            raise ValueError("Invalid JSON")
        s1 = AudioSegment.from_file(os.path.join(folder, info["song1"])) - (1 - volume_levels[0]) * 60
        s2 = AudioSegment.from_file(os.path.join(folder, info["song2"])) - (1 - volume_levels[1]) * 60
        mixed = s1.overlay(s2) if info["mix_style"] == "overlay" else s1.append(s2, crossfade=3000)
        path = filedialog.asksaveasfilename(defaultextension=".mp3")
        if path:
            mixed.export(path, format="mp3")
            status_label.config(text=f"✅ Mixed {info['song1']} & {info['song2']}")
    except Exception as e:
        print("❌ AutoMix error:", e)
        status_label.config(text="❌ AutoMix error")

if gesture_enabled:
    def start_gesture_detection():
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
            cap = cv2.VideoCapture(0)
            while True:
                success, image = cap.read()
                if not success:
                    continue
                image = cv2.flip(image, 1)
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    for hl in result.multi_hand_landmarks:
                        lm = hl.landmark
                        if abs(lm[mp_hands.HandLandmark.THUMB_TIP].x - lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x) < 0.05:
                            root.event_generate("<<GesturePlayPause>>")
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("Gesture failed:", e)

    def on_gesture_play_pause(e):
        for i in range(2):
            if preview_audio[i] and preview_audio[i].is_playing():
                preview_audio[i].stop()
            else:
                preview_from_cue(i)

    root.bind("<<GesturePlayPause>>", on_gesture_play_pause)
    threading.Thread(target=start_gesture_detection, daemon=True).start()

root.mainloop()
