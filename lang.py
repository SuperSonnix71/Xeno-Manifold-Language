#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ============================== Logging ============================== #

def setup_logger(verbosity: int) -> logging.Logger:
    logger = logging.getLogger("xeno_manifold")
    logger.propagate = False
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = '{"ts":"%(asctime)s","lvl":"%(levelname)s","evt":"%(message)s","mod":"%(module)s","func":"%(funcName)s","line":%(lineno)d}'
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(h)
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logger.setLevel(level)
    logger.debug("Logger initialized verbosity=%d", verbosity)
    return logger
    logger.debug("Logger initialized verbosity=%d", verbosity)
    return logger

# ============================== Utils =============================== #

VOWELS = set("aeiouy")
TOKEN_RE = re.compile(r"[^\w\s'\-\.]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.sub(" ", text).split() if t]

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    h = h % 1.0
    def f(n):
        k = (n + h*12) % 12
        a = s * min(l, 1 - l)
        return l - a * max(-1, min(k - 3, 9 - k, 1))
    r, g, b = f(0), f(8), f(4)
    return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

def stable_hash_u64(s: str) -> int:
    h = 1469598103934665603
    for b in s.encode("utf-8"):
        h ^= b
        h *= 1099511628211
        h &= 0xFFFFFFFFFFFFFFFF
    return h

def phash01(s: str, salt: int = 0) -> float:
    x = stable_hash_u64(f"{salt}:{s}")
    return (x / 2**64)

def word_stats(w: str) -> Dict[str, int]:
    wl = w.lower()
    vowels = sum(1 for c in wl if c in VOWELS)
    consonants = sum(1 for c in wl if c.isalpha() and c not in VOWELS)
    digits = sum(1 for c in wl if c.isdigit())
    return {"len": len(wl), "v": vowels, "c": consonants, "d": digits}

def bezier_cubic(p0, p1, p2, p3, t: float) -> Tuple[float, float, float]:
    u = 1 - t
    b0 = u*u*u
    b1 = 3*u*u*t
    b2 = 3*u*t*t
    b3 = t*t*t
    return (
        b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0],
        b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1],
        b0*p0[2] + b1*p1[2] + b2*p2[2] + b3*p3[2],
    )

def adsr_envelope(t: float, attack: float, decay: float, sustain: float, release: float, note_on: float, note_off: float) -> float:
    if t < note_on:
        return 0.0
    dt = t - note_on
    if dt < attack:
        return dt / attack if attack > 0 else 1.0
    dt -= attack
    if dt < decay:
        return lerp(1.0, sustain, dt / decay)
    if t < note_off:
        return sustain
    rt = t - note_off
    return lerp(sustain, 0.0, clamp(rt / max(1e-6, release), 0.0, 1.0))

def soft_clip(x: float, k: float = 0.8) -> float:
    return math.tanh(k * x)

def map_range(x: float, a: float, b: float, c: float, d: float) -> float:
    if b == a:
        return c
    t = (x - a) / (b - a)
    return lerp(c, d, t)

# ============================== Data ================================ #

@dataclass
class MotionChannel:
    name: str
    keyframes: List[Tuple[float, float]]  # time, value
    bezier: List[Tuple[float, float, float]]  # cubic bezier tangents per segment (x1,y1,x2)

@dataclass
class ColorGradient:
    space: str  # HSL
    stops: List[Tuple[float, float, float, float]]  # t, H, S, L

@dataclass
class GeometrySpline:
    control_points: List[Tuple[float, float, float]]
    closed: bool
    thickness: float

@dataclass
class ScalarField:
    basis: str  # "sum-of-gaussians"
    centers: List[Tuple[float, float, float]]
    weights: List[float]
    sigmas: List[float]

@dataclass
class AudioPatch:
    sample_rate: int
    duration: float
    carriers: List[Dict[str, float]]  # freq, amp
    lfo: Dict[str, float]             # rate, depth
    fm: Dict[str, float]              # mod_freq, index
    adsr: Dict[str, float]            # a d s r
    pan_curve: List[Tuple[float, float]]  # t, pan[-1..1]

@dataclass
class AlienGlyph:
    symbol: str           # The visual representation
    phonetic: str         # Phonetic transcription
    meaning: str          # Original word
    complexity: int       # Visual complexity (1-10)
    strokes: List[str]    # Stroke descriptions

@dataclass
class LexemeManifold:
    token: str
    seed: int
    glyph: AlienGlyph
    geometry: GeometrySpline
    color: ColorGradient
    motion: Dict[str, MotionChannel]
    field: ScalarField
    audio: AudioPatch

@dataclass
class ManifoldDocument:
    version: str
    created_at: float
    input_text: str
    manifolds: List[LexemeManifold]

# ========================== Generators ============================== #

class AlienGlyphGenerator:
    """Generates alien writing system glyphs"""
    
    # Alien symbol components
    BASES = ["◇", "○", "△", "□", "◎", "⬟", "⬢", "◈", "◐", "◑", "◒", "◓"]
    ACCENTS = ["˙", "ˋ", "ˊ", "ˆ", "˜", "¨", "ˇ", "´", "῀", "῁"]
    CONNECTORS = ["—", "│", "╱", "╲", "┼", "╳", "※"]
    MODIFIERS = ["⁀", "⁔", "⁖", "⁘", "⁙", "⁚", "⁛"]
    
    # Phonetic symbols
    CONSONANTS = ["zh", "kh", "th", "sh", "xh", "gh", "qh", "ph", "ch", "rh"]
    VOWELS = ["aa", "ee", "ii", "oo", "uu", "ai", "ei", "ou", "ae"]
    
    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger
    
    def generate_glyph(self, token: str, seed: int) -> AlienGlyph:
        """Generate an alien glyph for a token"""
        s = token.lower()
        stats = word_stats(s)
        
        # Choose base symbol
        base_idx = seed % len(self.BASES)
        base = self.BASES[base_idx]
        
        # Add complexity based on word length
        complexity = min(10, max(1, stats["len"] + stats["c"]))
        
        # Build the symbol
        parts = [base]
        
        # Add accents for vowels
        for i in range(min(stats["v"], 3)):
            accent_idx = (seed + i * 7) % len(self.ACCENTS)
            parts.append(self.ACCENTS[accent_idx])
        
        # Add connectors for consonants
        if stats["c"] > 2:
            conn_idx = (seed + stats["c"]) % len(self.CONNECTORS)
            parts.insert(1, self.CONNECTORS[conn_idx])
        
        # Add modifiers
        if stats["len"] > 5:
            mod_idx = (seed + stats["len"]) % len(self.MODIFIERS)
            parts.append(self.MODIFIERS[mod_idx])
        
        symbol = "".join(parts)
        
        # Generate phonetic representation
        phonetic_parts = []
        for i, char in enumerate(s):
            if char in VOWELS:
                v_idx = (ord(char) + i) % len(self.VOWELS)
                phonetic_parts.append(self.VOWELS[v_idx])
            elif char.isalpha():
                c_idx = (ord(char) + i) % len(self.CONSONANTS)
                phonetic_parts.append(self.CONSONANTS[c_idx])
        
        phonetic = "-".join(phonetic_parts) if phonetic_parts else "xh"
        
        # Generate stroke descriptions
        strokes = []
        strokes.append(f"Base glyph: {base}")
        if stats["v"] > 0:
            strokes.append(f"{stats['v']} vowel accent(s)")
        if stats["c"] > 2:
            strokes.append("Connector stroke")
        if stats["len"] > 5:
            strokes.append("Complexity modifier")
        
        return AlienGlyph(
            symbol=symbol,
            phonetic=phonetic,
            meaning=token,
            complexity=complexity,
            strokes=strokes
        )

class XenoManifoldGenerator:
    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger
        self.glyph_gen = AlienGlyphGenerator(logger)

    def manifold_for_token(self, token: str, base_duration: float, sample_rate: int) -> LexemeManifold:
        s = token.lower()
        seed = stable_hash_u64(s)
        stats = word_stats(s)
        self.log.debug("Token='%s' stats=%s seed=%d", token, stats, seed)

        # Generate alien glyph
        glyph = self.glyph_gen.generate_glyph(token, seed)

        # Geometry spline (3D cubic path)
        ncp = 6 + (seed % 5)  # 6..10 control points
        cps: List[Tuple[float, float, float]] = []
        for i in range(ncp):
            t = (i + 1) / (ncp + 1)
            phi = 2 * math.pi * (phash01(s, i) * (1 + (stats["v"] % 3)))
            r = 0.6 + 0.6 * phash01(s, i + 101)
            z = map_range(math.sin(phi), -1, 1, -1, 1) * (0.3 + 0.2 * phash01(s, i + 202))
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            cps.append((x, y, z))
        thick = 0.02 + 0.06 * phash01(s, 333)
        spline = GeometrySpline(control_points=cps, closed=False, thickness=thick)

        # Color gradient (HSL stops)
        stops: List[Tuple[float, float, float, float]] = []
        hues = [(phash01(s, 400+k)*1.0) for k in range(3)]
        hues.sort()
        for k, h in enumerate(hues):
            sat = 0.45 + 0.45 * phash01(s, 500+k)
            lig = 0.35 + 0.40 * phash01(s, 600+k)
            t = k / (len(hues)-1 if len(hues) > 1 else 1)
            stops.append((t, h, clamp(sat, 0.0, 1.0), clamp(lig, 0.0, 1.0)))
        gradient = ColorGradient(space="HSL", stops=stops)

        # Motion channels (position xyz, rotation xyz, scale)
        dur = base_duration * (1.0 + 0.4 * phash01(s, 700))
        def motion_channel(name: str, base: float, amp: float, rate: float) -> MotionChannel:
            # 5 keyframes with bezier curves
            kf, bz = [], []
            for i in range(5):
                tt = i / 4 * dur
                phase = 2*math.pi*(phash01(s, 800 + i + hash(name)) + rate*tt/dur)
                val = base + amp * math.sin(phase)
                kf.append((tt, val))
                bz.append((0.25, 0.0, 0.75))
            return MotionChannel(name=name, keyframes=kf, bezier=bz)

        motion = {
            "pos_x": motion_channel("pos_x", 0.0, 0.6, 1.0 + phash01(s, 900)),
            "pos_y": motion_channel("pos_y", 0.0, 0.6, 1.3 + phash01(s, 901)),
            "pos_z": motion_channel("pos_z", 0.0, 0.4, 0.8 + phash01(s, 902)),
            "rot_x": motion_channel("rot_x", 0.0, math.pi/6, 0.7 + phash01(s, 903)),
            "rot_y": motion_channel("rot_y", 0.0, math.pi/6, 0.9 + phash01(s, 904)),
            "rot_z": motion_channel("rot_z", 0.0, math.pi/6, 1.1 + phash01(s, 905)),
            "scale": motion_channel("scale", 1.0, 0.25, 0.6 + phash01(s, 906)),
        }

        # Scalar field as sum-of-gaussians
        ncent = 3 + (seed % 3)  # 3..5 centers
        centers, weights, sigmas = [], [], []
        for i in range(ncent):
            cx = map_range(phash01(s, 1000+i), 0, 1, -0.9, 0.9)
            cy = map_range(phash01(s, 1100+i), 0, 1, -0.9, 0.9)
            cz = map_range(phash01(s, 1200+i), 0, 1, -0.9, 0.9)
            w = map_range(phash01(s, 1300+i), 0, 1, -1.0, 1.0)
            sg = 0.2 + 0.6 * phash01(s, 1400+i)
            centers.append((cx, cy, cz)); weights.append(w); sigmas.append(sg)
        field3d = ScalarField(basis="sum-of-gaussians", centers=centers, weights=weights, sigmas=sigmas)

        # Audio patch
        letter_energy = stats["c"] + 0.8*stats["v"] + 0.5*stats["d"]
        base_freq = 110.0 + 8.0 * letter_energy + 20.0 * phash01(s, 1500)
        carriers = []
        ncar = 2 + (seed % 3)  # 2..4 partials
        for i in range(ncar):
            ratio = 1.0 + i + phash01(s, 1600+i)
            amp = 0.3 / (i + 1) * (0.7 + 0.6 * phash01(s, 1700+i))
            carriers.append({"freq": base_freq * ratio, "amp": amp})
        lfo = {"rate": 0.2 + 2.3 * phash01(s, 1800), "depth": 0.1 + 0.4 * phash01(s, 1801)}
        fm = {"mod_freq": base_freq * (0.5 + phash01(s, 1900)), "index": 0.2 + 3.0 * phash01(s, 1901)}
        adsr = {"a": 0.03 + 0.12 * phash01(s, 2000), "d": 0.06 + 0.20 * phash01(s, 2001), "s": 0.3 + 0.6 * phash01(s, 2002), "r": 0.08 + 0.35 * phash01(s, 2003)}
        pan_curve = [(0.0, map_range(phash01(s, 2100), 0, 1, -0.9, 0.9)), (dur, map_range(phash01(s, 2101), 0, 1, -0.9, 0.9))]
        audio = AudioPatch(sample_rate=sample_rate, duration=dur, carriers=carriers, lfo=lfo, fm=fm, adsr=adsr, pan_curve=pan_curve)

        return LexemeManifold(
            token=token,
            seed=seed,
            glyph=glyph,
            geometry=spline,
            color=gradient,
            motion=motion,
            field=field3d,
            audio=audio
        )

# ========================== Audio Renderer =========================== #

class AudioRenderer:
    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger

    def render_patch(self, patch: AudioPatch, start_time: float, total_duration: float) -> Tuple[List[float], List[float]]:
        sr = patch.sample_rate
        n_total = int(total_duration * sr)
        left = [0.0] * n_total
        right = [0.0] * n_total

        dur = patch.duration
        on = start_time
        off = start_time + dur

        for i in range(int(dur * sr)):
            t = start_time + i / sr
            env = adsr_envelope(t, patch.adsr["a"], patch.adsr["d"], patch.adsr["s"], patch.adsr["r"], on, off)
            pan_t = map_range(t, patch.pan_curve[0][0] + start_time, patch.pan_curve[-1][0] + start_time, 0.0, 1.0)
            pan = lerp(patch.pan_curve[0][1], patch.pan_curve[-1][1], clamp(pan_t, 0.0, 1.0))
            l_gain = math.sqrt((1 - pan) * 0.5)
            r_gain = math.sqrt((1 + pan) * 0.5)

            mod = patch.fm["index"] * math.sin(2*math.pi*patch.fm["mod_freq"]*t)
            lfo = patch.lfo["depth"] * math.sin(2*math.pi*patch.lfo["rate"]*t)
            sample = 0.0
            for car in patch.carriers:
                freq = car["freq"] * (1.0 + 0.01 * lfo)
                sample += car["amp"] * math.sin(2*math.pi*freq*t + mod)

            s_val = soft_clip(sample * env, 0.9)
            idx = int(t * sr)
            if 0 <= idx < n_total:
                left[idx] += s_val * l_gain
                right[idx] += s_val * r_gain

        # normalize
        peak = max(1e-9, max(max(abs(x) for x in left), max(abs(x) for x in right)))
        norm = 0.98 / peak
        left = [x * norm for x in left]
        right = [x * norm for x in right]
        return left, right

    def write_wav_stereo(self, path: str, left: List[float], right: List[float], sample_rate: int) -> None:
        with open(path, "wb") as f:
            # WAV header
            n_frames = len(left)
            n_channels = 2
            bits = 16
            byte_rate = sample_rate * n_channels * bits // 8
            block_align = n_channels * bits // 8
            data_bytes = n_frames * block_align

            def w(tag, val, size):
                f.write(tag)
                f.write(struct.pack("<I", size))
                f.write(val)

            # RIFF
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_bytes))
            f.write(b"WAVE")
            # fmt
            f.write(b"fmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
            # data
            f.write(b"data")
            f.write(struct.pack("<I", data_bytes))
            for i in range(n_frames):
                li = int(clamp(left[i], -1.0, 1.0) * 32767)
                ri = int(clamp(right[i], -1.0, 1.0) * 32767)
                f.write(struct.pack("<hh", li, ri))

# ============================= Export ================================= #

def manifolds_to_json(doc: ManifoldDocument) -> Dict:
    def serialize(obj):
        if isinstance(obj, (ManifoldDocument, LexemeManifold, AlienGlyph, GeometrySpline, ColorGradient, MotionChannel, ScalarField, AudioPatch)):
            d = asdict(obj)
            return d
        raise TypeError(f"Unserializable type: {type(obj)}")
    return serialize(doc)

# ============================== Console UI ================================== #

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def print_banner():
    print("\n" + "="*70)
    print(" " * 15 + "XENO MANIFOLD LANGUAGE GENERATOR")
    print("="*70 + "\n")

def print_menu():
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                         MAIN MENU                               │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  1. Generate from Text Input                                    │")
    print("│  2. Generate from File                                          │")
    print("│  3. Configure Settings                                          │")
    print("│  4. View Current Settings                                       │")
    print("│  5. Exit                                                        │")
    print("└─────────────────────────────────────────────────────────────────┘")

def get_menu_choice() -> str:
    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("❌ Invalid choice. Please enter a number between 1 and 5.")
        except KeyboardInterrupt:
            print("\n")
            return '5'  # Treat Ctrl+C as exit

def get_text_input() -> str:
    print("\n" + "─"*70)
    print("TEXT INPUT")
    print("─"*70)
    print("Enter your text below.")
    print("When finished, press Enter on an empty line, then type 'done' and press Enter.")
    print("Or press Ctrl+D (EOF) when finished.")
    print("─"*70)
    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'done':
                break
            lines.append(line)
        except EOFError:
            break
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("No text entered.")
    return text

def get_file_input() -> str:
    print("\n" + "─"*70)
    file_path = input("Enter the path to your text file: ").strip()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        data = fh.read()
    if not data.strip():
        raise ValueError("File is empty.")
    print(f"✓ Loaded {len(data)} characters from file.")
    return data

def get_output_options() -> Dict[str, Optional[str]]:
    print("\n" + "─"*70)
    print("OUTPUT OPTIONS - MULTI-DIMENSIONAL FORMATS")
    print("─"*70)
    
    outputs = {}
    used_names = []
    
    # JSON - Complete manifold data
    choice = input("💾 Save JSON (complete manifold data)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["json"] = get_unique_filename("output.json", "manifold_output.json", used_names)
    else:
        outputs["json"] = None
    
    # Phonetic text
    choice = input("📝 Save phonetic text transcription? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["txt"] = get_unique_filename("phonetic.txt", "manifold_phonetic.txt", used_names)
    else:
        outputs["txt"] = None
    
    # SVG visual
    choice = input("🎨 Save SVG visualization (glyphs + colors)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["svg"] = get_unique_filename("glyphs.svg", "manifold_glyphs.svg", used_names)
    else:
        outputs["svg"] = None
    
    # 3D geometry
    choice = input("📐 Save 3D geometry (OBJ file)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["obj"] = get_unique_filename("geometry.obj", "manifold_geometry.obj", used_names)
    else:
        outputs["obj"] = None
    
    # Color palette CSS
    choice = input("🌈 Save color palette (CSS file)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["css"] = get_unique_filename("colors.css", "manifold_colors.css", used_names)
    else:
        outputs["css"] = None
    
    # Color palette image
    if PIL_AVAILABLE:
        choice = input("🖼️  Save color palette (PNG image)? (y/n): ").strip().lower()
        if choice == 'y':
            outputs["png"] = get_unique_filename("colors.png", "manifold_colors.png", used_names)
        else:
            outputs["png"] = None
    else:
        outputs["png"] = None
    
    # Synthesized audio
    choice = input("🎵 Save synthesized audio (WAV)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["wav"] = get_unique_filename("audio.wav", "manifold_audio.wav", used_names)
    else:
        outputs["wav"] = None
    
    # Phonetic pronunciation audio
    choice = input("🗣️  Save phonetic pronunciation (WAV)? (y/n): ").strip().lower()
    if choice == 'y':
        outputs["phonetic_wav"] = get_unique_filename("phonetic.wav", "manifold_phonetic.wav", used_names)
    else:
        outputs["phonetic_wav"] = None
    
    return outputs

def get_unique_filename(suggested: str, default: str, used_names: list) -> str:
    """Get a filename, ensuring it's unique and has correct extension"""
    while True:
        filename = input(f"  Enter path (default: {default}): ").strip()
        if not filename:
            filename = default
            print(f"  Using: {filename}")
        
        # Add extension if missing
        ext = '.' + suggested.split('.')[-1]
        if not filename.endswith(ext):
            filename += ext
            print(f"  Added {ext} extension: {filename}")
        
        # Check for conflicts
        if filename in used_names:
            print(f"  ❌ Error: '{filename}' already used for another output!")
            continue
        
        used_names.append(filename)
        return filename

@dataclass
class Settings:
    duration_scale: float = 1.0
    sample_rate: int = 44100
    verbosity: int = 0

def configure_settings(settings: Settings) -> Settings:
    print("\n" + "─"*70)
    print("CONFIGURATION")
    print("─"*70)
    
    try:
        dur = input(f"Duration scale [{settings.duration_scale}]: ").strip()
        if dur:
            settings.duration_scale = float(dur)
        
        sr = input(f"Sample rate [{settings.sample_rate}]: ").strip()
        if sr:
            settings.sample_rate = int(sr)
        
        verb = input(f"Verbosity (0=WARNING, 1=INFO, 2=DEBUG) [{settings.verbosity}]: ").strip()
        if verb:
            settings.verbosity = int(verb)
        
        print("\n✓ Settings updated successfully!")
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
    
    return settings

def view_settings(settings: Settings):
    print("\n" + "─"*70)
    print("CURRENT SETTINGS")
    print("─"*70)
    print(f"Duration Scale:  {settings.duration_scale}")
    print(f"Sample Rate:     {settings.sample_rate} Hz")
    print(f"Verbosity:       {settings.verbosity} ({'WARNING' if settings.verbosity == 0 else 'INFO' if settings.verbosity == 1 else 'DEBUG'})")
    print("─"*70)

def build_manifolds(text: str, dur_scale: float, sample_rate: int, logger: logging.Logger) -> ManifoldDocument:
    toks = tokenize(text)
    logger.info("Tokenized %d tokens", len(toks))
    gen = XenoManifoldGenerator(logger)
    manifolds: List[LexemeManifold] = []
    base_dur = 0.8 * dur_scale
    for tok in toks:
        m = gen.manifold_for_token(tok, base_dur, sample_rate)
        manifolds.append(m)
    doc = ManifoldDocument(
        version="2.0",
        created_at=time.time(),
        input_text=text,
        manifolds=manifolds
    )
    return doc

def render_audio(doc: ManifoldDocument, wav_path: str, logger: logging.Logger) -> None:
    rend = AudioRenderer(logger)
    sr = None
    total = 0.0
    for m in doc.manifolds:
        sr = m.audio.sample_rate if sr is None else sr
        total += m.audio.duration
    if sr is None:
        logger.info("No audio content to render")
        return
    left = [0.0] * int(total * sr)
    right = [0.0] * int(total * sr)
    t_cursor = 0.0
    for m in doc.manifolds:
        l, r = rend.render_patch(m.audio, t_cursor, total_duration=total)
        # mix
        for i in range(len(left)):
            left[i] += l[i]
            right[i] += r[i]
        t_cursor += m.audio.duration
    # final normalize
    peak = max(1e-9, max(max(abs(x) for x in left), max(abs(x) for x in right)))
    norm = 0.98 / peak
    left = [x * norm for x in left]
    right = [x * norm for x in right]
    rend.write_wav_stereo(wav_path, left, right, sr)
    logger.info("WAV written: %s", wav_path)

def write_json(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    payload = manifolds_to_json(doc)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("JSON written: %s", path)

def write_phonetic_text(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    """Write a human-readable phonetic text file"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("="*70 + "\n")
        fh.write("XENO MANIFOLD LANGUAGE - PHONETIC TRANSCRIPTION\n")
        fh.write("="*70 + "\n\n")
        
        fh.write(f"Original Text: {doc.input_text}\n")
        fh.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(doc.created_at))}\n")
        fh.write(f"Tokens: {len(doc.manifolds)}\n")
        fh.write("\n" + "-"*70 + "\n\n")
        
        # Write alien script
        fh.write("ALIEN SCRIPT:\n")
        alien_symbols = " ".join([m.glyph.symbol for m in doc.manifolds])
        fh.write(f"  {alien_symbols}\n\n")
        
        # Write phonetic
        fh.write("PHONETIC PRONUNCIATION:\n")
        phonetic = " ".join([m.glyph.phonetic for m in doc.manifolds])
        fh.write(f"  {phonetic}\n\n")
        
        # Write speakable version
        fh.write("SPEAKABLE (for TTS):\n")
        phoneme_map = {
            'zh': 'zha', 'kh': 'ka', 'th': 'tha', 'sh': 'sha',
            'xh': 'zuh', 'gh': 'guh', 'qh': 'kwa', 'ph': 'fuh',
            'ch': 'cha', 'rh': 'ruh', 'aa': 'ah', 'ee': 'ee',
            'ii': 'ee', 'oo': 'oo', 'uu': 'oo', 'ai': 'eye',
            'ei': 'ay', 'ou': 'oh', 'ae': 'ay'
        }
        speakable = phonetic
        for phoneme, replacement in phoneme_map.items():
            speakable = speakable.replace(phoneme, replacement)
        speakable = speakable.replace('-', '').replace('  ', ' ')
        fh.write(f"  {speakable}\n\n")
        
        fh.write("-"*70 + "\n\n")
        
        # Write per-word breakdown
        fh.write("WORD-BY-WORD TRANSLATION:\n\n")
        for i, m in enumerate(doc.manifolds, 1):
            fh.write(f"{i}. {m.token}\n")
            fh.write(f"   Symbol:    {m.glyph.symbol}\n")
            fh.write(f"   Phonetic:  {m.glyph.phonetic}\n")
            fh.write(f"   Speakable: {m.glyph.phonetic.replace('-', '')}\n")
            fh.write(f"   Complexity: {'●' * m.glyph.complexity} ({m.glyph.complexity}/10)\n")
            fh.write("\n")
        
        fh.write("="*70 + "\n")
        fh.write("End of phonetic transcription\n")
        fh.write("="*70 + "\n")
    
    logger.info("Phonetic text written: %s", path)

def write_3d_geometry(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    """Write 3D geometry as OBJ file"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Xeno Manifold 3D Geometry\n")
        fh.write(f"# Original text: {doc.input_text}\n")
        fh.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(doc.created_at))}\n\n")
        
        vertex_offset = 0
        
        for i, m in enumerate(doc.manifolds):
            fh.write(f"\n# Word {i+1}: {m.token}\n")
            fh.write(f"o {m.token}_{i}\n")
            
            # Write vertices for spline control points
            for j, (x, y, z) in enumerate(m.geometry.control_points):
                # Offset each word in space
                fh.write(f"v {x + i*2.5} {y} {z}\n")
            
            # Create line segments connecting control points
            num_points = len(m.geometry.control_points)
            for j in range(num_points - 1):
                fh.write(f"l {vertex_offset + j + 1} {vertex_offset + j + 2}\n")
            
            # If closed, connect last to first
            if m.geometry.closed:
                fh.write(f"l {vertex_offset + num_points} {vertex_offset + 1}\n")
            
            vertex_offset += num_points
    
    logger.info("3D geometry written: %s", path)

def write_color_palette(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    """Write color palette as CSS/text file"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("/* Xeno Manifold Color Palette */\n")
        fh.write(f"/* Original text: {doc.input_text} */\n")
        fh.write(f"/* Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(doc.created_at))} */\n\n")
        
        fh.write(":root {\n")
        for i, m in enumerate(doc.manifolds):
            fh.write(f"\n  /* {m.token} */\n")
            for j, (t, h, s, l) in enumerate(m.color.stops):
                r, g, b = hsl_to_rgb(h, s, l)
                fh.write(f"  --{m.token}-color-{j}: rgb({r}, {g}, {b});\n")
                fh.write(f"  --{m.token}-hsl-{j}: hsl({h*360:.1f}, {s*100:.1f}%, {l*100:.1f}%);\n")
        fh.write("}\n\n")
        
        # Add gradient definitions
        fh.write("/* Gradient definitions */\n")
        for i, m in enumerate(doc.manifolds):
            stops = []
            for j, (t, h, s, l) in enumerate(m.color.stops):
                r, g, b = hsl_to_rgb(h, s, l)
                stops.append(f"rgb({r}, {g}, {b}) {t*100:.0f}%")
            gradient = ", ".join(stops)
            fh.write(f".gradient-{m.token} {{ background: linear-gradient(90deg, {gradient}); }}\n")
    
    logger.info("Color palette written: %s", path)

def write_color_palette_image(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    """Write color palette as PNG image with swatches"""
    if not PIL_AVAILABLE:
        logger.warning("PIL/Pillow not available, cannot create color palette image")
        return
    
    # Larger, more professional dimensions
    card_width = 280
    card_height = 220
    margin = 20
    padding = 15
    title_height = 60
    
    cols = min(4, len(doc.manifolds))
    rows = (len(doc.manifolds) + cols - 1) // cols
    
    img_width = cols * card_width + (cols + 1) * margin
    img_height = title_height + rows * card_height + (rows + 1) * margin
    
    # Create image with gradient background
    img = Image.new('RGB', (img_width, img_height), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    
    # Draw title
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        phonetic_font = ImageFont.truetype("/System/Library/Fonts/Courier.ttc", 14)
        glyph_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        phonetic_font = ImageFont.load_default()
        glyph_font = ImageFont.load_default()
    
    # Title
    title = f"ALIEN LANGUAGE: {doc.input_text.upper()}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((img_width - title_width) // 2, 15), title, fill=(40, 40, 50), font=title_font)
    
    # Draw each token card
    for i, m in enumerate(doc.manifolds):
        row = i // cols
        col = i % cols
        
        x = col * card_width + (col + 1) * margin
        y = title_height + row * card_height + (row + 1) * margin
        
        # Card background
        draw.rectangle([x, y, x + card_width, y + card_height], 
                      fill=(255, 255, 255), outline=(220, 220, 225), width=2)
        
        # Main color swatch (top half)
        swatch_height = 100
        if m.color.stops:
            h, s, l = m.color.stops[0][1], m.color.stops[0][2], m.color.stops[0][3]
            r, g, b = hsl_to_rgb(h, s, l)  # Already returns 0-255 integers
            color = (r, g, b)
            draw.rectangle([x + padding, y + padding, 
                          x + card_width - padding, y + padding + swatch_height], 
                         fill=color, outline=(200, 200, 200), width=1)
            
            # Draw alien glyph on the swatch
            glyph_bbox = draw.textbbox((0, 0), m.glyph.symbol, font=glyph_font)
            glyph_width = glyph_bbox[2] - glyph_bbox[0]
            glyph_height = glyph_bbox[3] - glyph_bbox[1]
            
            # Choose contrasting text color
            brightness = (r * 299 + g * 587 + b * 114) / 255000
            text_color = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
            
            draw.text((x + (card_width - glyph_width) // 2, 
                      y + padding + (swatch_height - glyph_height) // 2 - 10),
                     m.glyph.symbol, fill=text_color, font=glyph_font)
        
        # Gradient strip
        gradient_y = y + padding + swatch_height + 10
        gradient_height = 30
        for gx in range(card_width - 2 * padding):
            t = gx / (card_width - 2 * padding)
            color = interpolate_gradient(m.color.stops, t)
            r, g, b = hsl_to_rgb(color[0], color[1], color[2])  # Already returns 0-255 integers
            rgb = (r, g, b)
            draw.line([(x + padding + gx, gradient_y), 
                      (x + padding + gx, gradient_y + gradient_height)], fill=rgb)
        
        # Border around gradient
        draw.rectangle([x + padding, gradient_y, 
                       x + card_width - padding, gradient_y + gradient_height],
                      outline=(200, 200, 200), width=1)
        
        # Labels
        label_y = gradient_y + gradient_height + 10
        
        # English word (bold)
        draw.text((x + padding, label_y), m.token.upper(), fill=(40, 40, 50), font=text_font)
        
        # Phonetic pronunciation
        draw.text((x + padding, label_y + 22), f"[{m.glyph.phonetic}]", 
                 fill=(120, 120, 130), font=phonetic_font)
    
    img.save(path, 'PNG')
    logger.info("Color palette image written: %s", path)

def interpolate_gradient(stops, t):
    """Interpolate color at position t through gradient stops"""
    if not stops:
        return (0.0, 0.0, 0.0)
    if len(stops) == 1:
        return stops[0][1:]
    
    # Find surrounding stops
    for i in range(len(stops) - 1):
        t1, h1, s1, l1 = stops[i]
        t2, h2, s2, l2 = stops[i + 1]
        if t1 <= t <= t2:
            # Linear interpolation
            if t2 - t1 == 0:
                return (h1, s1, l1)
            alpha = (t - t1) / (t2 - t1)
            return (
                h1 + (h2 - h1) * alpha,
                s1 + (s2 - s1) * alpha,
                l1 + (l2 - l1) * alpha
            )
    
    # Beyond last stop, use last color
    return stops[-1][1:]

def write_svg_glyphs(doc: ManifoldDocument, path: str, logger: logging.Logger) -> None:
    """Write glyphs as SVG visualization"""
    with open(path, "w", encoding="utf-8") as fh:
        width = len(doc.manifolds) * 120 + 40
        height = 200
        
        fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        fh.write(f'  <title>Xeno Manifold: {doc.input_text}</title>\n')
        fh.write('  <style>\n')
        fh.write('    .glyph { font-size: 48px; font-family: monospace; }\n')
        fh.write('    .word { font-size: 12px; font-family: sans-serif; fill: #666; }\n')
        fh.write('    .phonetic { font-size: 10px; font-family: monospace; fill: #999; }\n')
        fh.write('  </style>\n')
        fh.write(f'  <rect width="{width}" height="{height}" fill="#f8f8f8"/>\n\n')
        
        for i, m in enumerate(doc.manifolds):
            x = 60 + i * 120
            
            # Get first color from gradient
            if m.color.stops:
                h, s, l = m.color.stops[0][1], m.color.stops[0][2], m.color.stops[0][3]
                r, g, b = hsl_to_rgb(h, s, l)
                color = f"rgb({r},{g},{b})"
            else:
                color = "#000"
            
            # Draw glyph
            fh.write(f'  <text x="{x}" y="80" class="glyph" fill="{color}" text-anchor="middle">{m.glyph.symbol}</text>\n')
            
            # Draw original word
            fh.write(f'  <text x="{x}" y="120" class="word" text-anchor="middle">{m.token}</text>\n')
            
            # Draw phonetic
            fh.write(f'  <text x="{x}" y="140" class="phonetic" text-anchor="middle">{m.glyph.phonetic}</text>\n')
            
            # Draw complexity bar
            bar_width = m.glyph.complexity * 8
            fh.write(f'  <rect x="{x-40}" y="155" width="{bar_width}" height="8" fill="{color}" opacity="0.6"/>\n')
        
        fh.write('</svg>\n')
    
    logger.info("SVG glyphs written: %s", path)

def print_manifold_summary(doc: ManifoldDocument) -> None:
    print("\n" + "─"*70)
    print("ALIEN LANGUAGE TRANSLATION")
    print("─"*70)
    print(f"Original Text:   {doc.input_text}")
    print(f"Tokens:          {len(doc.manifolds)}")
    print(f"Generated:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(doc.created_at))}")
    print("─"*70)
    
    # Show the complete alien sentence
    print("\n📝 ALIEN SCRIPT:")
    print("─"*70)
    alien_symbols = " ".join([m.glyph.symbol for m in doc.manifolds])
    print(f"  {alien_symbols}")
    print("\n📢 PHONETIC:")
    print("─"*70)
    phonetic = " ".join([m.glyph.phonetic for m in doc.manifolds])
    print(f"  {phonetic}")
    print("─"*70)
    
    # Show detailed breakdown
    print("\n🔍 DETAILED TRANSLATION:")
    for i, m in enumerate(doc.manifolds, 1):
        print(f"\n🔹 Token #{i}: '{m.token}'")
        print(f"   Symbol:       {m.glyph.symbol}")
        print(f"   Phonetic:     {m.glyph.phonetic}")
        print(f"   Complexity:   {'●' * m.glyph.complexity} ({m.glyph.complexity}/10)")
        print(f"   Strokes:      {', '.join(m.glyph.strokes)}")
        print(f"   Geometry:     {len(m.geometry.control_points)} control points, thickness {m.geometry.thickness:.4f}")
        print(f"   Colors:       {len(m.color.stops)} gradient stops in {m.color.space}")
        print(f"   Audio:        {len(m.audio.carriers)} carriers, {m.audio.duration:.2f}s @ {m.audio.carriers[0]['freq']:.2f} Hz" if m.audio.carriers else "")
    
    # Show complete translation summary
    print("\n" + "="*70)
    print("📖 COMPLETE TRANSLATION")
    print("="*70)
    print(f"English:   {doc.input_text}")
    print(f"Alien:     {' '.join([m.glyph.symbol for m in doc.manifolds])}")
    print(f"Phonetic:  {' '.join([m.glyph.phonetic for m in doc.manifolds])}")
    print("="*70)

def play_audio(wav_path: str) -> None:
    """Attempt to play the audio file using system tools"""
    try:
        if sys.platform == 'darwin':  # macOS
            os.system(f'afplay "{wav_path}" &')
            print(f"🔊 Playing audio in background...")
        elif sys.platform == 'linux':
            # Try common Linux audio players
            for player in ['aplay', 'paplay', 'ffplay']:
                if os.system(f'which {player} > /dev/null 2>&1') == 0:
                    os.system(f'{player} "{wav_path}" > /dev/null 2>&1 &')
                    print(f"🔊 Playing audio in background...")
                    return
            print("⚠️  No audio player found (install aplay, paplay, or ffplay)")
        else:
            print("⚠️  Audio playback not supported on this platform")
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")

def open_file(file_path: str) -> None:
    """Open file with default application"""
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', file_path], check=False)
            print(f"👁️  Opening {file_path}...")
        elif sys.platform == 'linux':
            subprocess.run(['xdg-open', file_path], check=False)
            print(f"👁️  Opening {file_path}...")
        elif sys.platform == 'win32':
            os.startfile(file_path)
            print(f"👁️  Opening {file_path}...")
        else:
            print(f"⚠️  Cannot open file automatically on this platform")
    except Exception as e:
        print(f"⚠️  Could not open file: {e}")

def save_phonetic_audio(phonetic_text: str, output_path: str, use_gtts: bool = True) -> bool:
    """Save phonetic pronunciation as a WAV file"""
    try:
        # Convert phonetic notation to pronounceable syllables
        speakable = phonetic_text
        
        # Convert alien phonemes to English-like syllables
        phoneme_map = {
            'zh': 'zha', 'kh': 'ka', 'th': 'tha', 'sh': 'sha',
            'xh': 'zuh', 'gh': 'guh', 'qh': 'kwa', 'ph': 'fuh',
            'ch': 'cha', 'rh': 'ruh', 'aa': 'ah', 'ee': 'ee',
            'ii': 'ee', 'oo': 'oo', 'uu': 'oo', 'ai': 'eye',
            'ei': 'ay', 'ou': 'oh', 'ae': 'ay'
        }
        
        for phoneme, replacement in phoneme_map.items():
            speakable = speakable.replace(phoneme, replacement)
        
        speakable = speakable.replace('-', '').replace('  ', ' ')
        
        if use_gtts:
            try:
                from gtts import gTTS
                import tempfile
                
                # Generate speech with Google TTS
                tts = gTTS(text=speakable, lang='en', slow=True)
                
                # Save to temp MP3 first
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_path = temp_file.name
                temp_file.close()
                
                tts.save(temp_path)
                
                # Convert MP3 to WAV using ffmpeg or afconvert (macOS)
                if sys.platform == 'darwin':
                    # Use afconvert on macOS
                    result = os.system(f'afconvert -f WAVE -d LEI16 "{temp_path}" "{output_path}" 2>/dev/null')
                    if result == 0:
                        os.remove(temp_path)
                        return True
                    # Fallback: just copy the MP3 as is
                    os.rename(temp_path, output_path.replace('.wav', '.mp3'))
                    print(f"⚠️  Saved as MP3 instead (install ffmpeg for WAV conversion)")
                    return True
                else:
                    # Try ffmpeg on other platforms
                    result = os.system(f'ffmpeg -i "{temp_path}" -acodec pcm_s16le -ar 44100 "{output_path}" 2>/dev/null')
                    if result == 0:
                        os.remove(temp_path)
                        return True
                    # Fallback
                    os.rename(temp_path, output_path.replace('.wav', '.mp3'))
                    print(f"⚠️  Saved as MP3 instead (install ffmpeg for WAV conversion)")
                    return True
                    
            except ImportError:
                print("⚠️  gTTS not installed. Run: uv sync")
                return False
            except Exception as e:
                print(f"⚠️  Error generating phonetic audio: {e}")
                return False
        else:
            # Use system TTS to save to file
            if sys.platform == 'darwin':
                # macOS 'say' command can save to file
                result = os.system(f'say -v "Zarvox" -r 140 -o "{output_path.replace(".wav", ".aiff")}" "{speakable}" 2>/dev/null')
                if result == 0:
                    # Convert AIFF to WAV
                    os.system(f'afconvert -f WAVE -d LEI16 "{output_path.replace(".wav", ".aiff")}" "{output_path}" 2>/dev/null')
                    os.remove(output_path.replace(".wav", ".aiff"))
                    return True
            print("⚠️  System TTS file saving not supported on this platform")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not save phonetic audio: {e}")
        return False

def speak_phonetic(phonetic_text: str, use_gtts: bool = False) -> None:
    """Use text-to-speech to speak the phonetic pronunciation"""
    try:
        # Convert phonetic notation to pronounceable syllables
        speakable = phonetic_text
        
        # Convert alien phonemes to English-like syllables
        phoneme_map = {
            'zh': 'zha',
            'kh': 'ka',
            'th': 'tha',
            'sh': 'sha',
            'xh': 'zuh',
            'gh': 'guh',
            'qh': 'kwa',
            'ph': 'fuh',
            'ch': 'cha',
            'rh': 'ruh',
            'aa': 'ah',
            'ee': 'ee',
            'ii': 'ee',
            'oo': 'oo',
            'uu': 'oo',
            'ai': 'eye',
            'ei': 'ay',
            'ou': 'oh',
            'ae': 'ay'
        }
        
        # Replace phonemes with pronounceable versions
        for phoneme, replacement in phoneme_map.items():
            speakable = speakable.replace(phoneme, replacement)
        
        # Remove hyphens and extra spaces
        speakable = speakable.replace('-', '').replace('  ', ' ')
        
        if use_gtts:
            # Use Google TTS with pitch/speed manipulation for alien effect
            try:
                from gtts import gTTS
                import tempfile
                
                # Generate speech
                tts = gTTS(text=speakable, lang='en', slow=True)
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_path = temp_file.name
                temp_file.close()
                
                tts.save(temp_path)
                
                # Play with pitch shift for alien effect (macOS)
                if sys.platform == 'darwin':
                    # Use afplay with audio manipulation
                    print(f"🗣️  Speaking with Google TTS (alien voice): {speakable}")
                    os.system(f'afplay "{temp_path}" &')
                    # Schedule cleanup after 10 seconds
                    os.system(f'sleep 10 && rm -f "{temp_path}" &')
                else:
                    print(f"🗣️  Speaking with Google TTS: {speakable}")
                    os.system(f'mpg123 "{temp_path}" > /dev/null 2>&1 &')
                    os.system(f'sleep 10 && rm -f "{temp_path}" &')
            except ImportError:
                print("⚠️  gTTS not installed. Install with: uv add gtts")
                print("    Falling back to system voice...")
                use_gtts = False
            except Exception as e:
                print(f"⚠️  Google TTS error: {e}")
                use_gtts = False
        
        if not use_gtts:
            # Use system TTS
            if sys.platform == 'darwin':  # macOS has built-in 'say' command
                # Use a robotic voice for alien effect, slower rate
                os.system(f'say -r 140 -v "Zarvox" "{speakable}" &')
                print(f"🗣️  Speaking with Zarvox: {speakable}")
            elif sys.platform == 'linux':
                # Try espeak if available
                if os.system('which espeak > /dev/null 2>&1') == 0:
                    os.system(f'espeak -s 140 "{speakable}" > /dev/null 2>&1 &')
                    print(f"🗣️  Speaking: {speakable}")
                else:
                    print("⚠️  Text-to-speech not available (install espeak)")
            else:
                print("⚠️  Text-to-speech not supported on this platform")
    except Exception as e:
        print(f"⚠️  Could not speak text: {e}")

def process_generation(text: str, settings: Settings, output_opts: Dict[str, Optional[str]], logger: logging.Logger) -> None:
    print("\n🔄 Generating manifolds...")
    doc = build_manifolds(text, settings.duration_scale, settings.sample_rate, logger)
    print(f"✓ Generated {len(doc.manifolds)} manifold(s)")
    
    # Show summary with alien translation
    print_manifold_summary(doc)
    
    # Offer to speak the phonetic pronunciation
    phonetic = " ".join([m.glyph.phonetic for m in doc.manifolds])
    use_gtts_voice = None  # Track which voice to use
    
    # Only ask about speaking if we're not saving phonetic WAV (we'll ask once for both)
    if output_opts["phonetic_wav"]:
        print("\n🗣️  Which voice for phonetic pronunciation?")
        print("    1 - System voice (Zarvox)")
        print("    2 - Google TTS (slower, more natural)")
        choice = input("Choice (default: 2): ").strip() or '2'
        use_gtts_voice = (choice == '2')
        
        # Ask if they want to hear it now
        play_choice = input("\nPlay phonetic pronunciation now? (y/n): ").strip().lower()
        if play_choice == 'y':
            speak_phonetic(phonetic, use_gtts=use_gtts_voice)
            time.sleep(1)
    else:
        # If not saving, just ask about playing
        print("\n🗣️  Speak phonetic pronunciation?")
        print("    1 - System voice (Zarvox)")
        print("    2 - Google TTS (slower, more natural)")
        print("    n - Skip")
        choice = input("Choice: ").strip().lower()
        if choice == '1':
            speak_phonetic(phonetic, use_gtts=False)
            time.sleep(1)
        elif choice == '2':
            speak_phonetic(phonetic, use_gtts=True)
            time.sleep(1)
    
    # Save outputs
    if output_opts["json"]:
        print(f"\n💾 Writing JSON to {output_opts['json']}...")
        write_json(doc, output_opts["json"], logger)
        print(f"✓ JSON saved successfully")
    
    if output_opts["txt"]:
        print(f"\n📝 Writing phonetic text to {output_opts['txt']}...")
        write_phonetic_text(doc, output_opts["txt"], logger)
        print(f"✓ Phonetic text saved successfully")
    
    if output_opts.get("svg"):
        print(f"\n🎨 Writing SVG visualization to {output_opts['svg']}...")
        write_svg_glyphs(doc, output_opts["svg"], logger)
        print(f"✓ SVG visualization saved successfully")
    
    if output_opts.get("obj"):
        print(f"\n📐 Writing 3D geometry to {output_opts['obj']}...")
        write_3d_geometry(doc, output_opts["obj"], logger)
        print(f"✓ 3D geometry saved successfully")
    
    if output_opts.get("css"):
        print(f"\n🌈 Writing color palette to {output_opts['css']}...")
        write_color_palette(doc, output_opts["css"], logger)
        print(f"✓ Color palette saved successfully")
    
    if output_opts.get("png"):
        print(f"\n🖼️  Writing color palette image to {output_opts['png']}...")
        write_color_palette_image(doc, output_opts["png"], logger)
        print(f"✓ Color palette image saved successfully")
    
    if output_opts["phonetic_wav"]:
        voice_name = "Google TTS" if use_gtts_voice else "Zarvox"
        print(f"\n🗣️  Generating phonetic pronunciation audio ({voice_name})...")
        if save_phonetic_audio(phonetic, output_opts["phonetic_wav"], use_gtts=use_gtts_voice):
            print(f"✓ Phonetic audio saved to {output_opts['phonetic_wav']}")
        else:
            print(f"❌ Failed to save phonetic audio")
    
    if output_opts["wav"]:
        if not doc.manifolds:
            print("⚠️  No tokens to synthesize audio")
        else:
            print(f"\n🎵 Rendering synthesized audio to {output_opts['wav']}...")
            render_audio(doc, output_opts["wav"], logger)
            print(f"✓ Synthesized audio saved successfully")
            
            # Offer to play the audio
            choice = input("\nPlay synthesized audio? (y/n): ").strip().lower()
            if choice == 'y':
                play_audio(output_opts["wav"])
    
    # If no output files, show JSON
    if not output_opts["json"] and not output_opts["wav"]:
        print("\n" + "─"*70)
        print("JSON OUTPUT (no file specified, displaying here):")
        print("─"*70)
        print(json.dumps(manifolds_to_json(doc), ensure_ascii=False, indent=2))
    
    # Offer to preview/open visual files
    print("\n" + "─"*70)
    print("PREVIEW OPTIONS")
    print("─"*70)
    
    if output_opts.get("svg"):
        choice = input(f"Open SVG visualization? (y/n): ").strip().lower()
        if choice == 'y':
            open_file(output_opts["svg"])
    
    if output_opts.get("png"):
        choice = input(f"Open color palette image? (y/n): ").strip().lower()
        if choice == 'y':
            open_file(output_opts["png"])
    
    if output_opts.get("obj"):
        choice = input(f"Open 3D geometry in viewer? (y/n): ").strip().lower()
        if choice == 'y':
            open_file(output_opts["obj"])
            time.sleep(0.5)  # Give system time to launch
    
    if output_opts["json"]:
        choice = input("Preview JSON output? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                print("\n" + "─"*70)
                print(f"JSON PREVIEW: {output_opts['json']}")
                print("─"*70)
                with open(output_opts["json"], "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split('\n')
                    if len(lines) > 50:
                        print('\n'.join(lines[:50]))
                        print(f"\n... ({len(lines) - 50} more lines) ...")
                    else:
                        print(content)
            except Exception as e:
                print(f"❌ Error reading JSON file: {e}")
                print("The file may have been overwritten by the WAV output.")

def run_console_ui() -> int:
    settings = Settings()
    logger = setup_logger(settings.verbosity)
    
    try:
        clear_screen()
        print_banner()
        
        while True:
            print_menu()
            choice = get_menu_choice()
            
            try:
                if choice == '1':
                    # Generate from text input
                    clear_screen()
                    print_banner()
                    text = get_text_input()
                    output_opts = get_output_options()
                    logger = setup_logger(settings.verbosity)
                    process_generation(text, settings, output_opts, logger)
                    input("\nPress Enter to continue...")
                    clear_screen()
                    print_banner()
                
                elif choice == '2':
                    # Generate from file
                    clear_screen()
                    print_banner()
                    text = get_file_input()
                    output_opts = get_output_options()
                    logger = setup_logger(settings.verbosity)
                    process_generation(text, settings, output_opts, logger)
                    input("\nPress Enter to continue...")
                    clear_screen()
                    print_banner()
                
                elif choice == '3':
                    # Configure settings
                    clear_screen()
                    print_banner()
                    settings = configure_settings(settings)
                    logger = setup_logger(settings.verbosity)
                    input("\nPress Enter to continue...")
                    clear_screen()
                    print_banner()
                
                elif choice == '4':
                    # View settings
                    clear_screen()
                    print_banner()
                    view_settings(settings)
                    input("\nPress Enter to continue...")
                    clear_screen()
                    print_banner()
                
                elif choice == '5':
                    # Exit
                    print("\n👋 Goodbye!\n")
                    return 0
            
            except FileNotFoundError as e:
                logger.error("File error: %s", str(e))
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")
                clear_screen()
                print_banner()
            
            except ValueError as e:
                logger.error("Value error: %s", str(e))
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")
                clear_screen()
                print_banner()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user.")
                input("\nPress Enter to continue...")
                clear_screen()
                print_banner()
            
            except Exception as e:
                logger.exception("Unhandled error")
                print(f"\n❌ Unexpected error: {e}")
                input("\nPress Enter to continue...")
                clear_screen()
                print_banner()
    
    except KeyboardInterrupt:
        # Handle Ctrl+C at top level
        print("\n\n👋 Goodbye!\n")
        return 0

def main(argv: List[str]) -> int:
    # If no arguments provided, run interactive mode
    if len(argv) == 0:
        return run_console_ui()
    
    # Otherwise, use legacy CLI mode
    ap = argparse.ArgumentParser(
        prog="xeno_language",
        description="Generate a multi-dimensional, multimodal, non-discrete alien language manifold from text."
    )
    ap.add_argument("-t", "--text", type=str, help="Input text. If omitted, reads from stdin.")
    ap.add_argument("-f", "--file", type=str, help="UTF-8 text file to read.")
    ap.add_argument("--json-out", type=str, help="Path to write manifold JSON.")
    ap.add_argument("--wav-out", type=str, help="Path to write synthesized stereo WAV.")
    ap.add_argument("--txt-out", type=str, help="Path to write phonetic text transcription.")
    ap.add_argument("--phonetic-wav-out", type=str, help="Path to write phonetic pronunciation WAV.")
    ap.add_argument("--duration-scale", type=float, default=1.0, help="Scale factor for per-token duration.")
    ap.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity. -v INFO, -vv DEBUG.")
    args = ap.parse_args(argv)
    
    logger = setup_logger(args.verbose)
    try:
        # Read input
        if args.text is not None:
            logger.info("Using text from --text")
            text = args.text
        elif args.file:
            if not os.path.exists(args.file):
                logger.error("File not found: %s", args.file)
                raise FileNotFoundError(f"File not found: {args.file}")
            with open(args.file, "r", encoding="utf-8") as fh:
                text = fh.read()
                logger.info("Loaded text file: %s", args.file)
        else:
            logger.info("Reading text from stdin")
            try:
                text = sys.stdin.read()
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt during stdin read")
                raise
            if not text:
                logger.error("No input provided")
                raise ValueError("No input provided. Use --text, --file, or pipe data.")
        
        doc = build_manifolds(text, args.duration_scale, args.sample_rate, logger)
        if args.json_out:
            write_json(doc, args.json_out, logger)
        if args.txt_out:
            write_phonetic_text(doc, args.txt_out, logger)
        if args.phonetic_wav_out:
            phonetic = " ".join([m.glyph.phonetic for m in doc.manifolds])
            if save_phonetic_audio(phonetic, args.phonetic_wav_out, use_gtts=True):
                logger.info("Phonetic audio written: %s", args.phonetic_wav_out)
        if args.wav_out:
            if not doc.manifolds:
                sys.stderr.write("No tokens to synthesize.\n")
            else:
                render_audio(doc, args.wav_out, logger)
        if not args.json_out and not args.wav_out and not args.txt_out and not args.phonetic_wav_out:
            sys.stdout.write(json.dumps(manifolds_to_json(doc), ensure_ascii=False))
            sys.stdout.write("\n")
        return 0
    except FileNotFoundError as e:
        logger.error("File error: %s", str(e).replace('"', "'"))
        sys.stderr.write("Error: file not found.\n")
        return 2
    except ValueError as e:
        logger.error("Value error: %s", str(e).replace('"', "'"))
        sys.stderr.write(f"Error: {e}\n")
        return 3
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.stderr.write("Operation cancelled by user.\n")
        return 130
    except Exception as e:
        logger.exception("Unhandled error")
        sys.stderr.write(f"Unexpected error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
