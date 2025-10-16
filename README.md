# 🛸 Xeno Manifold Language Generator

A multi-dimensional, multimodal alien language generator that transforms human text into a complete alien communication system with visual glyphs, phonetic sounds, 3D geometry, colors, motion, and synthesized audio.

## 🌟 Overview

The Xeno Manifold Language is not just a simple character substitution cipher. It's a **complete alien communication system** that represents language across multiple dimensions:

- **Visual Glyphs**: Unique alien symbols with complexity ratings
- **Phonetic System**: Pronounceable alien sounds
- **3D Geometry**: Spatial representation of concepts
- **Color Gradients**: Emotional/semantic coloring
- **Motion Channels**: Temporal dynamics
- **Scalar Fields**: Dimensional density mapping
- **Audio Synthesis**: FM-synthesized alien sounds

## 🧬 Linguistic Foundation

### The Manifold Theory

The Xeno language is based on **Manifold Linguistics** - a theoretical framework where language exists not as a linear sequence of sounds or symbols, but as a multi-dimensional construct that encodes meaning across several perceptual domains simultaneously.

### Core Principles

1. **Deterministic Generation**: Uses cryptographic hashing (FNV-1a) to ensure each human word maps consistently to the same alien representation. This creates a stable, learnable vocabulary rather than random generation.

2. **Phonotactic Structure**: The phonetic system avoids human language patterns:
   - **No natural language phonemes**: Uses digraphs (zh, kh, xh) instead of single consonants
   - **Alien-sounding combinations**: Patterns like "xh-oo-zh" that feel foreign to English speakers
   - **Consistent CV (consonant-vowel) structure**: Maintains pronounceability while sounding otherworldly

3. **Logographic Writing System**: Similar to Chinese characters or Egyptian hieroglyphs, each glyph is a composite symbol:
   - **Base shape** represents the core concept (semantic radical)
   - **Accents** indicate phonetic modifications (like Korean hangul components)
   - **Connectors** show grammatical or conceptual relationships
   - **Modifiers** add contextual complexity

4. **Synesthetic Encoding**: Inspired by theories of synesthesia, where senses cross-connect:
   - **Visual ↔ Auditory**: Glyph complexity correlates with audio frequency patterns
   - **Spatial ↔ Semantic**: 3D geometry represents abstract meaning in physical space
   - **Color ↔ Emotion**: HSL gradients encode semantic/emotional tone
   - **Motion ↔ Time**: Animation channels represent temporal aspects of meaning

5. **Information Density**: Each word contains multiple layers:
   - **Surface form**: Visual glyph (what you see)
   - **Phonological form**: Sound pattern (what you hear)
   - **Geometric form**: 3D spatial structure (conceptual shape)
   - **Chromatic form**: Color progression (emotional/contextual tone)
   - **Kinetic form**: Motion over time (dynamic meaning)
   - **Field form**: Scalar density distribution (semantic weight)

### Theoretical Influences

The language design draws from:

- **Linguistic Typology**: Incorporates agglutinative features (building complexity through combination) and logographic principles (meaning-based writing)
- **Information Theory**: Uses hash functions to maximize uniqueness while maintaining consistency
- **Gestalt Psychology**: Visual glyphs follow principles of figure-ground, closure, and continuation
- **Cymatics**: Audio synthesis reflects how sound shapes matter (frequency determines form)
- **Alien Communication Hypotheses**: Based on speculation that advanced species might communicate across multiple sensory modalities simultaneously, as explored in works like:
  - Ted Chiang's "Story of Your Life" (non-linear time perception in language)
  - Arrival's Heptapod B (visual language divorced from phonetic form)
  - Solaris's living ocean (communication beyond symbolic systems)

### Why This System?

Traditional human languages are **linear** (one sound/word at a time) and **uni-modal** (primarily auditory or visual). The Xeno system explores what language might look like for beings who:

- Perceive multiple dimensions simultaneously
- Process visual, auditory, and spatial information in parallel
- Encode meaning holistically rather than sequentially
- Experience time non-linearly

This creates a language that's **learnable** (deterministic, consistent rules) yet **alien** (operates on principles foreign to human linguistic experience).

## 📖 The Language System

### Visual Writing System (Glyphs)

Each English word is transformed into a unique alien glyph composed of:

#### Base Symbols
```
◇ ○ △ □ ◎ ⬟ ⬢ ◈ ◐ ◑ ◒ ◓
```

#### Accent Marks (for vowels)
```
˙ ˋ ˊ ˆ ˜ ¨ ˇ ´ ῀ ῁
```

#### Connectors (for complex words)
```
— │ ╱ ╲ ┼ ╳ ※
```

#### Modifiers (for additional complexity)
```
⁀ ⁔ ⁖ ⁘ ⁙ ⁚ ⁛
```

**Example:**
- "hello" → `◎˙˜⁔`
- "world" → `○—ˋˊ⁖`

### Phonetic System

The alien language uses a distinct phonetic system:

#### Consonants
```
zh, kh, th, sh, xh, gh, qh, ph, ch, rh
```

#### Vowels
```
aa, ee, ii, oo, uu, ai, ei, ou, ae
```

**Example:**
- "hello" → `zh-ee-xh-xh-oo` (pronounced: "zhah-ee-zuh-zuh-oo")
- "world" → `kh-oo-rh-xh-ii` (pronounced: "kah-oo-ruh-zuh-ee")

### Glyph Complexity

Each glyph has a complexity rating (1-10) based on:
- Word length
- Number of vowels
- Number of consonants
- Presence of digits

More complex words generate more intricate glyphs with additional strokes and modifiers.

### Audio Synthesis

Each word generates a unique audio signature using:
- **FM Synthesis**: Frequency modulation for alien tones
- **Multiple Carriers**: 2-4 harmonic partials per word
- **LFO (Low-Frequency Oscillator)**: Adds vibrato and movement
- **ADSR Envelope**: Attack, Decay, Sustain, Release shaping
- **Stereo Panning**: Spatial positioning in the audio field

The base frequency is determined by the word's letter composition, ensuring consistency.

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone or navigate to the project directory
cd alien

# Sync dependencies (installs gTTS for text-to-speech)
uv sync
```

## 💻 Usage

### Interactive Mode

Simply run the program without arguments to enter interactive mode:

```bash
uv run lang.py
```

You'll see a menu with options:

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN MENU                               │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate from Text Input                                    │
│  2. Generate from File                                          │
│  3. Configure Settings                                          │
│  4. View Current Settings                                       │
│  5. Exit                                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Example Workflow

1. **Select option 1** (Generate from Text Input)
2. Type your text (e.g., "hello world")
3. Type `done` to finish input
4. Choose whether to save JSON output
5. Choose whether to save WAV audio output
6. View the alien translation:

```
📝 ALIEN SCRIPT:
◎˙˜⁔ ○—ˋˊ⁖

📢 PHONETIC:
zh-ee-xh-xh-oo  kh-oo-rh-xh-ii
```

7. Choose voice for speaking:
   - Option 1: System voice (Zarvox - robotic)
   - Option 2: Google TTS (more natural)
8. Play the synthesized alien audio

### Command-Line Mode (Advanced)

You can also use CLI arguments:

```bash
# Generate from text
uv run lang.py -t "hello world" --json-out output.json --wav-out output.wav

# Generate from file
uv run lang.py -f input.txt --json-out output.json --wav-out output.wav

# Adjust settings
uv run lang.py -t "test" --duration-scale 1.5 --sample-rate 48000 -vv
```

### Configuration Options

Access via menu option 3:

- **Duration Scale**: Multiplier for audio duration (default: 1.0)
- **Sample Rate**: Audio quality in Hz (default: 44100)
- **Verbosity**: Logging level (0=WARNING, 1=INFO, 2=DEBUG)

## 📁 Output Files

### JSON Output

Contains the complete alien language manifold:

```json
{
  "version": "2.0",
  "created_at": 1729123456.789,
  "input_text": "hello",
  "manifolds": [
    {
      "token": "hello",
      "seed": 25347132070217633,
      "glyph": {
        "symbol": "◎˙˜⁔",
        "phonetic": "zh-ee-xh-xh-oo",
        "meaning": "hello",
        "complexity": 5,
        "strokes": ["Base glyph: ◎", "2 vowel accent(s)", "Complexity modifier"]
      },
      "geometry": { ... },
      "color": { ... },
      "motion": { ... },
      "field": { ... },
      "audio": { ... }
    }
  ]
}
```

### WAV Output

A stereo audio file containing the synthesized alien speech for all words in sequence.

## 🎨 Understanding the Output

### Translation Summary

After generation, you'll see:

1. **Alien Script**: The visual glyphs
2. **Phonetic**: How to pronounce it
3. **Detailed Translation**: Per-word breakdown with:
   - Symbol and phonetic
   - Complexity rating
   - Stroke composition
   - Geometry details (3D control points)
   - Color gradient information
   - Audio synthesis parameters

### Complete Translation Box

At the end, a summary box shows the full translation:

```
======================================================================
📖 COMPLETE TRANSLATION
======================================================================
English:   hello world
Alien:     ◎˙˜⁔ ○—ˋˊ⁖
Phonetic:  zh-ee-xh-xh-oo kh-oo-rh-xh-ii
======================================================================
```

## 🎵 Audio Features

### Text-to-Speech (Phonetic)

Two options for hearing the alien phonetics:

1. **Zarvox (System Voice)**: Built-in macOS robotic voice
2. **Google TTS**: More natural voice with slow speech rate

### Synthesized Audio (Alien Sounds)

Each word generates a unique sound based on:
- Letter composition (vowels vs consonants)
- Word length
- Character hash values

The audio uses FM synthesis to create otherworldly tones that sound truly alien.

## 🔬 Technical Details

### Hash-Based Generation

The system uses a stable 64-bit FNV-1a hash function to ensure:
- **Consistency**: Same word always generates same glyph/sound
- **Uniqueness**: Different words produce different outputs
- **Determinism**: No random elements, fully reproducible

### 3D Geometry

Each word generates a 3D spline path with:
- 6-10 control points based on word complexity
- Cylindrical coordinates (r, φ, z)
- Thickness variation
- Closed/open path determination

### Color System

HSL color gradients with 3 stops:
- Hue: Derived from character hash
- Saturation: 45-90%
- Lightness: 35-75%

### Motion Channels

7 animation channels per word:
- Position (X, Y, Z)
- Rotation (X, Y, Z)
- Scale

Each with 5 keyframes and cubic bezier interpolation.

## 🐛 Troubleshooting

### "gTTS not installed" error

If you see this error, run:
```bash
uv sync
```

### No sound on Linux

Install a compatible audio player:
```bash
# For TTS
sudo apt-get install espeak

# For audio playback
sudo apt-get install mpg123 # or aplay, or paplay
```

### File extension errors

The program automatically adds `.json` and `.wav` extensions and prevents filename conflicts.

## 📝 Examples

### Simple Translation
```
Input:  "hello"
Glyph:  ◎˙˜⁔
Sound:  zh-ee-xh-xh-oo
```

### Sentence Translation
```
Input:  "take me to your leader"
Glyphs: △ˊˇ⁖ □˙ ○˜ ⬟—ˋˆ ◎˙˜˜⁔
```

### Complex Word
```
Input:  "extraterrestrial"
Glyph:  ◈—˙ˋˊˆ˜⁔⁖⁘ (high complexity: 10/10)
```

## 🎯 Use Cases

- **Creative Writing**: Generate alien dialogue for sci-fi stories
- **Game Development**: Create alien language for video games
- **Music/Art**: Use the audio synthesis for experimental music
- **Worldbuilding**: Develop consistent alien linguistics
- **Education**: Demonstrate phonetics and language structure
- **Fun**: Just explore what alien communication might sound like!

## 🤝 Contributing

The language system is deterministic and extensible. You can modify:
- `BASES`, `ACCENTS`, `CONNECTORS` in `AlienGlyphGenerator`
- `CONSONANTS`, `VOWELS` for phonetic system
- Audio synthesis parameters in `AudioPatch`
- Color/geometry generation algorithms

## 📄 License

This project is open source. Feel free to use and modify!

## 🌌 About

The Xeno Manifold Language represents a thought experiment: What if alien communication wasn't just auditory, but existed simultaneously across visual, spatial, temporal, and sonic dimensions? This generator attempts to create such a multidimensional language system where each word exists as a complete sensory manifold.

---

**Made with 🛸 for exploring alien linguistics**
