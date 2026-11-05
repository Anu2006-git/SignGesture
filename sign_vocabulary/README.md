# Stage 5: Verified Sign Language Vocabulary Library & Communication Guide

This module powers **Stage 5: Hearing-person-to-sign-language communication** in the Two-Way Sign Language Communication System.

It translates spoken English or typed text into a sequence of verified American Sign Language (ASL) videos, provides an explicitly labelled alphabet fingerspelling fallback for unavailable words, and runs alongside the real-time hand gesture recognition system.

---

## 1. Directory Structure

```text
C:\SignGesture\sign_vocabulary\
├── config.py                  # Configurable paths & playback settings
├── manager.py                 # VocabularyManager: phrase parsing & fallback dispatch
├── player.py                  # SignVideoPlayer: video stream, card rendering, pause/replay
├── speech_listener.py         # Asynchronous microphone & Speech-to-Text engine
├── generate_default_videos.py # Generator for initial verified demonstration clips
├── vocabulary.json            # Vocabulary mapping catalog
├── videos\                    # Storage directory for verified sign video clips (.mp4)
│   ├── hello.mp4
│   ├── thank_you.mp4
│   ├── yes.mp4
│   ├── no.mp4
│   ├── please.mp4
│   ├── help.mp4
│   ├── good.mp4
│   ├── morning.mp4
│   ├── good_morning.mp4
│   ├── how_are_you.mp4
│   ├── sign.mp4
│   ├── welcome.mp4
│   ├── friend.mp4
│   ├── goodbye.mp4
│   └── name.mp4
└── alphabet\                  # Verified reference signs for fingerspelling fallback
    ├── A.jpg
    ├── B.jpg
    └── ... Z.jpg
```

---

## 2. How to Add New Verified Sign Videos

You can expand the vocabulary library at any time by adding verified `.mp4` video files.

### Step 1: Record or Acquire the Verified Sign Video

For maximum clarity and consistency, use these recommended video specifications:
- **Format**: `.mp4` (H.264 or MPEG-4 codec).
- **Resolution**: $480 \times 360$ or $640 \times 480$ (4:3 aspect ratio recommended).
- **Framerate**: $25$ or $30$ FPS.
- **Duration**: Typically $1.0$ to $2.5$ seconds for a single word or phrase.
- **Framing**: Upper body / torso and hands clearly visible against a clean, contrasting background.
- **Movement**: The sign should execute smoothly and return to a neutral rest position.

### Step 2: Place the Video in the Videos Directory

Copy the video file into:
```text
C:\SignGesture\sign_vocabulary\videos\<your_sign_name>.mp4
```
*Example:* `C:\SignGesture\sign_vocabulary\videos\water.mp4`

### Step 3: Register the Sign in `vocabulary.json`

Open `C:\SignGesture\sign_vocabulary\vocabulary.json` in any text editor and add an entry under the `"vocabulary"` object:

```json
"water": {
  "video": "water.mp4",
  "label": "WATER",
  "category": "noun",
  "verified": true,
  "description": "Index, middle, and ring fingers form a 'W' hand shape and tap twice against the chin."
}
```

#### Fields Explained:
| Field | Type | Description |
| :--- | :--- | :--- |
| **`key`** | `string` | The lowercase search keyword (e.g., `"water"` or `"thank you"`). |
| **`video`** | `string` | The exact filename inside `sign_vocabulary/videos/`. |
| **`label`** | `string` | The uppercase display name shown in the UI badge. |
| **`category`** | `string` | Category tag (e.g., `greeting`, `courtesy`, `noun`, `action`). |
| **`verified`** | `boolean` | Must be `true` to confirm linguistic accuracy against standard sign language. |
| **`description`**| `string` | Concise movement and hand shape instructions shown on the HUD. |

---

### Step 4: Adding Synonyms or Multi-Word Phrases

#### Registering Synonyms
Map multiple spoken words to the same verified sign video:
```json
"drink water": {
  "video": "water.mp4",
  "label": "WATER",
  "category": "action",
  "verified": true,
  "description": "Sign for water."
}
```

#### Multi-Word Phrases (Greedy Matching)
The `VocabularyManager` automatically matches multi-word phrases first before individual words.
For example, if you register:
```json
"nice to meet you": {
  "video": "nice_to_meet_you.mp4",
  "label": "NICE TO MEET YOU",
  "category": "greeting",
  "verified": true,
  "description": "Compound sign: Open flat hand slides across palm, followed by index fingers meeting."
}
```
When a user says *"Hello, nice to meet you"*, the system correctly segments the sentence into:
1. `HELLO` (`hello.mp4`)
2. `NICE TO MEET YOU` (`nice_to_meet_you.mp4`)

---

### Alternative: Programmatic Registration via Python

You can also add signs programmatically in your own scripts:

```python
from sign_vocabulary.manager import VocabularyManager

vm = VocabularyManager()
vm.register_sign(
    word_or_phrase="water",
    video_filename="water.mp4",
    label="WATER",
    category="noun",
    description="W-hand taps against chin twice.",
    verified=True,
)
print("Sign registered successfully!")
```

---

## 3. Fallback and Linguistic Safety Rules

To maintain linguistic integrity:
1. **Never Invent Signs**: If a word is not present in `vocabulary.json`, the system **never generates an arbitrary sign**.
2. **Explicitly Labelled Fingerspelling**: Unavailable words are automatically decomposed into their constituent letters and displayed using the verified alphabet reference library (`sign_vocabulary/alphabet/`), labelled with:  
   `⚠️ EXPLICIT FALLBACK: FINGERSPELLING: <WORD>`
3. **Missing Video Graceful Fallback**: If a video file listed in `vocabulary.json` is missing from disk, the system displays an alert and falls back to fingerspelling rather than crashing.

---

## 4. Playback Controls in the Dual-Panel Interface

While `realtime_detector.py` is running:
- **`M`**: Activate microphone speech recognition.
- **`T`**: Open interactive keyboard text entry (type prompt + press Enter).
- **`P`**: Pause / Resume sign video playback.
- **`R`**: Replay the current sign sequence from the beginning.
- **`N`**: Skip to the next sign in the sequence.
- **`Q` / `ESC`**: Exit application.
