# Aetherion 🛸
### Offline AI Assistant for Astronauts — Built for the Edge of Human Presence

> *"What happens when an astronaut needs help and there's no internet? No cloud. No API. No signal from Earth. Just silence — and whatever intelligence is onboard."*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model: Qwen2.5-3B](https://img.shields.io/badge/Model-Qwen2.5--3B-green.svg)]()
[![Platform: Raspberry Pi 5](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)]()
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

---

## What is Aetherion?

Aetherion is a **fully offline, edge-deployable AI assistant** purpose-built for astronauts. It runs on a Raspberry Pi 5, requires no internet connection, and is fine-tuned on real NASA documentation covering medical protocols, emergency procedures, ISS operations, EVA procedures, psychological support, and crew health.

The core premise is simple but important: **space has no internet**. Current AI assistants — no matter how capable — are useless the moment connectivity drops. Aetherion is designed from the ground up to work in exactly that environment.

It is not a general-purpose chatbot adapted for space. It is a domain-specific assistant trained on NASA's own documentation, designed to run on hardware that can physically fly to a space station.

---

## The Problem

Modern AI assistants have a fatal dependency: the cloud. Every major assistant — GPT, Gemini, Claude — requires a live internet connection to function. For Earth-based use, this is fine. For space operations, it is a fundamental architectural flaw.

Consider the constraints of real space missions:

- **Communication latency**: Earth-to-Mars signals take 3–22 minutes one way. Real-time AI assistance is physically impossible.
- **Bandwidth**: ISS uplink is roughly 3 Mbps shared across the entire station. Streaming LLM tokens is not viable.
- **Reliability**: Solar events, orbital geometry, and ground station windows create regular communication blackouts.
- **Autonomy requirements**: NASA's own research identifies crew autonomy as a critical capability for deep-space missions where ground support is unavailable.

The gap is real. Astronauts need intelligent assistance precisely when connectivity fails — during emergencies, system failures, medical events, and the kind of 3 AM psychological distress that doesn't wait for a ground station pass.

---

## The Original Project — Aetherion v1 (Kasaka)

The first version of this project was built with ambition that outpaced its implementation. It included:

- PyQt5 GUI with 7 tabs
- YOLOv8 object detection
- MediaPipe gesture recognition
- Vosk speech recognition
- **Groq cloud API for the core AI**

The fatal flaw: the system claimed to be an offline astronaut assistant while its entire intelligence ran on a cloud API. This was identified, documented honestly, and used as the foundation for a complete rebuild.

**Rating of v1:** 6.5/10. The vision was right. The execution had a 60–70% gap between claims and reality.

---

## The Rebuild — Aetherion v2 (Current)

The rebuild started with one rule: **no cloud dependencies for core functionality**. Everything that matters must work with the network cable unplugged.

### Architecture Decisions

| Component | v1 | v2 | Reason |
|---|---|---|---|
| Core AI | Groq cloud API | Qwen2.5-3B Q4 via llama.cpp | Space has no internet |
| Interface | PyQt5 7-tab GUI | Voice-first, terminal fallback | Astronauts don't use desktop apps |
| STT | Vosk small model | Vosk large-en-us | Technical vocabulary needs accuracy |
| TTS | pyttsx3 | Piper TTS | Faster, lighter, offline |
| Object detection | YOLOv8 all classes | Narrow distress detection only | Do one thing right |
| Target hardware | Powerful dev machine | Raspberry Pi 5 8GB from day one | Avoid v1's biggest mistake |

### The North Star Demo

> Astronaut says: *"What's the airlock depressurization procedure?"*
> Aetherion responds with the correct NASA procedure in under 5 seconds. Fully offline. No network. No API call.

---

## How It Works

### Four Operational Modes

Aetherion automatically detects context from the conversation and switches modes:

| Mode | Trigger | Behavior |
|---|---|---|
| **Emergency** | fire, leak, alarm, pressure loss, O2 drop keywords | Direct numbered steps, no preamble, maximum speed |
| **Operational** | default | Knowledgeable colleague, concise and accurate |
| **Medical** | pain, symptom, injury, medication keywords | Calm methodical responses, one question at a time |
| **Companion** | lonely, miss, family, feel, scared keywords | Warm, human, 2–4 sentences, one follow-up question |

### Sensor Integration

A background sensor system monitors simulated ISS readings — O2, CO2, pressure, temperature, humidity, and radiation — and injects live readings into every prompt. When sensors hit danger thresholds, Aetherion automatically escalates to Emergency mode without waiting for the astronaut to ask.

### Training Pipeline

The training data was built entirely from real NASA documentation:

**25 source documents across three categories:**
- *Medical*: hypoxia risk, CO2 management, EVA prebreathe protocols, microgravity physiology, VIIP syndrome, drug stability in spaceflight, telemedicine
- *Psychology*: behavioral health, isolation research, crew interactions, sleep, cognitive performance, HI-SEAS Mars analog studies
- *Operations*: ISS operations, EVA procedures, fire safety, evacuation protocols, CO2 scrubber, water recovery, oxygen generation

**Dataset generation pipeline (v3):**
1. Extract and chunk NASA PDFs (800 char chunks, 100 char overlap)
2. Run local Qwen2.5-3B on Google Colab to synthesize Q&A pairs from each chunk
3. Quality filter rejects raw chunk copies — answers must be direct responses, not document excerpts
4. Final dataset: ~1,384 high-quality synthesized pairs

**Why we generate with the same model we deploy:**
The Q&A pairs are written in a style that Qwen2.5-3B naturally understands and reproduces. No mismatch between training data language and inference behavior.

**Fine-tuning:**
- Method: LoRA via Unsloth (4x faster, 60% less VRAM)
- Base model: Qwen2.5-3B-Instruct
- Epochs: 3 | Final loss: ~1.08 | Time: ~31 min on T4
- Export: GGUF Q4_K_M for llama.cpp deployment

---

## Hardware Target

**Raspberry Pi 5 8GB**

| Spec | Value |
|---|---|
| RAM | 8GB LPDDR4X |
| CPU | Cortex-A76, 4 cores, 2.4GHz |
| Model size | ~2.0GB (Q4_K_M) |
| RAM headroom | ~6GB after model load |
| Inference speed | ~5–10 tokens/sec (estimated) |
| Power draw | ~5–8W under load |
| Form factor | Credit card sized |

The Pi 5 is physically small enough to be mission-qualified hardware. The model is small enough to fit in RAM with room to spare. The inference speed is fast enough for real-time conversation.

---

## Development Environment

- **Daily development**: WSL2 Ubuntu on Intel i5 13th gen, 16GB DDR5
- **Pi compatibility testing**: Docker ARM64 with `--memory=6g --cpus=4`
- **Fine-tuning**: Google Colab T4 GPU (free tier)
- **Final validation**: Docker ARM64 emulation before any demo

---

## Current Status

| Phase | Status |
|---|---|
| Base model serving via llama.cpp | ✅ Complete |
| Four-mode conversation system | ✅ Complete |
| Sensor simulation + alert escalation | ✅ Complete |
| NASA document dataset (25 PDFs) | ✅ Complete |
| Synthetic Q&A generation pipeline | ✅ Complete |
| Fine-tuned model (v1 dataset) | ✅ Complete — answers wrong style |
| Fine-tuned model (v3 dataset) | ✅ Complete — answers correct |
| GGUF export and deployment | 🔄 In progress |
| Voice layer (Vosk STT + Piper TTS) | ⏸️ Paused — WSL2 audio limitation |
| Distress detection (YOLOv8n) | 📋 Planned |
| Docker ARM64 validation | 📋 Planned |
| Real Pi 5 hardware testing | 📋 Pending hardware access |

---

## What Makes This Different

**1. Honest about constraints**
This project started by identifying exactly where v1 failed and why. The rebuild is built on acknowledged limitations, not marketing claims.

**2. Same model end-to-end**
The model used to generate training data, the model fine-tuned, and the model deployed are all the same architecture. This is an intentional design choice that reduces domain mismatch.

**3. Grounded in real NASA documentation**
Training data comes from actual NASA technical reports and medical standards — not synthetic space fiction or general knowledge. The model is taught what NASA actually says, not what a language model thinks NASA would say.

**4. Built for the actual hardware**
Pi 5 constraints were the design target from day one, not an afterthought. Every model choice, quantization decision, and context window size was made with 6GB effective RAM in mind.

**5. Multi-modal awareness**
Sensor readings are injected into every prompt. The model doesn't just answer questions — it answers questions in the context of current environmental conditions.

---

## Roadmap

### Near Term
- Deploy fine-tuned GGUF on WSL2, run full conversation tests
- Fix sensor context persistence across conversation turns
- Complete voice layer testing on real hardware
- Docker ARM64 validation run

### Medium Term
- Expand training dataset to remaining 1,097 NASA document chunks
- Add procedure citation — responses reference source document
- Implement confidence scoring for answer reliability
- Integrate YOLOv8n narrow distress detection (smoke, fire, suit breach)
- Real Raspberry Pi 5 hardware deployment

### Long Term
- Multi-crew support — track context per crew member
- Telemedicine mode — structured symptom collection for ground surgeon relay
- Procedure execution tracking — step-by-step checklist with confirmation
- Mars mission profile — adapted for 20+ minute communication delays
- Integration with real sensor hardware (GPIO on Pi 5)
- Potential adaptation for submarine, Antarctic, and other isolated deployment environments

---

## Why Open Source

Space exploration should not be gated behind proprietary AI systems. The same problem Aetherion solves for astronauts exists for:

- Submarine crews with communication blackout
- Antarctic research station personnel
- Remote medical workers in low-connectivity environments
- Disaster response teams operating without infrastructure

The architecture — offline fine-tuned SLM + domain-specific dataset + edge hardware — is replicable for any of these contexts. Open sourcing the full pipeline makes that possible.

---

## Repository Structure

```
aetherion/
├── src/
│   ├── main.py              # Entry point — full system
│   ├── ahlm.py              # Core query engine + mode detection
│   ├── sensors.py           # Sensor simulation + GPIO placeholder
│   ├── voice.py             # Vosk STT + Piper TTS (in progress)
│   ├── prepare_dataset.py   # PDF → chunks pipeline
│   ├── generate_qa.py       # NASA chunk → Q&A synthesis
│   └── check_dataset.py     # Dataset quality verification
├── models/                  # GGUF model files (not committed)
├── data/
│   ├── nasa_docs/           # 25 NASA PDFs (53MB)
│   └── training/            # Generated datasets
├── notebooks/
│   ├── Aetherion_Finetune_Qwen.ipynb    # Fine-tuning pipeline
│   └── Aetherion_DataGen_v3.ipynb       # Dataset generation
└── docs/
    └── architecture.md
```

---

## Built By

**Yajat Prabhakar** — solo developer, student project  
Organization: **TheLastAeron**  
GitHub: [github.com/Yajat-prabhakar/Aetherion](https://github.com/Yajat-prabhakar/Aetherion)

This project is built without institutional funding, without a team, and without access to the target hardware yet. Everything here runs on a student laptop, free-tier cloud compute, and the conviction that the problem is worth solving.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

*"The best time to build offline AI for space was before we needed it. The second best time is now."*
