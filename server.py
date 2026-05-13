#!/usr/bin/env python3
import asyncio
import base64
import io
import json
import os
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

if (
    VENV_PYTHON.exists()
    and sys.prefix == sys.base_prefix
    and os.environ.get("FASTVLM_SKIP_VENV_REEXEC") != "1"
):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

try:
    import edge_tts
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    edge_tts = None
    torch = None
    Image = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MODEL_PATH = os.environ.get("FASTVLM_MODEL_PATH", str(ROOT))
if "FASTVLM_MODEL_PATH" not in os.environ and (ROOT / "hf-model").exists():
    MODEL_PATH = str(ROOT / "hf-model")
HOST = os.environ.get("FASTVLM_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTVLM_PORT", "8000"))
MAX_NEW_TOKENS = int(os.environ.get("FASTVLM_MAX_NEW_TOKENS", "64"))
CREATIVE_MAX_NEW_TOKENS = int(os.environ.get("FASTVLM_CREATIVE_MAX_NEW_TOKENS", "160"))
SYSTEM_PROMPT = (
    "Você é um assistente local do FastVLM. Responda somente ao pedido atual do usuário, "
    "em português do Brasil, de forma curta e direta. Não invente fatos, detalhes, "
    "intenções, dados ou contexto que o usuário não forneceu. Se não souber, diga "
    "\"não sei\" ou peça a informação necessária. Por padrão, use no máximo 3 frases "
    "ou uma lista curta. Em pedidos criativos como poema, história, letra, mensagem "
    "ou reescrita, entregue um texto completo e natural dentro do formato pedido. "
    "Só escreva respostas longas se o usuário pedir explicitamente. "
    "Siga exatamente qualquer limite de frases, itens, versos ou formato solicitado. "
    "Não faça introduções, desculpas, metaexplicações nem diga que vai tentar."
)
VOICE_OPTIONS = [
    ("pt-BR-AntonioNeural", "Antonio Neural"),
    ("pt-BR-ThalitaMultilingualNeural", "Thalita Neural"),
    ("pt-BR-FranciscaNeural", "Francisca Neural"),
]

HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FastVLM Chat</title>
  <style>
    :root {
      --bg: #020814;
      --panel: rgba(8, 15, 28, 0.82);
      --panel-2: rgba(10, 20, 38, 0.92);
      --panel-3: rgba(6, 12, 22, 0.95);
      --text: #e8f1ff;
      --muted: #8fa6ca;
      --muted-2: #587195;
      --accent: #23c8ff;
      --accent-2: #3f7cff;
      --accent-3: #15f0b6;
      --border: rgba(87, 145, 255, 0.22);
      --border-strong: rgba(64, 157, 255, 0.42);
      --user: rgba(26, 55, 104, 0.88);
      --assistant: rgba(10, 48, 43, 0.92);
      --shadow: 0 28px 88px rgba(0, 0, 0, 0.48);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "Segoe UI", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 12%, rgba(35, 200, 255, 0.20), transparent 22%),
        radial-gradient(circle at 78% 8%, rgba(63, 124, 255, 0.22), transparent 24%),
        radial-gradient(circle at 50% 50%, rgba(21, 240, 182, 0.08), transparent 34%),
        linear-gradient(180deg, #01050d 0%, #020814 40%, #020a16 100%);
      overflow-x: hidden;
    }
    body::before,
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
    }
    body::before {
      background:
        linear-gradient(rgba(35, 200, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(35, 200, 255, 0.04) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: radial-gradient(circle at center, black 42%, transparent 100%);
      opacity: 0.34;
    }
    body::after {
      background:
        radial-gradient(circle at 50% 18%, rgba(44, 144, 255, 0.28), transparent 18%),
        radial-gradient(circle at 50% 82%, rgba(21, 240, 182, 0.10), transparent 18%);
      filter: blur(40px);
      opacity: 0.72;
    }
    .frame {
      min-height: 100vh;
      padding: 20px;
      position: relative;
      z-index: 1;
    }
    .shell {
      max-width: 1600px;
      margin: 0 auto;
      border-radius: 30px;
      border: 1px solid rgba(96, 152, 255, 0.28);
      background:
        linear-gradient(180deg, rgba(7, 13, 24, 0.96), rgba(4, 9, 18, 0.98));
      box-shadow:
        0 0 0 1px rgba(35, 200, 255, 0.08) inset,
        0 30px 80px rgba(0, 0, 0, 0.54);
      backdrop-filter: blur(18px);
      overflow: hidden;
      position: relative;
    }
    .shell::before {
      content: "";
      position: absolute;
      inset: 12px;
      border-radius: 22px;
      border: 1px solid rgba(87, 145, 255, 0.12);
      pointer-events: none;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(260px, 1.1fr) minmax(280px, 1fr) minmax(220px, 0.8fr);
      gap: 16px;
      align-items: center;
      padding: 18px 22px;
      border-bottom: 1px solid rgba(87, 145, 255, 0.16);
      background: linear-gradient(180deg, rgba(10, 20, 38, 0.92), rgba(6, 12, 24, 0.72));
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }
    .brand-mark {
      width: 58px;
      height: 58px;
      border-radius: 18px;
      background:
        radial-gradient(circle at 35% 30%, rgba(255, 255, 255, 0.42), transparent 20%),
        radial-gradient(circle at 50% 50%, rgba(35, 200, 255, 0.26), rgba(7, 16, 34, 0.95) 72%);
      border: 1px solid rgba(113, 180, 255, 0.28);
      box-shadow:
        0 0 0 1px rgba(35, 200, 255, 0.16) inset,
        0 0 28px rgba(35, 200, 255, 0.18);
      display: grid;
      place-items: center;
      color: var(--accent);
      font-size: 26px;
    }
    .brand-copy { min-width: 0; }
    .brand-copy h2 {
      margin: 0;
      font-size: clamp(1.15rem, 1.7vw, 1.55rem);
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    .brand-copy p {
      margin: 4px 0 0;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .top-center {
      text-align: center;
      padding: 4px 10px;
    }
    .top-center h1 {
      margin: 0;
      font-size: clamp(1.35rem, 2vw, 2.25rem);
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #dfeaff;
    }
    .top-center p {
      margin: 8px 0 0;
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .top-actions {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .pill,
    .icon-chip {
      border-radius: 999px;
      border: 1px solid rgba(87, 145, 255, 0.22);
      background: rgba(8, 16, 30, 0.72);
      color: var(--text);
      box-shadow: 0 0 0 1px rgba(35, 200, 255, 0.06) inset;
    }
    .pill {
      padding: 10px 14px;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #5ef5d8;
    }
    .icon-chip {
      width: 46px;
      height: 46px;
      display: grid;
      place-items: center;
      color: #cfe2ff;
      font-size: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 360px;
      gap: 18px;
      padding: 18px;
    }
    .sidebar,
    .main,
    .right-rail {
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .main {
      position: relative;
      padding-top: 26px;
    }
    .panel {
      position: relative;
      border-radius: 24px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(9, 18, 34, 0.90), rgba(5, 11, 21, 0.96));
      box-shadow:
        0 0 0 1px rgba(35, 200, 255, 0.05) inset,
        var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      z-index: 1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 18px 20px 0;
    }
    .panel-title {
      margin: 0;
      font-size: 13px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .panel-sub {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .nav-list {
      position: relative;
      z-index: 1;
      list-style: none;
      margin: 0;
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .nav-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid transparent;
      background: rgba(9, 19, 35, 0.62);
      color: #cfe2ff;
      letter-spacing: 0.03em;
    }
    .nav-item.active {
      background: linear-gradient(135deg, rgba(35, 200, 255, 0.20), rgba(63, 124, 255, 0.30));
      border-color: rgba(35, 200, 255, 0.36);
      box-shadow: 0 0 24px rgba(35, 200, 255, 0.10);
    }
    .nav-icon {
      width: 28px;
      text-align: center;
      color: var(--accent);
      font-size: 18px;
    }
    .nav-label {
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .status-grid {
      position: relative;
      z-index: 1;
      padding: 12px 18px 18px;
      display: grid;
      gap: 12px;
    }
    .status-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 6px 0;
      color: var(--muted);
      font-size: 13px;
    }
    .status-row strong {
      color: #5ef5d8;
      font-weight: 600;
    }
    .resource-list {
      position: relative;
      z-index: 1;
      padding: 8px 18px 20px;
      display: grid;
      gap: 14px;
    }
    .resource-item {
      display: grid;
      gap: 8px;
    }
    .resource-head {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }
    .meter {
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(87, 145, 255, 0.18);
      background: rgba(6, 13, 24, 0.95);
      overflow: hidden;
    }
    .meter > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
      box-shadow: 0 0 18px rgba(35, 200, 255, 0.32);
    }
    .core-panel {
      position: absolute;
      inset: 0 0 auto;
      min-height: 760px;
      display: grid;
      place-items: center;
      padding: 48px 22px 220px;
      z-index: 0;
      pointer-events: none;
      opacity: 0.92;
    }
    .core-visual {
      position: relative;
      width: min(100%, 800px);
      aspect-ratio: 16 / 10;
      display: grid;
      place-items: center;
      isolation: isolate;
      transform: translateZ(0);
    }
    .core-visual::before,
    .core-visual::after {
      content: "";
      position: absolute;
      inset: 10%;
      border-radius: 50%;
      border: 1px solid rgba(35, 200, 255, 0.18);
      box-shadow:
        0 0 0 1px rgba(35, 200, 255, 0.06) inset,
        0 0 36px rgba(35, 200, 255, 0.16);
      animation: spin 30s linear infinite;
    }
    .core-visual::after {
      inset: 16%;
      border-style: dashed;
      animation-direction: reverse;
      opacity: 0.7;
    }
    .core-orb {
      position: absolute;
      width: min(62vw, 460px);
      aspect-ratio: 1;
      border-radius: 50%;
      background:
        radial-gradient(circle at 35% 32%, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.12) 10%, transparent 18%),
        radial-gradient(circle at 62% 39%, rgba(255, 255, 255, 0.70), rgba(255, 255, 255, 0.10) 9%, transparent 16%),
        radial-gradient(circle at 50% 54%, rgba(35, 200, 255, 0.55), rgba(35, 200, 255, 0.18) 30%, rgba(7, 16, 34, 0.92) 68%);
      filter: blur(0.2px);
      box-shadow:
        0 0 0 1px rgba(35, 200, 255, 0.12) inset,
        0 0 80px rgba(35, 200, 255, 0.36);
    }
    .core-orb::before,
    .core-orb::after {
      content: "";
      position: absolute;
      inset: 9%;
      border-radius: 50%;
      border: 1px solid rgba(112, 189, 255, 0.20);
    }
    .core-orb::after {
      inset: 19%;
      border-color: rgba(21, 240, 182, 0.18);
    }
    .core-network {
      position: absolute;
      inset: 16% 18%;
      border-radius: 42% 58% 56% 44% / 44% 52% 48% 56%;
      background:
        radial-gradient(circle at 24% 34%, rgba(255, 255, 255, 0.95) 0 4px, transparent 5px),
        radial-gradient(circle at 46% 28%, rgba(255, 255, 255, 0.92) 0 4px, transparent 5px),
        radial-gradient(circle at 67% 38%, rgba(255, 255, 255, 0.90) 0 4px, transparent 5px),
        radial-gradient(circle at 33% 62%, rgba(255, 255, 255, 0.84) 0 4px, transparent 5px),
        radial-gradient(circle at 60% 62%, rgba(255, 255, 255, 0.86) 0 4px, transparent 5px),
        radial-gradient(circle at 79% 58%, rgba(255, 255, 255, 0.78) 0 4px, transparent 5px);
      filter: drop-shadow(0 0 14px rgba(35, 200, 255, 0.8));
      opacity: 0.96;
    }
    .core-connection {
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, transparent 49.7%, rgba(35, 200, 255, 0.22) 50%, transparent 50.3%),
        linear-gradient(180deg, transparent 49.5%, rgba(35, 200, 255, 0.18) 50%, transparent 50.5%);
      mask-image: radial-gradient(circle at center, black 0 44%, transparent 66%);
      opacity: 0.75;
      animation: pulse 4s ease-in-out infinite;
    }
    .core-floor {
      position: absolute;
      bottom: 2%;
      width: 84%;
      height: 20%;
      border-radius: 50%;
      background:
        radial-gradient(circle, rgba(35, 200, 255, 0.30) 0 3px, transparent 4px) center/18px 18px,
        radial-gradient(circle, rgba(35, 200, 255, 0.18) 0 2px, transparent 3px) center/42px 42px;
      filter: blur(0.3px);
      opacity: 0.88;
    }
    .core-caption {
      position: relative;
      z-index: 1;
      margin-top: 12px;
      display: grid;
      gap: 10px;
      justify-items: center;
      text-align: center;
    }
    .core-caption h3 {
      margin: 0;
      font-size: clamp(1rem, 1.4vw, 1.35rem);
      text-transform: uppercase;
      letter-spacing: 0.32em;
      color: #87d7ff;
    }
    .core-caption p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      max-width: 50ch;
    }
    .pips {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: center;
      margin-top: 2px;
    }
    .pips span {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: rgba(80, 148, 255, 0.55);
      box-shadow: 0 0 12px rgba(35, 200, 255, 0.26);
      animation: blink 1.8s ease-in-out infinite;
    }
    .pips span:nth-child(2n) { animation-delay: .15s; }
    .pips span:nth-child(3n) { animation-delay: .3s; }
    .chat-panel {
      min-height: 520px;
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      position: relative;
      z-index: 3;
      margin-top: 96px;
      background: transparent;
      backdrop-filter: none;
      box-shadow:
        0 -22px 48px rgba(0, 0, 0, 0.34),
        0 30px 88px rgba(0, 0, 0, 0.50);
    }
    .chat-panel::before { opacity: 0; }
    .split-panels {
      position: relative;
      z-index: 3;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    
    .chat {
      position: relative;
      z-index: 1;
      min-height: 420px;
      max-height: 56vh;
      overflow: auto;
      padding: 18px 20px 8px;
      scrollbar-color: rgba(78, 144, 255, 0.72) rgba(5, 12, 23, 0.7);
    }
    .chat::-webkit-scrollbar { width: 10px; }
    .chat::-webkit-scrollbar-track { background: rgba(5, 12, 23, 0.7); }
    .chat::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, rgba(35, 200, 255, 0.60), rgba(63, 124, 255, 0.80));
      border-radius: 999px;
    }
    .msg {
      display: flex;
      margin-bottom: 14px;
      animation: rise 240ms ease-out both;
    }
    .msg.user { justify-content: flex-end; }
    .msg.assistant { justify-content: flex-start; }
    .bubble {
      max-width: min(820px, 100%);
      padding: 15px 18px;
      border-radius: 18px;
      border: 1px solid rgba(87, 145, 255, 0.18);
      line-height: 1.65;
      white-space: pre-wrap;
      word-wrap: break-word;
      box-shadow: 0 0 0 1px rgba(35, 200, 255, 0.05) inset;
    }
    .msg.user .bubble {
      background: linear-gradient(135deg, rgba(26, 55, 104, 0.96), rgba(18, 33, 67, 0.98));
      border-bottom-right-radius: 6px;
    }
    .msg.assistant .bubble {
      background: linear-gradient(135deg, rgba(10, 48, 43, 0.96), rgba(8, 28, 30, 0.98));
      border-bottom-left-radius: 6px;
    }
    .composer {
      position: relative;
      z-index: 1;
      padding: 14px 16px 16px;
      border-top: 1px solid rgba(87, 145, 255, 0.16);
      background: linear-gradient(180deg, rgba(4, 9, 18, 0.18), rgba(5, 11, 20, 0.96));
    }
    .composer-shell {
      display: grid;
      gap: 12px;
      padding: 10px;
      border-radius: 22px;
      border: 1px solid rgba(87, 145, 255, 0.18);
      background: rgba(8, 15, 28, 0.74);
      box-shadow: 0 0 28px rgba(35, 200, 255, 0.05) inset;
    }
    .composer-top {
      display: grid;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 108px;
      resize: vertical;
      border-radius: 18px;
      border: 1px solid rgba(87, 145, 255, 0.18);
      background: rgba(4, 10, 20, 0.92);
      color: var(--text);
      padding: 16px 18px;
      font: inherit;
      outline: none;
      box-shadow: 0 0 0 1px rgba(35, 200, 255, 0.03) inset;
    }
    textarea::placeholder { color: #7086aa; }
    textarea:focus,
    input[type=file]:focus,
    button:focus {
      border-color: rgba(35, 200, 255, 0.55);
      box-shadow: 0 0 0 4px rgba(35, 200, 255, 0.12);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-height: 48px;
      padding: 0 14px;
      border-radius: 16px;
      border: 1px solid rgba(87, 145, 255, 0.18);
      background: rgba(4, 10, 20, 0.92);
      color: #dbe8ff;
      white-space: nowrap;
    }
    .toggle input {
      width: 18px;
      height: 18px;
      accent-color: #23c8ff;
      margin: 0;
    }
    .toggle span {
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    input[type=file] {
      width: 100%;
      flex: 1 1 260px;
      min-width: 220px;
      color: var(--muted);
      border: 1px solid rgba(87, 145, 255, 0.18);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(4, 10, 20, 0.92);
      max-width: 100%;
    }
    button {
      border: 0;
      border-radius: 16px;
      padding: 13px 18px;
      font: inherit;
      font-weight: 700;
      color: #031019;
      background: linear-gradient(135deg, #87eff0, #5a8bff 50%, #24d7ff);
      cursor: pointer;
      box-shadow: 0 16px 32px rgba(35, 200, 255, 0.18);
    }
    button.secondary {
      color: #dfeaff;
      background: linear-gradient(180deg, rgba(11, 23, 44, 0.98), rgba(7, 14, 27, 0.98));
      border: 1px solid rgba(87, 145, 255, 0.18);
      box-shadow: none;
    }
    button.mic {
      color: #dfeaff;
      background: linear-gradient(180deg, rgba(8, 36, 58, 0.98), rgba(5, 19, 31, 0.98));
      border: 1px solid rgba(35, 200, 255, 0.26);
      box-shadow: none;
    }
    button.mic.listening {
      background: linear-gradient(135deg, rgba(21, 240, 182, 0.28), rgba(35, 200, 255, 0.28));
      box-shadow: 0 0 24px rgba(35, 200, 255, 0.20);
    }
    button:disabled { opacity: 0.65; cursor: not-allowed; }
    .preview {
      display: none;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 16px;
      border: 1px solid rgba(87, 145, 255, 0.16);
      background: rgba(4, 10, 20, 0.72);
      color: var(--muted);
      font-size: 14px;
    }
    .preview img {
      width: 58px;
      height: 58px;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid rgba(87, 145, 255, 0.18);
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
      letter-spacing: 0.02em;
    }
    .compact {
      padding: 18px 20px 20px;
      min-height: 160px;
    }
    .compact h4 {
      margin: 0;
      font-size: 13px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .recent-list {
      margin: 14px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 10px;
    }
    .recent-list li {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      color: #d9e6fb;
      font-size: 14px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(87, 145, 255, 0.10);
    }
    .recent-list li span {
      color: var(--muted);
      white-space: nowrap;
    }
    .learning {
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }
    .learning p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .wave-card {
      padding: 18px 20px 20px;
    }
    .wave-card h4 {
      margin: 0 0 12px;
      font-size: 13px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .wave {
      display: flex;
      align-items: end;
      gap: 4px;
      height: 94px;
      padding: 12px 6px 6px;
      border-radius: 18px;
      border: 1px solid rgba(87, 145, 255, 0.16);
      background:
        linear-gradient(180deg, rgba(8, 15, 28, 0.66), rgba(4, 10, 20, 0.92));
      overflow: hidden;
    }
    .wave span {
      flex: 1 1 0;
      min-width: 4px;
      border-radius: 999px;
      background: linear-gradient(180deg, #8ef7f1, #3da2ff 45%, #1942c8);
      opacity: 0.92;
      box-shadow: 0 0 14px rgba(35, 200, 255, 0.24);
      animation: beat 1.8s ease-in-out infinite;
    }
    .wave.wave-alt span {
      background: linear-gradient(180deg, #5ef5d8, #3f7cff 56%, #1638b8);
    }
    .command-box,
    .reply-box {
      padding: 18px 20px 20px;
    }
    .command-box h4,
    .reply-box h4,
    .controls-box h4 {
      margin: 0 0 12px;
      font-size: 13px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .quote {
      margin: 0;
      color: #d9e6fb;
      line-height: 1.7;
      font-size: 15px;
    }
    .quote .accent {
      color: #87eff0;
    }
    .controls-box {
      padding: 18px 20px 20px;
    }
    .control-row {
      display: grid;
      gap: 9px;
      margin-bottom: 14px;
    }
    .control-label {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 13px;
    }
    .slider {
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(87, 145, 255, 0.16);
      background: rgba(4, 10, 20, 0.9);
      overflow: hidden;
    }
    .slider span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
      box-shadow: 0 0 18px rgba(35, 200, 255, 0.24);
    }
    .select-like {
      width: 100%;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(87, 145, 255, 0.16);
      background: rgba(4, 10, 20, 0.92);
      color: #dbe8ff;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      font-size: 14px;
    }
    .select-like span:last-child { color: var(--muted); }
    select.select-like {
      appearance: none;
      display: block;
      cursor: pointer;
      font: inherit;
      background-image:
        linear-gradient(45deg, transparent 50%, #7fb8ff 50%),
        linear-gradient(135deg, #7fb8ff 50%, transparent 50%);
      background-position:
        calc(100% - 18px) calc(50% - 3px),
        calc(100% - 12px) calc(50% - 3px);
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
      padding-right: 36px;
    }
    .panel-title-inline {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }
    .footer-status {
      position: relative;
      z-index: 1;
      padding: 0 18px 20px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.48; transform: scale(0.98); }
      50% { opacity: 1; transform: scale(1.02); }
    }
    @keyframes beat {
      0%, 100% { transform: scaleY(0.40); opacity: 0.58; }
      50% { transform: scaleY(1.00); opacity: 1; }
    }
    @keyframes blink {
      0%, 100% { transform: scale(0.80); opacity: 0.38; }
      50% { transform: scale(1.1); opacity: 1; }
    }
    @keyframes rise {
      from { transform: translateY(8px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    @keyframes voiceShake {
      0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
      25% { transform: translate3d(0, -6px, 0) scale(1.02); }
      50% { transform: translate3d(0, 4px, 0) scale(0.995); }
      75% { transform: translate3d(0, -3px, 0) scale(1.015); }
    }
    @keyframes voiceGlow {
      0%, 100% { opacity: 0.70; filter: drop-shadow(0 0 14px rgba(35, 200, 255, 0.55)); }
      50% { opacity: 1; filter: drop-shadow(0 0 28px rgba(21, 240, 182, 0.80)); }
    }
    .core-panel.voice-active .core-orb,
    .core-panel.listening-active .core-orb {
      animation: voiceShake 0.9s ease-in-out infinite;
    }
    .core-panel.voice-active .core-network,
    .core-panel.listening-active .core-network {
      animation: voiceGlow 0.8s ease-in-out infinite;
    }
    .core-panel.voice-active .core-visual::before,
    .core-panel.voice-active .core-visual::after,
    .core-panel.listening-active .core-visual::before,
    .core-panel.listening-active .core-visual::after {
      animation-duration: 7s;
    }
    .core-panel.voice-active .core-connection,
    .core-panel.listening-active .core-connection {
      animation-duration: 1.1s;
    }
    .core-panel.voice-active .pips span,
    .core-panel.listening-active .pips span {
      animation-duration: 0.5s;
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        scroll-behavior: auto !important;
      }
    }
    @media (max-width: 1320px) {
      .grid { grid-template-columns: 290px minmax(0, 1fr); }
      .right-rail { grid-column: 1 / -1; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 980px) {
      .topbar { grid-template-columns: 1fr; text-align: left; }
      .top-actions { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .right-rail { grid-template-columns: 1fr; }
      .split-panels { grid-template-columns: 1fr; }
      .main { padding-top: 0; }
      .core-panel {
        position: relative;
        min-height: 430px;
        padding: 28px 22px 26px;
        inset: auto;
      }
      .chat-panel { margin-top: 0; }
    }
    @media (max-width: 720px) {
      .frame { padding: 12px; }
      .shell { border-radius: 22px; }
      .topbar, .grid { padding-left: 12px; padding-right: 12px; }
      .panel, .compact, .wave-card, .command-box, .reply-box, .controls-box { border-radius: 20px; }
      .core-visual { width: 100%; aspect-ratio: 1.12 / 1; }
      .chat { max-height: 52vh; padding: 16px 14px 8px; }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="frame">
    <div class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark">◉</div>
          <div class="brand-copy">
            <h2>FastVLM Core</h2>
            <p>Interface multimodal local</p>
          </div>
        </div>
        <div class="top-center">
          <h1>Sistema de IA Visual</h1>
          <p>Chat local com texto, imagem e visual futurista</p>
        </div>
        <div class="top-actions">
          <div class="pill" id="backendBadge">Carregando backend...</div>
          <div class="icon-chip" title="Audio">◉</div>
          <div class="icon-chip" title="Configurações">⚙</div>
        </div>
      </header>

      <div class="grid">
        <aside class="sidebar">
          <section class="panel">
            <div class="panel-head">
              <div>
                <div class="panel-title">Visão geral</div>
                <p class="panel-sub">Fluxo, contexto e comandos do sistema.</p>
              </div>
            </div>
            <ul class="nav-list">
              <li class="nav-item active"><span class="nav-icon">⌁</span><span class="nav-label">Visão geral</span></li>
              <li class="nav-item"><span class="nav-icon">⌘</span><span class="nav-label">Conversas</span></li>
              <li class="nav-item"><span class="nav-icon">◷</span><span class="nav-label">Histórico</span></li>
              <li class="nav-item"><span class="nav-icon">◌</span><span class="nav-label">Análises</span></li>
              <li class="nav-item"><span class="nav-icon">⚙</span><span class="nav-label">Configurações</span></li>
            </ul>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <div class="panel-title">Status do sistema</div>
                <p class="panel-sub">Monitoramento do backend local.</p>
              </div>
            </div>
            <div class="status-grid">
              <div class="status-row"><span>Conexão</span><strong>Estável</strong></div>
              <div class="status-row"><span>Processamento IA</span><strong>Ativo</strong></div>
              <div class="status-row"><span>Modo multimodal</span><strong>Pronto</strong></div>
              <div class="status-row"><span>Resposta</span><strong id="conversationCount">0 turns</strong></div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <div class="panel-title">Recursos</div>
                <p class="panel-sub">Painel de uso visual do sistema.</p>
              </div>
            </div>
            <div class="resource-list">
              <div class="resource-item">
                <div class="resource-head"><span>CPU</span><span>32%</span></div>
                <div class="meter"><span style="width: 32%"></span></div>
              </div>
              <div class="resource-item">
                <div class="resource-head"><span>Memória</span><span>47%</span></div>
                <div class="meter"><span style="width: 47%"></span></div>
              </div>
              <div class="resource-item">
                <div class="resource-head"><span>Rede</span><span>28%</span></div>
                <div class="meter"><span style="width: 28%"></span></div>
              </div>
            </div>
          </section>
        </aside>

        <main class="main">
          <section class="panel core-panel">
            <div class="core-visual" aria-hidden="true">
              <div class="core-orb"></div>
              <div class="core-network"></div>
              <div class="core-connection"></div>
              <div class="core-floor"></div>
            </div>
            <div class="core-caption">
              <h3>IA pensando...</h3>
              <p>Converse com o modelo local, envie imagens e acompanhe a resposta em um cockpit visual inspirado em interfaces sci-fi.</p>
              <div class="pips" aria-hidden="true">
                <span></span><span></span><span></span><span></span><span></span><span></span>
                <span></span><span></span><span></span><span></span><span></span><span></span>
              </div>
            </div>
          </section>

          <section class="panel chat-panel">
            <div class="chat-head">
              <div>
                <div class="panel-title">Conversa neural</div>
                <div class="meta">Texto, imagem e resposta direta do modelo local.</div>
              </div>
              <div class="badge" id="backendBadgeInline">Aguardando backend</div>
            </div>
            <div class="chat" id="chat"></div>
            <div class="composer">
              <div class="composer-shell">
                <div class="composer-top">
                  <textarea id="prompt" placeholder="Escreva sua mensagem..."></textarea>
                  <div class="preview" id="preview"></div>
                </div>
                <div class="controls">
                  <input id="image" type="file" accept="image/*" />
                  <label class="toggle" for="audioReply">
                    <input id="audioReply" type="checkbox" />
                    <span>Audio da resposta</span>
                  </label>
                  <label class="toggle" for="voiceInput">
                    <input id="voiceInput" type="checkbox" />
                    <span>Conversa por voz</span>
                  </label>
                  <button class="secondary" id="clearBtn" type="button">Limpar conversa</button>
                  <button class="mic" id="micBtn" type="button">Iniciar voz</button>
                  <button id="sendBtn" type="button">Enviar</button>
                </div>
                <div class="status" id="status"></div>
              </div>
            </div>
            <div class="footer-status">Modo local ativo, sem dependência de nuvem.</div>
          </section>

          <div class="split-panels">
            <section class="panel compact">
              <h4>Últimas conversas</h4>
              <ul class="recent-list" id="recentList"></ul>
            </section>
            <section class="panel compact">
              <h4>Aprendizado contínuo</h4>
              <div class="learning">
                <p id="learningText">A IA está organizando o contexto desta sessão para responder melhor.</p>
                <div class="meter"><span id="learningBar" style="width: 42%"></span></div>
                <p id="learningMeta">Sessão em progresso</p>
              </div>
            </section>
          </div>
        </main>

        <aside class="right-rail">
          <section class="panel wave-card">
            <h4>Onda de voz</h4>
            <div class="wave" id="voiceWave" aria-hidden="true"></div>
          </section>

          <section class="panel command-box">
            <h4>Comando atual</h4>
            <p class="quote" id="currentCommand">Escreva uma mensagem para começar.</p>
          </section>

          <section class="panel reply-box">
            <h4>Resposta da IA</h4>
            <div class="wave wave-alt" id="replyWave" aria-hidden="true" style="height: 78px; margin-bottom: 12px;"></div>
            <p class="quote" id="assistantStatus">Aguardando entrada do usuário.</p>
          </section>

          <section class="panel controls-box">
            <h4>Controles de voz</h4>
            <div class="control-row">
              <div class="control-label"><span>Tom</span><span>85%</span></div>
              <div class="slider"><span style="width: 85%"></span></div>
            </div>
            <div class="control-row">
              <div class="control-label"><span>Velocidade</span><span>70%</span></div>
              <div class="slider"><span style="width: 70%"></span></div>
            </div>
            <div class="control-row">
              <div class="control-label"><span>Volume</span><span>90%</span></div>
              <div class="slider"><span style="width: 90%"></span></div>
            </div>
            <select class="select-like" id="voiceSelect">
              <option value="pt-BR-AntonioNeural" selected>Antonio Neural</option>
              <option value="pt-BR-ThalitaMultilingualNeural">Thalita Neural</option>
              <option value="pt-BR-FranciscaNeural">Francisca Neural</option>
            </select>
          </section>
        </aside>
      </div>
    </div>
  </div>
  <script>
    const chat = document.getElementById('chat');
    const promptEl = document.getElementById('prompt');
    const imageEl = document.getElementById('image');
    const audioReplyEl = document.getElementById('audioReply');
    const voiceInputEl = document.getElementById('voiceInput');
    const voiceSelectEl = document.getElementById('voiceSelect');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const micBtn = document.getElementById('micBtn');
    const statusEl = document.getElementById('status');
    const previewEl = document.getElementById('preview');
    const backendBadge = document.getElementById('backendBadge');
    const backendBadgeInline = document.getElementById('backendBadgeInline');
    const currentCommand = document.getElementById('currentCommand');
    const assistantStatus = document.getElementById('assistantStatus');
    const recentList = document.getElementById('recentList');
    const learningBar = document.getElementById('learningBar');
    const learningText = document.getElementById('learningText');
    const learningMeta = document.getElementById('learningMeta');
    const conversationCount = document.getElementById('conversationCount');
    const voiceWave = document.getElementById('voiceWave');
    const replyWave = document.getElementById('replyWave');
    const corePanelEl = document.querySelector('.core-panel');
    const history = [];
    const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;
    let selectedImage = null;
    let activeAudio = null;
    let recognition = null;
    let isListening = false;
    let pendingVoiceAutoSend = false;
    let manualRecognitionStop = false;
    let voiceSilenceTimer = null;
    let uiBusy = false;
    let voiceBasePrompt = '';
    let voiceFinalBuffer = '';
    let voiceInterimBuffer = '';

    function setCoreVoiceState(state) {
      corePanelEl.classList.toggle('voice-active', state === 'speaking');
      corePanelEl.classList.toggle('listening-active', state === 'listening');
    }

    function clearVoiceSilenceTimer() {
      if (voiceSilenceTimer) {
        clearTimeout(voiceSilenceTimer);
        voiceSilenceTimer = null;
      }
    }

    function composeVoicePrompt() {
      return [voiceBasePrompt, voiceFinalBuffer, voiceInterimBuffer].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
    }

    function syncVoiceUi() {
      if (!SpeechRecognitionCtor) {
        micBtn.disabled = true;
        micBtn.textContent = 'Voz indisponivel';
        return;
      }
      micBtn.disabled = uiBusy || !voiceInputEl.checked;
      micBtn.textContent = isListening ? 'Parar voz' : 'Iniciar voz';
      micBtn.classList.toggle('listening', isListening);
    }

    function ensureRecognition() {
      if (!SpeechRecognitionCtor || recognition) return recognition;
      recognition = new SpeechRecognitionCtor();
      recognition.lang = 'pt-BR';
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onstart = () => {
        isListening = true;
        setCoreVoiceState('listening');
        setStatus('Escutando sua voz...');
        syncVoiceUi();
      };

      recognition.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const transcript = event.results[i][0].transcript.trim();
          if (!transcript) continue;
          if (event.results[i].isFinal) {
            voiceFinalBuffer = `${voiceFinalBuffer} ${transcript}`.trim();
          } else {
            interim = `${interim} ${transcript}`.trim();
          }
        }
        voiceInterimBuffer = interim;
        promptEl.value = composeVoicePrompt();
        clearVoiceSilenceTimer();
        voiceSilenceTimer = window.setTimeout(() => {
          if (!promptEl.value.trim()) return;
          pendingVoiceAutoSend = true;
          stopListening(false);
          setStatus('Silencio detectado. Enviando...');
          syncAssistantState('Enviando sua voz para o modelo.');
        }, 2000);
      };

      recognition.onerror = (event) => {
        isListening = false;
        setCoreVoiceState(null);
        clearVoiceSilenceTimer();
        syncVoiceUi();
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          voiceInputEl.checked = false;
          micBtn.disabled = true;
          setStatus('Permissao de microfone negada no navegador.');
          return;
        }
        if (event.error !== 'aborted') {
          setStatus(`Falha no reconhecimento de voz: ${event.error}`);
        }
      };

      recognition.onend = () => {
        isListening = false;
        setCoreVoiceState(activeAudio ? 'speaking' : null);
        clearVoiceSilenceTimer();
        syncVoiceUi();
        if (pendingVoiceAutoSend) {
          pendingVoiceAutoSend = false;
          sendMessage();
          return;
        }
        const shouldRestart = voiceInputEl.checked && !manualRecognitionStop && !uiBusy && !activeAudio;
        manualRecognitionStop = false;
        if (shouldRestart) {
          window.setTimeout(() => startListening(), 250);
        }
      };

      return recognition;
    }

    function startListening() {
      if (!SpeechRecognitionCtor || uiBusy || activeAudio || isListening || !voiceInputEl.checked) return;
      ensureRecognition();
      voiceBasePrompt = promptEl.value.trim();
      voiceFinalBuffer = '';
      voiceInterimBuffer = '';
      manualRecognitionStop = false;
      try {
        recognition.start();
      } catch (_err) {
        syncVoiceUi();
      }
    }

    function stopListening(manual = true) {
      clearVoiceSilenceTimer();
      if (!recognition || !isListening) {
        manualRecognitionStop = manual;
        syncVoiceUi();
        return;
      }
      manualRecognitionStop = manual;
      recognition.stop();
    }

    function stopSpeech() {
      setCoreVoiceState(null);
      if (activeAudio) {
        activeAudio.pause();
        activeAudio = null;
      }
    }

    async function speakReply(text) {
      if (!audioReplyEl.checked || !text) return;
      stopSpeech();
      try {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, voice: voiceSelectEl.value }),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || 'Falha ao gerar audio');
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        activeAudio = new Audio(url);
        setCoreVoiceState('speaking');
        await new Promise((resolve, reject) => {
          activeAudio.onended = () => {
            URL.revokeObjectURL(url);
            activeAudio = null;
            setCoreVoiceState(null);
            resolve();
          };
          activeAudio.onerror = () => {
            URL.revokeObjectURL(url);
            activeAudio = null;
            setCoreVoiceState(null);
            reject(new Error('Falha ao tocar audio'));
          };
          activeAudio.play().catch(reject);
        });
      } catch (err) {
        setCoreVoiceState(null);
        setStatus(`Resposta pronta. Audio indisponivel: ${err.message}`);
      }
    }

    function buildWave(el, bars = 36) {
      if (!el) return;
      el.innerHTML = '';
      for (let i = 0; i < bars; i += 1) {
        const bar = document.createElement('span');
        const height = 18 + ((i * 17 + bars * 7) % 78);
        bar.style.height = height + '%';
        bar.style.animationDelay = (i % 9) * 0.08 + 's';
        el.appendChild(bar);
      }
    }

    function syncRecent() {
      const recent = history
        .filter(item => item.role === 'user')
        .slice(-4)
        .reverse();
      recentList.innerHTML = '';
      if (!recent.length) {
        const li = document.createElement('li');
        const left = document.createElement('strong');
        left.textContent = 'Sem interações ainda';
        const right = document.createElement('span');
        right.textContent = 'agora';
        li.appendChild(left);
        li.appendChild(right);
        recentList.appendChild(li);
      } else {
        recent.forEach((item, index) => {
          const li = document.createElement('li');
          const label = item.content || 'Mensagem vazia';
          const left = document.createElement('strong');
          left.textContent = label.slice(0, 36);
          const right = document.createElement('span');
          right.textContent = `#${index + 1}`;
          li.appendChild(left);
          li.appendChild(right);
          recentList.appendChild(li);
        });
      }
      conversationCount.textContent = `${history.filter(item => item.role === 'user').length} turns`;
      const learning = Math.min(92, 38 + history.length * 6);
      learningBar.style.width = learning + '%';
      learningMeta.textContent = `${learning}% do contexto da sessão em uso`;
      if (history.length) {
        learningText.textContent = 'O painel está combinando as últimas instruções para manter respostas mais alinhadas.';
      } else {
        learningText.textContent = 'A IA está organizando o contexto desta sessão para responder melhor.';
      }
    }

    function scrollBottom() { chat.scrollTop = chat.scrollHeight; }
    function addMessage(role, text) {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      wrap.appendChild(bubble);
      chat.appendChild(wrap);
      scrollBottom();
    }
    function setStatus(text) { statusEl.textContent = text || ''; }
    function setBusy(busy) {
      uiBusy = busy;
      sendBtn.disabled = busy;
      imageEl.disabled = busy;
      audioReplyEl.disabled = busy;
      voiceInputEl.disabled = busy;
      voiceSelectEl.disabled = busy;
      clearBtn.disabled = busy;
      promptEl.disabled = busy;
      sendBtn.textContent = busy ? 'Processando...' : 'Enviar';
      syncVoiceUi();
    }
    function renderPreview() {
      if (!selectedImage) {
        previewEl.style.display = 'none';
        previewEl.innerHTML = '';
        return;
      }
      previewEl.style.display = 'flex';
      previewEl.innerHTML = `<img src="${selectedImage.url}" alt="preview"><div>${selectedImage.name}</div>`;
    }
    function syncCommand(text) {
      const value = text || 'Escreva uma mensagem para começar.';
      currentCommand.textContent = value;
    }
    function syncAssistantState(text) {
      assistantStatus.textContent = text || 'Aguardando entrada do usuário.';
    }
    function applyBackendState(data) {
      const label = data.ready ? 'Modelo carregado' : 'Modelo indisponível';
      const extra = data.error ? ' • ' + data.error : '';
      backendBadge.textContent = label + extra;
      backendBadgeInline.textContent = data.ready ? 'ONLINE' : 'OFFLINE';
      backendBadgeInline.style.color = data.ready ? '#5ef5d8' : '#ff8e8e';
    }
    function clearConversation() {
      stopListening(true);
      stopSpeech();
      history.length = 0;
      chat.innerHTML = '';
      setStatus('');
      promptEl.value = '';
      imageEl.value = '';
      selectedImage = null;
      renderPreview();
      syncCommand('Escreva uma mensagem para começar.');
      syncAssistantState('Aguardando entrada do usuário.');
      syncRecent();
      addMessage('assistant', 'Olá! Me mande uma imagem ou escreva uma pergunta.');
    }
    async function sendMessage() {
      if (uiBusy) return;
      const text = promptEl.value.trim();
      if (!text && !selectedImage) return;
      stopListening(true);
      stopSpeech();
      addMessage('user', selectedImage ? `${text}\n\n[imagem anexada]` : text);
      history.push({ role: 'user', content: text });
      syncCommand(text || 'Imagem enviada.');
      promptEl.value = '';
      const payload = { messages: [{ role: 'user', content: text }] };
      if (selectedImage) payload.image = selectedImage.url;
      setBusy(true);
      setStatus('Chamando o modelo local...');
      syncAssistantState('Analisando dados e gerando resposta...');
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Falha na geração');
        addMessage('assistant', data.reply);
        history.push({ role: 'assistant', content: data.reply });
        await speakReply(data.reply);
        selectedImage = null;
        imageEl.value = '';
        renderPreview();
        syncRecent();
        syncAssistantState('Resposta pronta.');
        setStatus('Resposta pronta.');
      } catch (err) {
        addMessage('assistant', `Erro: ${err.message}`);
        syncAssistantState('Falha ao gerar a resposta.');
        setStatus('Não foi possível gerar a resposta.');
      } finally {
        setBusy(false);
        if (voiceInputEl.checked) startListening();
      }
    }
    imageEl.addEventListener('change', () => {
      const file = imageEl.files && imageEl.files[0];
      if (!file) {
        selectedImage = null;
        renderPreview();
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        selectedImage = { name: file.name, url: reader.result };
        renderPreview();
      };
      reader.readAsDataURL(file);
    });
    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', clearConversation);
    micBtn.addEventListener('click', () => {
      if (isListening) {
        stopListening(true);
      } else {
        voiceInputEl.checked = true;
        startListening();
      }
    });
    voiceInputEl.addEventListener('change', () => {
      if (voiceInputEl.checked) {
        startListening();
      } else {
        stopListening(true);
        setStatus('');
      }
      syncVoiceUi();
    });
    promptEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    if (!SpeechRecognitionCtor) {
      voiceInputEl.disabled = true;
      micBtn.disabled = true;
      micBtn.textContent = 'Voz indisponivel';
    } else {
      syncVoiceUi();
    }
    fetch('/health').then(r => r.json()).then(data => {
      applyBackendState(data);
      if (data.ready) {
        syncAssistantState('Sistema online e pronto para gerar respostas.');
      }
    }).catch(() => {
      backendBadge.textContent = 'Backend indisponível';
      backendBadgeInline.textContent = 'OFFLINE';
      backendBadgeInline.style.color = '#ff8e8e';
    });
    buildWave(voiceWave, 38);
    buildWave(replyWave, 34);
    clearConversation();
  </script>
</body>
</html>
"""


def _decode_data_url(data_url: str) -> bytes:
    header, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def _exact_reply_from_prompt(prompt):
    match = re.search(r"\bresponda\s+(?:apenas|somente|s[oó])\s*:\s*(.+)\s*$", prompt, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("\"'")
    return None


def _sentence_limit_from_prompt(prompt):
    match = re.search(r"\b(?:apenas|somente|s[oó])\s+(\d+)\s+frases?\b", prompt, re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    return 3


def _is_creative_request(prompt):
    normalized = prompt.strip().lower()
    creative_words = (
        r"\b(poema|poesia|verso|versos|hist[oó]ria|conto|crie|fa[cç]a|escreva|"
        r"reescreva|melhore esse texto|melhora esse texto|mensagem|legenda|letra)\b"
    )
    return bool(re.search(creative_words, normalized))


def _looks_like_unsupported_factual_question(prompt):
    normalized = prompt.strip().lower()
    if not normalized:
        return False
    if re.search(r"\b(seu|sua|você|voce|chat|modelo)\b", normalized):
        return False
    factual_words = r"\b(qual|quais|quem|quando|onde|quanto|quantos|capital|presidente|pre[cç]o|data|not[ií]cia|previs[aã]o|hoje|atual)\b"
    creative_words = r"\b(crie|fa[cç]a|escreva|poema|hist[oó]ria|resuma|traduza|explique|liste|gere)\b"
    return bool(re.search(factual_words, normalized)) and not re.search(creative_words, normalized)


def _trim_sentences(text, limit):
    parts = re.findall(r"[^.!?\n]+[.!?]?", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if len(sentences) <= limit:
        return text.strip()
    return " ".join(sentences[:limit]).strip()


def _generation_limit_for_prompt(prompt):
    return CREATIVE_MAX_NEW_TOKENS if _is_creative_request(prompt) else MAX_NEW_TOKENS


def _resolve_voice_name(voice_name):
    allowed = {name for name, _label in VOICE_OPTIONS}
    if voice_name in allowed:
        return voice_name
    return VOICE_OPTIONS[0][0]


class ModelBackend:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.error = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.dtype = None
        self._load()

    def _load(self):
        if _IMPORT_ERROR is not None:
            self.error = f"import failed: {_IMPORT_ERROR}"
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model.eval()
            self.device = next(self.model.parameters()).device
            self.dtype = next(self.model.parameters()).dtype
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def _prepare_messages(self, messages, image_data=None):
        current_user = next(
            (message for message in reversed(messages) if message.get("role") == "user"),
            {"role": "user", "content": ""},
        )
        prepared = [{"role": "system", "content": SYSTEM_PROMPT}]
        prepared.append({"role": "user", "content": current_user.get("content", "")})
        if image_data and prepared and prepared[-1]["role"] == "user":
            prepared[-1] = {
                "role": "user",
                "content": f"<image>\n{prepared[-1]['content']}".strip(),
            }
        return prepared

    def chat(self, messages, image_data=None):
        if not self.ready:
            raise RuntimeError(self.error or "model is not ready")
        with self.lock, torch.no_grad():
            rendered_messages = self._prepare_messages(messages, image_data=image_data)
            user_prompt = rendered_messages[-1]["content"].replace("<image>", "").strip()
            max_new_tokens = _generation_limit_for_prompt(user_prompt)
            exact_reply = _exact_reply_from_prompt(user_prompt)
            if exact_reply is not None and not image_data:
                return exact_reply
            if not image_data and _looks_like_unsupported_factual_question(user_prompt):
                return "Não sei confirmar isso com segurança neste modelo local. Envie uma referência, contexto ou imagem para eu responder sem inventar."

            if image_data:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                if rendered_messages[-1]["role"] != "user":
                    rendered_messages.append({"role": "user", "content": "<image>\nDescreva a imagem."})
                rendered = self.tokenizer.apply_chat_template(
                    rendered_messages, add_generation_prompt=True, tokenize=False
                )
                pre, post = rendered.split("<image>", 1)
                pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
                post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
                img_tok = torch.tensor([[-200]], dtype=pre_ids.dtype)
                input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)
                attention_mask = torch.ones_like(input_ids, device=self.device)
                px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
                px = px.to(self.device, dtype=self.dtype)
                out = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                )
                return self._postprocess_reply(
                    self._extract_assistant_reply(self.tokenizer.decode(out[0], skip_special_tokens=True)),
                    user_prompt,
                )

            rendered = self.tokenizer.apply_chat_template(
                rendered_messages, add_generation_prompt=True, tokenize=False
            )
            input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids.to(self.device)
            out = self.model.generate(
                inputs=input_ids,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
            return self._postprocess_reply(
                self._extract_assistant_reply(self.tokenizer.decode(out[0], skip_special_tokens=True)),
                user_prompt,
            )

    def _extract_assistant_reply(self, text):
        cleaned = text.strip()
        for marker in ("<|im_start|>assistant", "<|im_end|>"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[-1].strip()
        return cleaned

    def _postprocess_reply(self, reply, user_prompt):
        cleaned = reply.strip()
        cleaned = re.sub(r"^(claro|certamente|ok|sim)[,!.:\s]+", "", cleaned, flags=re.IGNORECASE).strip()
        if _is_creative_request(user_prompt):
            return cleaned
        return _trim_sentences(cleaned, _sentence_limit_from_prompt(user_prompt))

    def synthesize_speech(self, text, voice_name):
        if edge_tts is None:
            raise RuntimeError("tts is not available")
        payload = text.strip()
        if not payload:
            raise RuntimeError("text is required")
        voice = _resolve_voice_name(voice_name)

        async def _run():
            communicate = edge_tts.Communicate(payload, voice)
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)

        audio = asyncio.run(_run())
        if not audio:
            raise RuntimeError("empty audio response")
        return audio


BACKEND = ModelBackend()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        raw = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, HTML, "text/html; charset=utf-8")
            return
        if self.path == "/health":
            payload = json.dumps({"ready": BACKEND.ready, "error": BACKEND.error})
            self._send(HTTPStatus.OK, payload)
            return
        self._send(HTTPStatus.NOT_FOUND, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path not in {"/api/chat", "/api/tts"}:
            self._send(HTTPStatus.NOT_FOUND, json.dumps({"error": "not found"}))
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/tts":
            text = data.get("text") or ""
            voice = data.get("voice") or VOICE_OPTIONS[0][0]
            try:
                audio = BACKEND.synthesize_speech(text, voice)
            except Exception as exc:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(exc)}))
                return
            self._send(HTTPStatus.OK, audio, "audio/mpeg")
            return

        messages = data.get("messages") or []
        if not messages:
            self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": "messages is required"}))
            return

        image_bytes = None
        image_url = data.get("image")
        if image_url:
            try:
                image_bytes = _decode_data_url(image_url)
            except Exception as exc:
                self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": f"invalid image: {exc}"}))
                return

        try:
            reply = BACKEND.chat(messages, image_bytes)
        except Exception as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(exc)}))
            return

        self._send(HTTPStatus.OK, json.dumps({"reply": reply}))

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"FastVLM Chat running on http://{HOST}:{PORT}")
    if not BACKEND.ready:
        print(f"Model not ready: {BACKEND.error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
