"""
Layer 6 — Model Client

A pluggable interface to any vision-language model (VLM). The system is
model-agnostic: swap UI-Venus-2, GPT-4o, Claude, or any OpenAI-compatible
endpoint by changing the config.

The model is used for:
  1. Visual analysis — "what do you see in this grid?"
  2. Reasoning — "given these entities and rules, what should I do?"
  3. Goal inference — "what is the likely objective of this game?"
  4. Rule validation — "does this observation confirm or contradict this rule?"

The model does NOT choose actions directly. It provides reasoning that the
Planner layer uses to make decisions.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import numpy as np


class ModelClient:
    """
    OpenAI-compatible API client. Works with any endpoint that implements
    /v1/chat/completions (vLLM, llama-server, OpenAI, etc.).

    Configuration via env vars or constructor:
      MODEL_API_BASE   — e.g. http://localhost:8000/v1
      MODEL_API_KEY    — API key (defaults to "EMPTY" for local servers)
      MODEL_NAME       — model name (e.g. "UI-Venus-2-9B")
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_base = api_base or os.getenv("MODEL_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("MODEL_API_KEY", "EMPTY")
        self.model_name = model_name or os.getenv("MODEL_NAME", "UI-Venus-2-9B")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def _grid_to_base64(self, grid: np.ndarray, scale: int = 8) -> str:
        """Convert grid to a base64-encoded PNG for the model's vision input."""
        try:
            from PIL import Image
            import io
        except ImportError:
            return ""

        from ..perception.perception import ARC_PALETTE
        height, width = grid.shape
        img = Image.new("RGB", (width * scale, height * scale))
        for y in range(height):
            for x in range(width):
                color = ARC_PALETTE[int(grid[y, x]) % 16]
                for dy in range(scale):
                    for dx in range(scale):
                        img.putpixel((x * scale + dx, y * scale + dy), color)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def analyze(
        self,
        grid: np.ndarray,
        context: str,
        question: str,
        include_image: bool = True,
    ) -> str:
        """
        Send a grid + context + question to the model and get a text response.

        Args:
            grid:     2D numpy array (values 0-15)
            context:  text context (memory summary, rules, etc.)
            question: what to ask the model
            include_image: whether to send the grid as an image (VLM mode)

        Returns:
            Model's text response
        """
        client = self._get_client()

        messages: list[dict[str, Any]] = []

        system_prompt = (
            "You are a reasoning engine for ARC-AGI-3, an interactive grid-world benchmark. "
            "You receive a grid of colored cells (values 0-15) and must reason about what "
            "you see: entities, patterns, movement, potential goals, and game rules. "
            "Be precise and analytical. Respond concisely.\n\n"
            f"Context:\n{context}"
        )
        messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, Any]] = []

        if include_image:
            img_b64 = self._grid_to_base64(grid)
            if img_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        # Also send the grid as text for non-vision fallback
        grid_text = self._grid_to_text(grid)
        user_content.append({
            "type": "text",
            "text": f"Current grid ({grid.shape[1]}x{grid.shape[0]}):\n{grid_text}\n\nQuestion: {question}",
        })

        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[MODEL_ERROR] {e}"

    def reason(self, context: str, question: str) -> str:
        """Text-only reasoning (no grid image) — for rule/goal reasoning."""
        client = self._get_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning engine for ARC-AGI-3 interactive games. "
                    "You analyze game observations and infer rules, goals, and strategies. "
                    "Be logical and concise. Always support your claims with evidence."
                ),
            },
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[MODEL_ERROR] {e}"

    def _grid_to_text(self, grid: np.ndarray) -> str:
        symbols = "0123456789ABCDEF"
        height, width = grid.shape
        # Truncate large grids to avoid token explosion
        max_dim = 32
        if height > max_dim or width > max_dim:
            # Subsample — take every other cell
            grid = grid[::2, ::2]
            height, width = grid.shape

        lines = []
        for y in range(height):
            lines.append("".join(symbols[int(grid[y, x]) % 16] for x in range(width)))
        return "\n".join(lines)

    @property
    def is_available(self) -> bool:
        try:
            client = self._get_client()
            return True
        except Exception:
            return False
