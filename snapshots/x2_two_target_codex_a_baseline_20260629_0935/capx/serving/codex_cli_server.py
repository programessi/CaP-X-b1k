"""OpenAI-compatible wrapper around the local Codex CLI.

This server is intentionally small: it exists so CaP-X can use the same
``/chat/completions`` path while the actual code generation is handled by
``codex exec`` and the user's local Codex configuration.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "codex-a"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]


def _content_to_text(content: str | list[ContentItem] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append(item.text)
        elif item.type == "image_url":
            parts.append("[image omitted by codex_cli_server]")
    return "\n".join(parts)


def _messages_to_prompt(messages: list[Message]) -> str:
    sections = [
        "You are the code-generation backend for CaP-X.",
        "Return only the final assistant answer requested by the user prompt.",
        "Do not inspect files, do not run shell commands, and do not describe your process.",
        "For robot tasks, output executable Python code only unless the prompt explicitly asks otherwise.",
    ]
    for message in messages:
        text = _content_to_text(message.content).strip()
        if text:
            sections.append(f"\n### {message.role}\n{text}")
    return "\n".join(sections).strip() + "\n"


def _run_codex_exec(
    prompt: str,
    *,
    codex_bin: str,
    model_provider: str,
    cwd: str,
    timeout_s: float,
) -> str:
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=True) as out_file:
        cmd = [
            codex_bin,
            "-c",
            f'model_provider="{model_provider}"',
            "-a",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            cwd,
            "--output-last-message",
            out_file.name,
            "-",
        ]
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_s),
            check=False,
            env=None,
        )
        output = Path(out_file.name).read_text().strip()
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip()[-4000:]
        stdout_tail = completed.stdout.strip()[-4000:]
        raise RuntimeError(
            f"codex exec failed with rc={completed.returncode}\n"
            f"stdout:\n{stdout_tail}\n\nstderr:\n{stderr_tail}"
        )
    if output:
        return output
    fallback = completed.stdout.strip()
    if fallback:
        return fallback
    raise RuntimeError("codex exec completed but produced no final message")


def create_app(
    *,
    codex_bin: str,
    model_provider: str,
    cwd: str,
    timeout_s: float,
) -> FastAPI:
    app = FastAPI(title="Codex CLI Chat Completions Proxy", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(status_code=400, detail="codex_cli_server does not support stream=True")
        prompt = _messages_to_prompt(request.messages)
        try:
            content = _run_codex_exec(
                prompt,
                codex_bin=codex_bin,
                model_provider=model_provider,
                cwd=cwd,
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail=f"codex exec timed out after {timeout_s}s") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return ChatCompletionResponse(
            id=f"codex-cli-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model_provider": model_provider}

    return app


def main(
    host: str = "127.0.0.1",
    port: int = 8120,
    codex_bin: str = "codex",
    model_provider: str = "axonhub",
    cwd: str = ".",
    timeout_s: float = 180.0,
) -> None:
    """Start the Codex CLI chat-completions proxy."""
    app = create_app(
        codex_bin=codex_bin,
        model_provider=model_provider,
        cwd=str(Path(cwd).resolve()),
        timeout_s=float(timeout_s),
    )
    uvicorn.run(app, host=host, port=int(port))


if __name__ == "__main__":
    tyro.cli(main)
