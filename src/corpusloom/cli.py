from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import OllamaClient
from .utils import extract_json_str


def _add_common_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama model name/tag (default: gpt-oss:20b)",
    )
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host base URL")
    p.add_argument(
        "--db",
        default="./.ollama_client/cache.sqlite",
        help="SQLite path for cache/corpus",
    )
    p.add_argument(
        "--keep-alive",
        default="10m",
        help='Keep model warm after call (e.g., "10m", "0" to unload)',
    )
    p.add_argument("--calls-per-minute", type=int, default=0, help="Simple rate-limit (0 = off)")

    p.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="K=V",
        help="Add/override a generation option; can repeat. Example: --opt temperature=0.2 "
        "--opt num_ctx=16384",
    )
    p.add_argument(
        "--opts-json",
        default=None,
        metavar="JSON_OR_PATH",
        help="JSON string or path to JSON file of options, merged last.",
    )


def _parse_options(args: argparse.Namespace) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    for pair in args.opt or []:
        if "=" not in pair:
            print(f"[warn] ignoring --opt value (no '='): {pair}", file=sys.stderr)
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            val: Any = v.lower() == "true"
        else:
            try:
                if "." in v:
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                if "," in v:
                    val = [x.strip() for x in v.split(",")]
                else:
                    val = v
        opts[k] = val

    if args.opts_json:
        txt = args.opts_json
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(txt)
        if not isinstance(data, dict):
            raise SystemExit("--opts-json must be a dict (JSON object)")
        opts.update(data)
    return opts


def _build_client(args: argparse.Namespace) -> OllamaClient:
    return OllamaClient(
        model=args.model,
        host=args.host,
        cache_db_path=args.db,
        default_options=_parse_options(args),
        keep_alive=args.keep_alive,
        calls_per_minute=args.calls_per_minute,
    )


def cmd_ingest(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ocp ingest", description="Ingest files (chunk + embed + store)"
    )
    _add_common_client_args(ap)
    ap.add_argument("paths", nargs="+", help="Files or globs")
    ap.add_argument(
        "--strategy",
        choices=["auto", "replace", "skip"],
        default="auto",
        help="Re-ingest policy (default: auto)",
    )
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Embedding model name")
    ap.add_argument("--encoding", default="utf-8", help="File encoding")
    args = ap.parse_args(argv)

    client = _build_client(args)
    results = client.add_files(
        args.paths,
        encoding=args.encoding,
        embed_model=args.embed_model,
        strategy=args.strategy,
    )
    print(
        json.dumps(
            {"ingested": [{"doc_id": d, "chunks": len(ch)} for d, ch in results]},
            indent=2,
        )
    )
    return 0


def cmd_search(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="ocp search", description="Search similar chunks")
    _add_common_client_args(ap)
    ap.add_argument("query", help="Search query text")
    ap.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Embedding model for query")
    args = ap.parse_args(argv)

    client = _build_client(args)
    hits = client.search_similar(args.query, embed_model=args.embed_model, top_k=args.top_k)
    print(json.dumps(hits, indent=2, ensure_ascii=False))
    return 0


def cmd_context(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ocp context",
        description="Build a stitched context block from top-k similar chunks",
    )
    _add_common_client_args(ap)
    ap.add_argument("query", help="Context query text")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=None, help="Write context to file (default stdout)")
    args = ap.parse_args(argv)

    client = _build_client(args)
    ctx = client.build_context(args.query, top_k=args.top_k, embed_model=args.embed_model)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(ctx)
        print(f"[ok] wrote context to {args.out}")
    else:
        print(ctx)
    return 0


def _read_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "prompt_file", None):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    data = sys.stdin.read()
    if not data:
        raise SystemExit("No prompt provided. Use --prompt, --prompt-file, or pipe via stdin.")
    return data


def cmd_generate(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="ocp generate", description="Single-shot generation")
    _add_common_client_args(ap)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="Prompt text")
    g.add_argument("--prompt-file", help="Read prompt from file")
    ap.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    args = ap.parse_args(argv)

    client = _build_client(args)
    prompt = _read_prompt(args)

    if args.stream:
        gen = client.generate(prompt, stream=True)
        for tok in gen:
            sys.stdout.write(tok)
            sys.stdout.flush()
        print()
    else:
        res = client.generate(prompt)
        print(res.response_text)
    return 0


def cmd_chat(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="ocp chat", description="Chat with conversation state")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_common_client_args(common)

    ap_new = sub.add_parser("new", parents=[common], help="Create a new conversation")
    ap_new.add_argument("--name", default=None)
    ap_new.add_argument("--system", default=None)
    ap_new.add_argument("--message", default=None, help="Send a first message after creating")

    ap_send = sub.add_parser("send", parents=[common], help="Send a message to a conversation")
    ap_send.add_argument("--convo-id", required=True)
    ap_send.add_argument("--message", required=True)
    ap_send.add_argument("--stream", action="store_true")

    ap_hist = sub.add_parser("history", parents=[common], help="Show conversation history")
    ap_hist.add_argument("--convo-id", required=True)

    args = ap.parse_args(argv)

    if args.subcmd == "new":
        client = _build_client(args)
        cid = client.new_conversation(name=args.name, system=args.system)
        print(json.dumps({"convo_id": cid}, indent=2))
        if args.message:
            res = client.chat(cid, args.message)
            print(res.reply.content)
        return 0

    if args.subcmd == "send":
        client = _build_client(args)
        if args.stream:
            gen = client.chat(args.convo_id, args.message, stream=True)
            for tok in gen:
                sys.stdout.write(tok)
                sys.stdout.flush()
            print()
            return 0
        else:
            res = client.chat(args.convo_id, args.message)
            print(res.reply.content)
            return 0

    if args.subcmd == "history":
        client = _build_client(args)
        msgs = client.history(args.convo_id)
        for m in msgs:
            print(f"{m.role}: {m.content}")
        return 0

    return 1


def cmd_json(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ocp json",
        description="JSON-mode generation with optional Pydantic schema validation",
    )
    _add_common_client_args(ap)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="Prompt text")
    g.add_argument("--prompt-file", help="Read prompt from file")
    ap.add_argument(
        "--schema",
        default=None,
        help='Optional schema as "module:Class" for Pydantic model',
    )
    args = ap.parse_args(argv)

    client = _build_client(args)
    prompt = _read_prompt(args)

    if args.schema:
        try:
            mod_name, cls_name = args.schema.split(":", 1)
            mod = importlib.import_module(mod_name)
            schema = getattr(mod, cls_name)
        except Exception as e:
            raise SystemExit(f"Failed to import schema {args.schema}: {e}")
        obj = client.generate_json(prompt=prompt, schema=schema)
        if hasattr(obj, "model_dump_json"):
            print(obj.model_dump_json(indent=2))
        elif hasattr(obj, "json"):
            print(obj.json(indent=2))
        else:
            print(json.dumps(obj, indent=2, default=str))
        return 0
    else:
        strict = (
            "You are a strict JSON generator. Return ONLY valid JSON. "
            "Do not include prose or code fences."
        )
        cid = client.new_conversation(system=strict)
        res = client.chat(cid, prompt)
        jtxt = extract_json_str(res.reply.content) or res.reply.content.strip()
        print(jtxt)
        return 0


def cmd_template(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="ocp template", description="Manage prompt templates")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_common_client_args(common)

    ap_add = sub.add_parser("add", parents=[common], help="Register or update a template")
    ap_add.add_argument("--name", required=True)
    ap_add.add_argument("--file", required=True)
    sub.add_parser("list", parents=[common], help="List templates")
    ap_render = sub.add_parser("render", parents=[common], help="Render a template with variables")
    ap_render.add_argument("--name", required=True)
    ap_render.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="K=V",
        help="Template variable (repeatable).",
    )

    args = ap.parse_args(argv)
    client = _build_client(args)

    if args.subcmd == "add":
        text = open(args.file, "r", encoding="utf-8").read()
        client.register_template(args.name, text)
        print(f"[ok] template '{args.name}' registered")
        return 0

    if args.subcmd == "list":
        tmpls = client.list_templates()
        print(json.dumps(tmpls, indent=2, ensure_ascii=False))
        return 0

    if args.subcmd == "render":
        vars_dict: Dict[str, Any] = {}
        for pair in args.var:
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            vars_dict[k.strip()] = v.strip()
        out = client.render_template(args.name, **vars_dict)
        print(out)
        return 0

    return 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["--help"]

    top = argparse.ArgumentParser(prog="ocp", description="Ollama Client Plus (ocp) CLI")
    sub = top.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("search")
    sub.add_parser("context")
    sub.add_parser("generate")
    sub.add_parser("chat")
    sub.add_parser("json")
    sub.add_parser("template")

    ns, rest = top.parse_known_args(argv)
    cmd = ns.cmd
    if cmd == "ingest":
        return cmd_ingest(rest)
    if cmd == "search":
        return cmd_search(rest)
    if cmd == "context":
        return cmd_context(rest)
    if cmd == "generate":
        return cmd_generate(rest)
    if cmd == "chat":
        return cmd_chat(rest)
    if cmd == "json":
        return cmd_json(rest)
    if cmd == "template":
        return cmd_template(rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
