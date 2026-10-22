"""Command-line entry points.

Team 2 (gloss -> sign video -> pose -> stick figure):
    s2a gloss-to-video --gloss "HELLO IX-1 NAME CHERYL" --video out.gif
    s2a lexicon-wlasl --glosses hello name my library     # build the sign database from WLASL
    s2a lexicon-folder --videos recordings/              # ... or from your own videos
    s2a lexicon-add --gloss HELLO --video hello.mp4      # add a single video
    s2a lexicon-list                                     # which glosses have a sign
    s2a demo                                             # browser demo

Full pipeline (Team 1's text -> gloss model + Team 2's visualization):
    s2a text-to-gloss --text "Where is the library?"
    s2a text-to-video --text "My name is Ivy." --video out.gif
    s2a speech-to-video --audio question.wav --video out.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
KIND_COLORS = {"sign": GREEN, "fingerspell": YELLOW, "missing": RED}


def _add_glosser_args(p):
    p.add_argument("--glosser", default="model", choices=["model", "rules"],
                   help="model = our trained Transformer, rules = baseline")
    p.add_argument("--model", default=None, help="trained model directory (default: models/text2gloss)")
    p.add_argument("--beam", type=int, default=5)


def _add_pose_args(p):
    p.add_argument("--lexicon", nargs="*", default=None,
                   help="extra lexicon directories, searched first (data/lexicon and A-Z fingerspelling are always used)")
    p.add_argument("--no-fingerspell", action="store_true", help="skip glosses with no sign instead of spelling them")
    p.add_argument("--video", help="output .gif or .mp4")
    p.add_argument("--pose", help="output .npz pose file")


def _gloss_to_outputs(gloss: str, args) -> dict:
    from s2a.gloss_to_pose import gloss_to_pose
    from s2a.gloss_to_pose.lexicon import Lexicon, default_lexicons

    result = {"gloss": gloss}
    pose, matches = gloss_to_pose(gloss, Lexicon(default_lexicons(args.lexicon)), fingerspell=not args.no_fingerspell)
    coverage = " ".join(f"{KIND_COLORS[m.kind]}{m.gloss}{RESET}" for m in matches)
    print(f"coverage ({GREEN}sign{RESET} / {YELLOW}fingerspelled{RESET} / {RED}missing{RESET}): {coverage}",
          file=sys.stderr)
    result["coverage"] = {m.gloss: m.kind for m in matches}
    if len(pose) == 0:
        print("nothing to sign: no gloss matched the lexicon", file=sys.stderr)
        return result
    if args.pose:
        pose.save(args.pose)
        result["pose"] = args.pose
    if args.video:
        from s2a.pose_to_video import write_video

        write_video(pose, args.video)
        result["video"] = args.video
    return result


def _text_to_outputs(text: str, args) -> dict:
    from s2a.text_to_gloss import load_glosser

    gloss = load_glosser(args.glosser, args.model, beam=args.beam).translate(text)
    if not (args.video or args.pose):
        return {"text": text, "gloss": gloss}
    return {"text": text, **_gloss_to_outputs(gloss, args)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="s2a", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # ----- Team 2: gloss -> pose -> stick figure -----
    p = sub.add_parser("gloss-to-video", help="ASL gloss -> sign poses -> stick-figure video")
    p.add_argument("--gloss", required=True, help='e.g. "HELLO IX-1 NAME CHERYL"')
    _add_pose_args(p)

    p = sub.add_parser("lexicon-wlasl", help="build the sign database from WLASL videos")
    p.add_argument("--glosses", nargs="*", help="words to add, e.g. hello name my (default: --top)")
    p.add_argument("--glosses-file", help="text file with one gloss per line")
    p.add_argument("--top", type=int, default=100, help="without --glosses: the N most common WLASL glosses")
    p.add_argument("--per-gloss", type=int, default=1, help="clips to keep per gloss (different signers)")
    p.add_argument("--lexicon", default=None, help="output lexicon directory (default: data/lexicon)")

    p = sub.add_parser("lexicon-folder", help="build the sign database from a folder of your own videos")
    p.add_argument("--videos", required=True, help="folder of videos; HELLO.mp4 is stored as gloss HELLO")
    p.add_argument("--labels", help="optional CSV with columns file,gloss[,frame_start,frame_end,source]")
    p.add_argument("--per-gloss", type=int, default=1)
    p.add_argument("--lexicon", default=None, help="output lexicon directory (default: data/lexicon)")

    p = sub.add_parser("lexicon-add", help="add one sign video (or .pose/.npz file) to the sign database")
    p.add_argument("--gloss", required=True)
    p.add_argument("--video", help="sign video; landmarks are extracted with MediaPipe")
    p.add_argument("--pose-file", help="existing .pose / .npz file instead of a video")
    p.add_argument("--kind", default="sign", choices=["sign", "letter"])
    p.add_argument("--source", default="", help="where the clip came from (dataset / signer, for licensing)")
    p.add_argument("--lexicon", default=None, help="lexicon directory (default: data/lexicon)")

    p = sub.add_parser("lexicon-list", help="list the glosses that have a sign")
    p.add_argument("--lexicon", nargs="*", default=None)

    p = sub.add_parser("demo", help="open the browser demo")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None, help="trained model directory (default: models/text2gloss)")
    p.add_argument("--lexicon", nargs="*", default=None, help="extra lexicon directories")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")

    # ----- full pipeline (Team 1's model in front) -----
    p = sub.add_parser("text-to-gloss", help="English text -> ASL gloss")
    p.add_argument("--text", required=True)
    _add_glosser_args(p)

    p = sub.add_parser("text-to-video", help="English text -> gloss -> pose -> video")
    p.add_argument("--text", required=True)
    _add_glosser_args(p)
    _add_pose_args(p)

    p = sub.add_parser("speech-to-video", help="audio file -> text -> gloss -> pose -> video")
    p.add_argument("--audio", required=True)
    p.add_argument("--whisper-model", default="base")
    _add_glosser_args(p)
    _add_pose_args(p)

    args = parser.parse_args(argv)

    if args.command == "gloss-to-video":
        print(json.dumps(_gloss_to_outputs(args.gloss, args), indent=2))
    elif args.command in ("lexicon-wlasl", "lexicon-folder"):
        from s2a.gloss_to_pose import build
        from s2a.gloss_to_pose.lexicon import DATA_LEXICON

        out = Path(args.lexicon) if args.lexicon else DATA_LEXICON
        if args.command == "lexicon-wlasl":
            glosses = list(args.glosses or [])
            if args.glosses_file:
                glosses += [line.strip() for line in open(args.glosses_file) if line.strip()]
            clips = build.wlasl_clips(build.load_wlasl(), glosses, args.top)
        else:
            clips = build.folder_clips(Path(args.videos), Path(args.labels) if args.labels else None)
        build.import_clips(clips, out, per_gloss=args.per_gloss)
    elif args.command == "lexicon-add":
        from s2a.gloss_to_pose.lexicon import DATA_LEXICON, add_entry
        from s2a.gloss_to_pose.pose import SignPose

        if bool(args.video) == bool(args.pose_file):
            parser.error("give exactly one of --video or --pose-file")
        if args.video:
            from s2a.gloss_to_pose.extract import video_to_pose

            pose = video_to_pose(args.video)
        else:
            pose = SignPose.load(args.pose_file)
        path = add_entry(Path(args.lexicon) if args.lexicon else DATA_LEXICON, pose, args.gloss, args.kind, args.source)
        print(f"added {args.gloss.upper()} ({len(pose)} frames) -> {path}")
    elif args.command == "lexicon-list":
        from s2a.gloss_to_pose.lexicon import Lexicon, default_lexicons

        signs = sorted(Lexicon(default_lexicons(args.lexicon)).signs)
        print(f"{len(signs)} signs" + (": " + " ".join(signs) if signs else
                                       " (only A-Z fingerspelling so far; build the database with lexicon-wlasl)"))
    elif args.command == "demo":
        from s2a.demo import serve

        serve(args.port, args.model, args.lexicon, open_browser=not args.no_browser)
    elif args.command == "text-to-gloss":
        from s2a.text_to_gloss import load_glosser

        print(load_glosser(args.glosser, args.model, beam=args.beam).translate(args.text))
    elif args.command == "text-to-video":
        print(json.dumps(_text_to_outputs(args.text, args), indent=2))
    elif args.command == "speech-to-video":
        from s2a.speech_to_text import transcribe

        text = transcribe(args.audio, args.whisper_model)
        print(f"transcript: {text}", file=sys.stderr)
        print(json.dumps(_text_to_outputs(text, args), indent=2))


if __name__ == "__main__":
    main()
