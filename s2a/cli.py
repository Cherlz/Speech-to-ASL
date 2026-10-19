"""Command-line entry points for the full S2A pipeline.

    s2a text-to-gloss --text "Where is the library?"
    s2a text-to-video --text "My name is Ivy." --video out.mp4
    s2a speech-to-video --audio question.wav --video out.mp4
    s2a lexicon-add --lexicon data/lexicon --gloss HELLO --video hello.mp4
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
                   help="lexicon directories, searched in order (fingerspelling is always appended)")
    p.add_argument("--no-fingerspell", action="store_true", help="skip glosses with no sign instead of spelling them")
    p.add_argument("--video", help="output .mp4 or .gif")
    p.add_argument("--pose", help="output .npz pose file")


def _text_to_outputs(text: str, args) -> dict:
    from s2a.gloss_to_pose import gloss_to_pose
    from s2a.gloss_to_pose.lexicon import DEFAULT_LEXICON, Lexicon
    from s2a.text_to_gloss import load_glosser

    gloss = load_glosser(args.glosser, args.model, beam=args.beam).translate(text)
    result = {"text": text, "gloss": gloss}
    if args.video or args.pose:
        lexicon = Lexicon([*(args.lexicon or []), DEFAULT_LEXICON])
        pose, matches = gloss_to_pose(gloss, lexicon, fingerspell=not args.no_fingerspell)
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="s2a", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

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

    p = sub.add_parser("lexicon-add", help="add a sign video (or .pose file) to a lexicon directory")
    p.add_argument("--lexicon", required=True)
    p.add_argument("--gloss", required=True)
    p.add_argument("--video", help="sign video; landmarks are extracted with MediaPipe")
    p.add_argument("--pose-file", help="existing .pose / .npz file instead of a video")
    p.add_argument("--kind", default="sign", choices=["sign", "letter"])
    p.add_argument("--source", default="", help="where the clip came from (dataset / signer, for licensing)")

    args = parser.parse_args(argv)

    if args.command == "text-to-gloss":
        from s2a.text_to_gloss import load_glosser

        print(load_glosser(args.glosser, args.model, beam=args.beam).translate(args.text))
    elif args.command == "text-to-video":
        print(json.dumps(_text_to_outputs(args.text, args), indent=2))
    elif args.command == "speech-to-video":
        from s2a.speech_to_text import transcribe

        text = transcribe(args.audio, args.whisper_model)
        print(f"transcript: {text}", file=sys.stderr)
        print(json.dumps(_text_to_outputs(text, args), indent=2))
    elif args.command == "lexicon-add":
        from s2a.gloss_to_pose.lexicon import add_entry
        from s2a.gloss_to_pose.pose import SignPose

        if bool(args.video) == bool(args.pose_file):
            parser.error("give exactly one of --video or --pose-file")
        if args.video:
            from s2a.gloss_to_pose.extract import video_to_pose

            pose = video_to_pose(args.video)
        else:
            pose = SignPose.load(args.pose_file)
        path = add_entry(Path(args.lexicon), pose, args.gloss, args.kind, args.source)
        print(f"added {args.gloss.upper()} ({len(pose)} frames) -> {path}")


if __name__ == "__main__":
    main()
