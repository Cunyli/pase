import argparse
import csv
import wave
from pathlib import Path


def collect_audio(root, extension):
    root = Path(root).expanduser().resolve()
    files = sorted(root.rglob(f"*{extension}"))
    return {path.relative_to(root): path for path in files}


def audio_info(path):
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            duration = handle.getnframes() / sample_rate
        return sample_rate, duration

    # Optional dependency: only needed when creating metadata for non-WAV audio.
    import soundfile as sf

    info = sf.info(path)
    return info.samplerate, info.duration


def main():
    parser = argparse.ArgumentParser(
        description="Create a PASE paired noisy/clean CSV from matching directory trees."
    )
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extension", default=".wav")
    args = parser.parse_args()

    noisy_files = collect_audio(args.noisy_dir, args.extension)
    clean_files = collect_audio(args.clean_dir, args.extension)
    shared_keys = sorted(set(noisy_files) & set(clean_files))
    if not shared_keys:
        raise ValueError("No matching relative paths found between noisy-dir and clean-dir")

    rows = []
    for index, key in enumerate(shared_keys):
        noisy_path = noisy_files[key]
        clean_path = clean_files[key]
        noisy_sample_rate, noisy_duration = audio_info(noisy_path)
        clean_sample_rate, clean_duration = audio_info(clean_path)
        if noisy_sample_rate != clean_sample_rate:
            sample_rate = noisy_sample_rate
        else:
            sample_rate = clean_sample_rate

        rows.append(
            {
                "uid": f"pair_{index:08d}",
                "sample_rate": sample_rate,
                "noisy_filepath": str(noisy_path),
                "clean_filepath": str(clean_path),
                "filename": str(key),
                "audio_length": min(noisy_duration, clean_duration),
            }
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired examples to {output}")


if __name__ == "__main__":
    main()
