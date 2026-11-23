import os
import re
import sys
import glob
import json
import wave
import sqlite3
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from num2words import num2words
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"

ABBREV_MAP = {
    "mr": ["mister"],
    "mrs": ["misses"],
    "ms": ["miz"],
    "dr": ["doctor"],
    "jr": ["junior"],
    "sr": ["senior"],
    "st": ["street"],
    "ave": ["avenue"],
    "blvd": ["boulevard"],
    "inc": ["incorporated"],
    "corp": ["corporation"],
    "llc": ["l", "l", "c"],
    "llp": ["l", "l", "p"],
    "vs": ["versus"],
    "v": ["versus"],
    "no": ["number"],
    "ex": ["exhibit"],
}


@dataclass
class MediaSegment:
    index: int              # 1-based index (1, 2, 3...)
    id_stem: str            # Filename without extension
    full_path: str          # Full path or media path string
    duration_sec: float     # Exact duration in floating-point seconds
    offset_sec: float       # Cumulative start offset in floating-point seconds


@dataclass
class WordToken:
    word_id: int
    main_index: int         # Maps back to Main.Index (1-based row in Main table)
    page_num: int
    line_num: int
    raw_word: str
    spoken_subwords: List[str] = field(default_factory=list)
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    score: float = 0.0


@dataclass
class MainRow:
    index: int                      # 1-based sequential row ID in Main
    timecode: Optional[float]       # Global floating-point seconds (None for unsynced/headers)
    page_num: int
    line_num: int                   # 0 for page headers ('PG'), 1-25 for transcript lines ('LN')
    no_display: bool
    redact: bool
    timestamp: Optional[str]        # 'HH:MM:SS' when synced, ' ' for unsynced LN, None for PG
    temp: str                       # 'PG' or 'LN'
    text: str                       # Exact original line text (preserving leading spaces/line numbers)
    native: Optional[str] = None
    is_spoken_section: bool = False


def natural_sort_key(path_str: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path_str).name)]


def resolve_video_inputs(video_args: List[str]) -> List[Path]:
    resolved = []
    for arg in video_args:
        if "*" in arg or "?" in arg:
            matches = sorted(glob.glob(arg), key=natural_sort_key)
            resolved.extend(Path(m).resolve() for m in matches)
        else:
            p = Path(arg).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Video file not found: {p}")
            resolved.append(p)
    if not resolved:
        raise ValueError("No matching video files found.")
    return resolved


def format_hhmmss(seconds: float) -> str:
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _clean_alpha_list(text: str) -> List[str]:
    text = text.replace("-", " ").replace(",", " ")
    out = []
    for w in text.split():
        clean = re.sub(r"[^a-zA-Z']", "", w).lower()
        if clean:
            out.append(clean)
    return out


def _year_or_number_to_words(num_str: str) -> List[str]:
    val = int(num_str)
    if 1900 <= val <= 2099:
        return _clean_alpha_list(num2words(val, to="year"))
    return _clean_alpha_list(num2words(val, to="cardinal"))


def expand_token_to_spoken_words(raw_token: str) -> List[str]:
    tok = raw_token.strip(" \"'“”‘’()[]{},;!?")
    if not tok:
        return []

    alpha_lower = tok.rstrip(".").lower()
    if alpha_lower in ABBREV_MAP:
        return ABBREV_MAP[alpha_lower]

    m_curr = re.match(r"^\$([\d,]+)(?:\.(\d{1,2}))?(m|b|k|million|billion|thousand)?$", tok, re.IGNORECASE)
    if m_curr:
        dollars_str = m_curr.group(1).replace(",", "")
        cents_str = m_curr.group(2)
        suffix = (m_curr.group(3) or "").lower()
        words = []
        if dollars_str:
            words.extend(_clean_alpha_list(num2words(int(dollars_str), to="cardinal")))
        if suffix in ("m", "million"):
            words.append("million")
        elif suffix in ("b", "billion"):
            words.append("billion")
        elif suffix in ("k", "thousand"):
            words.append("thousand")
        words.append("dollar" if dollars_str == "1" and not suffix else "dollars")
        if cents_str and int(cents_str) > 0:
            words.extend(["and"] + _clean_alpha_list(num2words(int(cents_str), to="cardinal")))
            words.append("cent" if int(cents_str) == 1 else "cents")
        return words

    m_date = re.match(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$", tok)
    if m_date:
        m, d, y = int(m_date.group(1)), int(m_date.group(2)), m_date.group(3)
        if 1 <= m <= 12 and 1 <= d <= 31:
            words = _clean_alpha_list(num2words(m, to="cardinal"))
            words.extend(_clean_alpha_list(num2words(d, to="cardinal")))
            if len(y) == 2:
                y_val = int(y)
                if y_val < 10:
                    words.extend(["oh"] + _clean_alpha_list(num2words(y_val)))
                else:
                    words.extend(_clean_alpha_list(num2words(y_val)))
            else:
                words.extend(_year_or_number_to_words(y))
            return words

    m_time = re.match(r"^(\d{1,2}):(\d{2})(am|pm|a\.m\.|p\.m\.)?$", tok, re.IGNORECASE)
    if m_time:
        hr, mn = int(m_time.group(1)), int(m_time.group(2))
        meridiem = (m_time.group(3) or "").replace(".", "").lower()
        if 0 <= hr <= 24 and 0 <= mn <= 59:
            words = _clean_alpha_list(num2words(hr, to="cardinal"))
            if mn == 0:
                if not meridiem:
                    words.append("oclock")
            elif mn < 10:
                words.extend(["oh"] + _clean_alpha_list(num2words(mn, to="cardinal")))
            else:
                words.extend(_clean_alpha_list(num2words(mn, to="cardinal")))
            if meridiem == "am":
                words.extend(["a", "m"])
            elif meridiem == "pm":
                words.extend(["p", "m"])
            return words

    m_pct = re.match(r"^([\d,]+(?:\.\d+)?)%$", tok)
    if m_pct:
        num_clean = m_pct.group(1).replace(",", "")
        words = _clean_alpha_list(num2words(float(num_clean) if "." in num_clean else int(num_clean)))
        words.append("percent")
        return words

    m_ord = re.match(r"^(\d+)(st|nd|rd|th)$", tok, re.IGNORECASE)
    if m_ord:
        return _clean_alpha_list(num2words(int(m_ord.group(1)), to="ordinal"))

    m_int = re.match(r"^[\d,]+$", tok)
    if m_int:
        digits = tok.replace(",", "")
        if digits.isdigit():
            return _year_or_number_to_words(digits)

    m_dec = re.match(r"^(\d+)\.(\d+)$", tok)
    if m_dec:
        return _clean_alpha_list(num2words(float(tok)))

    parts = re.findall(r"[a-zA-Z']+|\d+", tok)
    if parts:
        words = []
        for p in parts:
            if p.isdigit():
                words.extend(_year_or_number_to_words(p))
            else:
                clean_p = re.sub(r"[^a-zA-Z']", "", p).lower()
                if clean_p:
                    words.append(clean_p)
        return words

    return []


def parse_deposition_for_oncue(file_path: Path) -> Tuple[List[MainRow], List[WordToken], int, int]:
    """
    Parses a legal deposition ASCII transcript into OnCue/Sanction MainRow records:
    - Preserves exact original text (including leading indentation & line numbers) in Main.Text.
    - Marks Page Header lines with Temp='PG', LineNum=0, Timestamp=None.
    - Marks numbered lines (1-25) with Temp='LN', LineNum=1..25.
    - Automatically detects where spoken testimony begins (e.g. 'THE VIDEOGRAPHER:', 'EXAMINATION',
      'WHEREUPON', 'sworn', or 'Q.') so cover/appearance pages stay unsynced just like Sanction!
    """
    raw_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

    page_re = re.compile(r"^\s*(?:Page\s+)?(\d{1,4})\s*$", re.IGNORECASE)
    line_re = re.compile(r"^(\s*(\d{1,2})\b(.*)$)")
    qa_re = re.compile(r"^(Q\.?|A\.?|QUESTION:?|ANSWER:?)\s+(.*)$", re.IGNORECASE)
    speaker_re = re.compile(
        r"^\s*((?:MR\.|MS\.|MRS\.|DR\.|THE\s+WITNESS|THE\s+VIDEOGRAPHER|THE\s+COURT|THE\s+REPORTER|BY\s+MR\.|BY\s+MS\.)[A-Z\s\.\-]{1,35}):\s*(.*)$"
    )
    speech_start_trigger = re.compile(
        r"(THE\s+VIDEOGRAPHER\s*:|VIDEOGRAPHER\s*:|on\s+the\s+record|having\s+been\s+first\s+duly\s+sworn|EXAMINATION\s+BY|^\s*\d{1,2}\s+Q\.)",
        re.IGNORECASE
    )
    paren_re = re.compile(r"\((?:[^)]*?(?:off the record|recess|marked|exhibit|playing|concluded|stenographically)[^)]*?)\)", re.IGNORECASE)

    current_page = 1
    max_page = 1
    max_line_per_page = 25
    speech_started = False

    main_rows: List[MainRow] = []
    word_tokens: List[WordToken] = []
    row_counter = 0
    word_counter = 0

    for raw in raw_lines:
        clean_raw = raw.replace("\f", "")
        stripped = clean_raw.strip()

        # Skip completely blank lines outside of 1-25 numbered blocks
        if not stripped:
            continue

        # Detect Page Header line (Temp = 'PG', LineNum = 0)
        m_page = page_re.match(stripped)
        if m_page and int(m_page.group(1)) >= current_page:
            current_page = int(m_page.group(1))
            max_page = max(max_page, current_page)
            row_counter += 1
            main_rows.append(
                MainRow(
                    index=row_counter,
                    timecode=None,
                    page_num=current_page,
                    line_num=0,
                    no_display=False,
                    redact=False,
                    timestamp=None,
                    temp="PG",
                    text=clean_raw,
                    native=None,
                    is_spoken_section=False,
                )
            )
            continue

        # Detect numbered line 1..25 (Temp = 'LN')
        m_line = line_re.match(clean_raw)
        if m_line and 1 <= int(m_line.group(2)) <= 25:
            line_num = int(m_line.group(2))
            max_line_per_page = max(max_line_per_page, line_num)
            content = m_line.group(3).strip()
        else:
            # Unnumbered line—skip or attach as Line 0 header
            continue

        # Check if spoken record has begun (skips cover page, caption, and appearances)
        if not speech_started and speech_start_trigger.search(clean_raw):
            speech_started = True

        row_counter += 1
        m_row = MainRow(
            index=row_counter,
            timecode=None,
            page_num=current_page,
            line_num=line_num,
            no_display=False,
            redact=False,
            timestamp=" ",
            temp="LN",
            text=clean_raw,
            native=None,
            is_spoken_section=speech_started,
        )
        main_rows.append(m_row)

        if not speech_started:
            continue

        # Strip parentheticals, speaker labels, and Q/A markers for acoustic alignment
        spoken_candidate = paren_re.sub(" ", content).strip()

        m_spk = speaker_re.match(spoken_candidate)
        if m_spk:
            spoken_candidate = m_spk.group(2).strip()

        m_qa = qa_re.match(spoken_candidate)
        if m_qa:
            spoken_candidate = m_qa.group(2).strip()

        for raw_w in spoken_candidate.split():
            subwords = expand_token_to_spoken_words(raw_w)
            if subwords:
                word_counter += 1
                word_tokens.append(
                    WordToken(
                        word_id=word_counter,
                        main_index=row_counter,
                        page_num=current_page,
                        line_num=line_num,
                        raw_word=raw_w,
                        spoken_subwords=subwords,
                    )
                )

    # Fallback: if no trigger phrase matched (e.g. non-standard transcript), align all numbered lines
    if not speech_started and main_rows:
        print("Notice: Standard videographer/Q&A header not detected; aligning all numbered lines.")
        for m_row in main_rows:
            if m_row.temp == "LN":
                m_row.is_spoken_section = True
                m_line = line_re.match(m_row.text)
                content = m_line.group(3).strip() if m_line else m_row.text
                for raw_w in content.split():
                    subwords = expand_token_to_spoken_words(raw_w)
                    if subwords:
                        word_counter += 1
                        word_tokens.append(
                            WordToken(
                                word_id=word_counter,
                                main_index=m_row.index,
                                page_num=m_row.page_num,
                                line_num=m_row.line_num,
                                raw_word=raw_w,
                                spoken_subwords=subwords,
                            )
                        )

    return main_rows, word_tokens, max_page, max_line_per_page


def extract_and_concat_16k_wav(video_paths: List[Path], combined_wav_path: Path) -> List[MediaSegment]:
    segments: List[MediaSegment] = []
    cumulative_samples = 0
    sample_rate = 16000

    print(f"[1/4] Extracting & joining 16kHz mono audio across {len(video_paths)} video file(s)...")

    with wave.open(str(combined_wav_path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)

        for idx, vp in enumerate(video_paths, start=1):
            part_wav = combined_wav_path.with_suffix(f".part{idx}.wav")
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(vp),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(sample_rate), "-ac", "1",
                    str(part_wav)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                with wave.open(str(part_wav), "rb") as wav_in:
                    n_frames = wav_in.getnframes()
                    while True:
                        data = wav_in.readframes(16000 * 60)
                        if not data:
                            break
                        wav_out.writeframesraw(data)

                offset_sec = float(cumulative_samples) / float(sample_rate)
                duration_sec = float(n_frames) / float(sample_rate)
                cumulative_samples += n_frames

                seg = MediaSegment(
                    index=idx,
                    id_stem=vp.stem,
                    full_path=str(vp),
                    duration_sec=duration_sec,
                    offset_sec=offset_sec,
                )
                segments.append(seg)
                print(f"      Media {idx}: {vp.name} | Duration: {duration_sec:.3f}s | Offset: {offset_sec:.3f}s")
            finally:
                if part_wav.exists():
                    part_wav.unlink()

    return segments


def run_forced_alignment(wav_path: Path, word_tokens: List[WordToken], batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[2/4] Loading MMS-300M Aligner on {device.upper()} ({dtype})...")

    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        model_path=MODEL_ID,
        dtype=dtype,
    )

    audio_waveform = load_audio(str(wav_path), alignment_model.dtype, alignment_model.device)

    print(f"[3/4] Generating acoustic emissions (batch_size={batch_size})...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    flat_subwords: List[str] = []
    token_slices = []
    for w_tok in word_tokens:
        start_idx = len(flat_subwords)
        flat_subwords.extend(w_tok.spoken_subwords)
        end_idx = len(flat_subwords)
        token_slices.append((w_tok, start_idx, end_idx))

    print(f"      Aligning {len(word_tokens)} transcript words ({len(flat_subwords)} expanded phonetic subwords)...")
    full_spoken_text = " ".join(flat_subwords)

    tokens_starred, text_starred = preprocess_text(
        full_spoken_text,
        romanize=True,
        language="eng",
        split_size="word",
    )

    segments_aligned, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments_aligned, blank_token)
    aligned_subwords = postprocess_results(text_starred, spans, stride, scores)

    for w_tok, s_idx, e_idx in token_slices:
        sub_slice = aligned_subwords[s_idx:e_idx]
        if not sub_slice:
            continue
        w_tok.start_sec = float(sub_slice[0]["start"])
        w_tok.end_sec = float(sub_slice[-1]["end"])
        avg_score = sum(float(item.get("score", 0.0)) for item in sub_slice) / len(sub_slice)
        w_tok.score = round(avg_score, 4)


def populate_oncue_main_timecodes(main_rows: List[MainRow], word_tokens: List[WordToken]):
    """
    Populates Main.Timecode (float seconds) and Main.Timestamp ('HH:MM:SS')
    from the first spoken word on each line, and smoothly interpolates any
    speaker-only or blank lines inside the spoken testimony block.
    """
    row_map = {r.index: r for r in main_rows}

    for w in word_tokens:
        if w.start_sec is None:
            continue
        r = row_map[w.main_index]
        if r.timecode is None or w.start_sec < r.timecode:
            r.timecode = w.start_sec
            r.timestamp = format_hhmmss(w.start_sec)

    # Interpolate any empty/speaker-header 'LN' lines inside the active testimony range
    synced_indices = [i for i, r in enumerate(main_rows) if r.timecode is not None]
    if not synced_indices:
        return

    first_synced_pos = synced_indices[0]
    last_synced_pos = synced_indices[-1]

    # Set the line immediately preceding first spoken line to 0.0 (matches Sanction behavior)
    if first_synced_pos > 0 and main_rows[first_synced_pos - 1].temp == "LN":
        prev_r = main_rows[first_synced_pos - 1]
        prev_r.timecode = 0.0
        prev_r.timestamp = "00:00:00"

    last_tc = 0.0
    for pos in range(first_synced_pos, last_synced_pos + 1):
        r = main_rows[pos]
        if r.temp != "LN":
            continue
        if r.timecode is not None:
            last_tc = r.timecode
        else:
            r.timecode = last_tc
            r.timestamp = format_hhmmss(last_tc)


def export_oncue_mdb_and_sqlite(
    out_prefix: Path,
    first_name: str,
    last_name: str,
    depo_date: str,
    media_group: str,
    end_page: int,
    lines_per_page: int,
    segments: List[MediaSegment],
    main_rows: List[MainRow],
    word_tokens: List[WordToken],
):
    print(f"[4/4] Exporting OnCue/Sanction .mdb and word-level .sqlite to {out_prefix}.* ...")

    # 1. Export SQLite (contains Info, MPEGS, Main, AND word-level AlignedWords)
    sqlite_path = out_prefix.with_suffix(".sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn_sq = sqlite3.connect(str(sqlite_path))
    cur_sq = conn_sq.cursor()
    cur_sq.execute("""
        CREATE TABLE [Info] (
            [Index] INTEGER, [StartPage] INTEGER, [EndPage] INTEGER,
            [LinesPerPage] INTEGER, [Complete] INTEGER, [MediaGroup] TEXT,
            [SourceCase] TEXT, [FirstName] TEXT, [LastName] TEXT, [Date] TEXT
        )
    """)
    cur_sq.execute("""
        CREATE TABLE [MPEGS] (
            [Index] INTEGER, [Duration] REAL, [Offset] REAL,
            [MediaGroup] TEXT, [ID] TEXT, [FullPath] TEXT
        )
    """)
    cur_sq.execute("""
        CREATE TABLE [Main] (
            [Index] INTEGER PRIMARY KEY, [Timecode] REAL, [PageNum] INTEGER,
            [LineNum] INTEGER, [NoDisplay] INTEGER, [Redact] INTEGER,
            [Timestamp] TEXT, [Temp] TEXT, [Text] TEXT, [Native] TEXT
        )
    """)
    cur_sq.execute("""
        CREATE TABLE [AlignedWords] (
            [word_id] INTEGER PRIMARY KEY, [main_index] INTEGER,
            [page_num] INTEGER, [line_num] INTEGER, [raw_word] TEXT,
            [spoken_expansion] TEXT, [start_sec] REAL, [end_sec] REAL, [confidence] REAL
        )
    """)

    cur_sq.execute(
        "INSERT INTO [Info] VALUES (?,?,?,?,?,?,?,?,?,?)",
        (1, 1, end_page, lines_per_page, 1, None, None, first_name, last_name, depo_date)
    )
    cur_sq.executemany(
        "INSERT INTO [MPEGS] VALUES (?,?,?,?,?,?)",
        [(s.index, s.duration_sec, s.offset_sec, media_group, s.id_stem, s.full_path) for s in segments]
    )
    cur_sq.executemany(
        "INSERT INTO [Main] VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(r.index, r.timecode, r.page_num, r.line_num, int(r.no_display), int(r.redact), r.timestamp, r.temp, r.text, r.native) for r in main_rows]
    )
    cur_sq.executemany(
        "INSERT INTO [AlignedWords] VALUES (?,?,?,?,?,?,?,?,?)",
        [(w.word_id, w.main_index, w.page_num, w.line_num, w.raw_word, " ".join(w.spoken_subwords), w.start_sec, w.end_sec, w.score) for w in word_tokens]
    )
    conn_sq.commit()
    conn_sq.close()
    print(f"  -> Saved SQLite DB (with word-level table): {sqlite_path}")

    # 2. Export Genuine Jet 3.5 (Access 97 / Sanction format) .mdb via built-in 32-bit Windows DAO.DBEngine.36
    mdb_path = out_prefix.with_suffix(".mdb").resolve()
    if mdb_path.exists():
        mdb_path.unlink()

    payload_json = out_prefix.with_suffix(".dao_payload.json").resolve()
    ps_script_path = out_prefix.with_suffix(".build_mdb.ps1").resolve()

    payload_data = {
        "mdb_path": str(mdb_path),
        "info": {
            "Index": 1,
            "StartPage": 1,
            "EndPage": end_page,
            "LinesPerPage": lines_per_page,
            "Complete": True,
            "FirstName": first_name,
            "LastName": last_name,
            "Date": depo_date,
        },
        "mpegs": [
            {
                "Index": s.index,
                "Duration": round(s.duration_sec, 4),
                "Offset": round(s.offset_sec, 4),
                "MediaGroup": media_group,
                "ID": s.id_stem,
                "FullPath": s.full_path,
            }
            for s in segments
        ],
        "main": [
            {
                "Index": r.index,
                "Timecode": round(r.timecode, 4) if r.timecode is not None else None,
                "PageNum": r.page_num,
                "LineNum": r.line_num,
                "NoDisplay": r.no_display,
                "Redact": r.redact,
                "Timestamp": r.timestamp,
                "Temp": r.temp,
                "Text": r.text,
            }
            for r in main_rows
        ],
    }

    payload_json.write_text(json.dumps(payload_data), encoding="utf-8")

    # DAO Type Constants: dbBoolean=1, dbLong=4, dbSingle=6, dbText=10, dbMemo=12, dbVersion30=32 (Jet 3.5)
    ps_dao_builder = r"""
param([string]$JsonPath)
$ErrorActionPreference = "Stop"
$data = Get-Content -Raw -Encoding UTF8 $JsonPath | ConvertFrom-Json

$dbe = New-Object -ComObject DAO.DBEngine.36
# dbVersion30 = 32 creates exact Jet 3.5 format used by Sanction & OnCue
$db = $dbe.CreateDatabase($data.mdb_path, ";LANGID=0x0409;CP=1252;COUNTRY=0", 32)

# 1. Create Info Table
$tInfo = $db.CreateTableDef("Info")
$tInfo.Fields.Append($tInfo.CreateField("Index", 4))
$tInfo.Fields.Append($tInfo.CreateField("StartPage", 4))
$tInfo.Fields.Append($tInfo.CreateField("EndPage", 4))
$tInfo.Fields.Append($tInfo.CreateField("LinesPerPage", 4))
$tInfo.Fields.Append($tInfo.CreateField("Complete", 1))
$f = $tInfo.CreateField("MediaGroup", 10, 255); $f.AllowZeroLength = $true; $tInfo.Fields.Append($f)
$f = $tInfo.CreateField("SourceCase", 10, 255); $f.AllowZeroLength = $true; $tInfo.Fields.Append($f)
$f = $tInfo.CreateField("FirstName", 10, 100);  $f.AllowZeroLength = $true; $tInfo.Fields.Append($f)
$f = $tInfo.CreateField("LastName", 10, 100);   $f.AllowZeroLength = $true; $tInfo.Fields.Append($f)
$f = $tInfo.CreateField("Date", 10, 50);        $f.AllowZeroLength = $true; $tInfo.Fields.Append($f)
$db.TableDefs.Append($tInfo)

# 2. Create Main Table
$tMain = $db.CreateTableDef("Main")
$tMain.Fields.Append($tMain.CreateField("Index", 4))
$tMain.Fields.Append($tMain.CreateField("Timecode", 6))
$tMain.Fields.Append($tMain.CreateField("PageNum", 4))
$tMain.Fields.Append($tMain.CreateField("LineNum", 4))
$tMain.Fields.Append($tMain.CreateField("NoDisplay", 1))
$tMain.Fields.Append($tMain.CreateField("Redact", 1))
$f = $tMain.CreateField("Timestamp", 10, 20); $f.AllowZeroLength = $true; $tMain.Fields.Append($f)
$f = $tMain.CreateField("Temp", 10, 10);      $f.AllowZeroLength = $true; $tMain.Fields.Append($f)
$f = $tMain.CreateField("Text", 10, 255);     $f.AllowZeroLength = $true; $tMain.Fields.Append($f)
$f = $tMain.CreateField("Native", 10, 255);   $f.AllowZeroLength = $true; $tMain.Fields.Append($f)
$db.TableDefs.Append($tMain)

# 3. Create MPEGS Table
$tMpegs = $db.CreateTableDef("MPEGS")
$tMpegs.Fields.Append($tMpegs.CreateField("Index", 4))
$tMpegs.Fields.Append($tMpegs.CreateField("Duration", 6))
$tMpegs.Fields.Append($tMpegs.CreateField("Offset", 6))
$f = $tMpegs.CreateField("MediaGroup", 10, 255); $f.AllowZeroLength = $true; $tMpegs.Fields.Append($f)
$f = $tMpegs.CreateField("ID", 10, 255);         $f.AllowZeroLength = $true; $tMpegs.Fields.Append($f)
$f = $tMpegs.CreateField("FullPath", 10, 255);   $f.AllowZeroLength = $true; $tMpegs.Fields.Append($f)
$db.TableDefs.Append($tMpegs)

$dbe.BeginTrans()

# Populate Info
$rsInfo = $db.OpenRecordset("Info")
$rsInfo.AddNew()
$rsInfo.Fields.Item("Index").Value = [int]$data.info.Index
$rsInfo.Fields.Item("StartPage").Value = [int]$data.info.StartPage
$rsInfo.Fields.Item("EndPage").Value = [int]$data.info.EndPage
$rsInfo.Fields.Item("LinesPerPage").Value = [int]$data.info.LinesPerPage
$rsInfo.Fields.Item("Complete").Value = [bool]$data.info.Complete
$rsInfo.Fields.Item("FirstName").Value = [string]$data.info.FirstName
$rsInfo.Fields.Item("LastName").Value = [string]$data.info.LastName
$rsInfo.Fields.Item("Date").Value = [string]$data.info.Date
$rsInfo.Update()
$rsInfo.Close()

# Populate MPEGS
$rsMpegs = $db.OpenRecordset("MPEGS")
foreach ($m in $data.mpegs) {
    $rsMpegs.AddNew()
    $rsMpegs.Fields.Item("Index").Value = [int]$m.Index
    $rsMpegs.Fields.Item("Duration").Value = [single]$m.Duration
    $rsMpegs.Fields.Item("Offset").Value = [single]$m.Offset
    $rsMpegs.Fields.Item("MediaGroup").Value = [string]$m.MediaGroup
    $rsMpegs.Fields.Item("ID").Value = [string]$m.ID
    $rsMpegs.Fields.Item("FullPath").Value = [string]$m.FullPath
    $rsMpegs.Update()
}
$rsMpegs.Close()

# Populate Main
$rsMain = $db.OpenRecordset("Main")
foreach ($r in $data.main) {
    $rsMain.AddNew()
    $rsMain.Fields.Item("Index").Value = [int]$r.Index
    if ($null -ne $r.Timecode) {
        $rsMain.Fields.Item("Timecode").Value = [single]$r.Timecode
    }
    $rsMain.Fields.Item("PageNum").Value = [int]$r.PageNum
    $rsMain.Fields.Item("LineNum").Value = [int]$r.LineNum
    $rsMain.Fields.Item("NoDisplay").Value = [bool]$r.NoDisplay
    $rsMain.Fields.Item("Redact").Value = [bool]$r.Redact
    if ($null -ne $r.Timestamp) {
        $rsMain.Fields.Item("Timestamp").Value = [string]$r.Timestamp
    }
    $rsMain.Fields.Item("Temp").Value = [string]$r.Temp
    $txt = [string]$r.Text
    if ($txt.Length -gt 255) { $txt = $txt.Substring(0, 255) }
    $rsMain.Fields.Item("Text").Value = $txt
    $rsMain.Update()
}
$rsMain.Close()

$dbe.CommitTrans()
$db.Close()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($dbe) | Out-Null
"""

    ps_script_path.write_text(ps_dao_builder, encoding="utf-8")
    syswow64_ps = r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"

    try:
        subprocess.run(
            [syswow64_ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_script_path), "-JsonPath", str(payload_json)],
            check=True
        )
        print(f"  -> Saved OnCue/Sanction Jet 3.5 DB (.mdb): {mdb_path}")
    finally:
        if payload_json.exists():
            payload_json.unlink()
        if ps_script_path.exists():
            ps_script_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="OnCue-Compatible Offline Deposition Forced Aligner")
    parser.add_argument("--video", nargs="+", required=True, help="One or more video files in chronological order")
    parser.add_argument("--transcript", required=True, help="Path to deposition ASCII transcript (.txt)")
    parser.add_argument("--out", required=True, help="Output file prefix (e.g., D:\\Output\\John_Goodfellow)")
    parser.add_argument("--first-name", default="First", help="Witness First Name for OnCue Info table")
    parser.add_argument("--last-name", default="Last", help="Witness Last Name for OnCue Info table")
    parser.add_argument("--date", default="01/01/2026", help="Deposition Date (MM/DD/YYYY) for OnCue Info table")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch size (16-32 for RTX 40/50 series)")
    args = parser.parse_args()

    video_paths = resolve_video_inputs(args.video)
    transcript_path = Path(args.transcript).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    media_group = f"{args.last_name}, {args.first_name} (Vol. 01) - {args.date}"
    temp_wav = out_prefix.with_suffix(".temp_combined_16k.wav")

    try:
        main_rows, word_tokens, max_page, lines_per_page = parse_deposition_for_oncue(transcript_path)
        print(f"Parsed {len(main_rows)} total Main rows ({max_page} pages) and {len(word_tokens)} spoken words.")

        segments = extract_and_concat_16k_wav(video_paths, temp_wav)
        run_forced_alignment(temp_wav, word_tokens, batch_size=args.batch_size)
        populate_oncue_main_timecodes(main_rows, word_tokens)
        export_oncue_mdb_and_sqlite(
            out_prefix=out_prefix,
            first_name=args.first_name,
            last_name=args.last_name,
            depo_date=args.date,
            media_group=media_group,
            end_page=max_page,
            lines_per_page=lines_per_page,
            segments=segments,
            main_rows=main_rows,
            word_tokens=word_tokens,
        )
    finally:
        if temp_wav.exists():
            temp_wav.unlink()


if __name__ == "__main__":
    main()