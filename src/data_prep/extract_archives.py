# TS/TL 일괄 압축 해제 유틸임
# ============================================================
# 📄 파일명: extract_archives.py
# 📁 위치: ai_modules/src/data_prep/extract_archives.py
# 📘 목적:
#   - TS_xx / TL_xx 형식의 원천데이터·라벨 ZIP 파일을 일괄 해제하는 유틸임.
#   - 이미지 ZIP 예: <IMG_ZIP_BASE>/zip/TS_81_단일.zip
#   - 라벨  ZIP 예: <LBL_ZIP_BASE>/TL_81_단일.zip
#
# 🔌 입력 인자:
#   --img-zip-base   : TS_* 루트 경로(예: ".../원천데이터/단일경구약제 5000종")
#   --lbl-zip-base   : TL_* 루트 경로(예: ".../라벨링데이터/단일경구약제 5000종")
#   --targets        : 콤마 구분 대상 목록 (예: "39,41,42,43")
#   --range          : 연속 구간 지정 (예: "38-54")  ※ --targets 대신 사용 가능
#   --suffix         : 접미사(기본: "단일")  → 파일명: TS_<n>_<suffix>.zip
#   --img-prefix     : 이미지 접두어(기본: "TS")
#   --lbl-prefix     : 라벨   접두어(기본: "TL")
#   --skip-labels    : 라벨 압축 해제를 건너뜀
#   --skip-images    : 이미지 압축 해제를 건너뜀
#   --overwrite      : 출력 폴더가 존재해도 덮어씀(기본은 존재하면 스킵)
#
# 🧪 사용 예시:
#   1) 지정 목록:
#      python -m ai_modules.src.data_prep.extract_archives \
#        --img-zip-base "/.../원천데이터/단일경구약제 5000종" \
#        --lbl-zip-base "/.../라벨링데이터/단일경구약제 5000종" \
#        --targets 39,41,42,43,46,48,51,54
#
#   2) 구간 지정:
#      python -m ai_modules.src.data_prep.extract_archives \
#        --img-zip-base "/.../원천데이터/단일경구약제 5000종" \
#        --lbl-zip-base "/.../라벨링데이터/단일경구약제 5000종" \
#        --range 38-54
#
#   3) 이미지만 해제:
#      --skip-labels 플래그 사용
#
#   4) 라벨만 해제:
#      --skip-images 플래그 사용
#
# ⚠️ 참고:
#   - Colab/Windows의 한글·공백 경로를 고려해 pathlib 사용함.
#   - 손상 ZIP은 건너뛰고 경고만 출력함.
# ============================================================

from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

def _parse_targets(targets_str: str | None, range_str: str | None) -> List[int]:
    """--targets 또는 --range(예: '38-54')를 리스트로 변환함."""
    if targets_str:
        out: List[int] = []
        for tok in targets_str.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                print(f"[WARN] 정수가 아님: '{tok}' → 스킵")
        return sorted(set(out))
    if range_str:
        try:
            a, b = range_str.split("-")
            lo, hi = int(a.strip()), int(b.strip())
            if lo > hi:
                lo, hi = hi, lo
            return list(range(lo, hi + 1))
        except Exception:
            print(f"[WARN] --range 파싱 실패: '{range_str}' → 빈 목록 반환")
            return []
    return []

def _extract_one(zip_path: Path, out_dir: Path, overwrite: bool = False) -> Tuple[bool, str]:
    """단일 ZIP을 out_dir에 해제함. (성공 여부, 메시지) 반환함."""
    if not zip_path.exists():
        return False, f"[MISS] {zip_path}"
    if out_dir.exists() and not overwrite:
        return True, f"[SKIP] 이미 존재: {out_dir}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        return True, f"[OK] {zip_path} -> {out_dir}"
    except zipfile.BadZipFile:
        return False, f"[ERROR] 손상된 ZIP: {zip_path}"
    except Exception as e:
        return False, f"[ERROR] {zip_path} → {e}"

def extract_targets(
    img_zip_base: Path,
    lbl_zip_base: Path,
    targets: Iterable[int],
    img_prefix: str = "TS",
    lbl_prefix: str = "TL",
    suffix: str = "단일",
    skip_images: bool = False,
    skip_labels: bool = False,
    overwrite: bool = False,
) -> None:
    """대상 번호들에 대해 이미지/라벨 ZIP을 일괄 해제함."""
    img_zip_base = Path(img_zip_base)
    lbl_zip_base = Path(lbl_zip_base)

    for t in targets:
        # 이미지 ZIP: <img_zip_base>/zip/TS_<t>_<suffix>.zip → <img_zip_base>/TS_<t>_<suffix>/
        if not skip_images:
            izip = img_zip_base / "zip" / f"{img_prefix}_{t}_{suffix}.zip"
            iout = img_zip_base / f"{img_prefix}_{t}_{suffix}"
            ok, msg = _extract_one(izip, iout, overwrite=overwrite)
            print(msg)

        # 라벨 ZIP: <lbl_zip_base>/TL_<t>_<suffix>.zip → <lbl_zip_base>/TL_<t>_<suffix>/
        if not skip_labels:
            lzip = lbl_zip_base / f"{lbl_prefix}_{t}_{suffix}.zip"
            lout = lbl_zip_base / f"{lbl_prefix}_{t}_{suffix}"
            ok, msg = _extract_one(lzip, lout, overwrite=overwrite)
            print(msg)

def main():
    ap = argparse.ArgumentParser(description="TS/TL 아카이브 일괄 압축 해제 유틸임.")
    ap.add_argument("--img-zip-base", required=True, help="TS_* 루트(예: .../원천데이터/단일경구약제 5000종)")
    ap.add_argument("--lbl-zip-base", required=True, help="TL_* 루트(예: .../라벨링데이터/단일경구약제 5000종)")
    ap.add_argument("--targets", type=str, default=None, help='콤마 목록, 예: "39,41,42,43"')
    ap.add_argument("--range", type=str, default=None, help='구간, 예: "38-54" (targets 대신 사용 가능)')
    ap.add_argument("--suffix", type=str, default="단일", help='파일 접미사(기본: "단일")')
    ap.add_argument("--img-prefix", type=str, default="TS")
    ap.add_argument("--lbl-prefix", type=str, default="TL")
    ap.add_argument("--skip-images", action="store_true", help="이미지 압축 해제를 건너뜀")
    ap.add_argument("--skip-labels", action="store_true", help="라벨   압축 해제를 건너뜀")
    ap.add_argument("--overwrite", action="store_true", help="출력 폴더가 있어도 덮어씀")
    args = ap.parse_args()

    targets = _parse_targets(args.targets, args.range)
    if not targets:
        print("[WARN] 대상 번호가 비어 있음. --targets 또는 --range를 지정해야 함.")
        return

    extract_targets(
        img_zip_base=Path(args.img_zip_base),
        lbl_zip_base=Path(args.lbl_zip_base),
        targets=targets,
        img_prefix=args.img_prefix,
        lbl_prefix=args.lbl_prefix,
        suffix=args.suffix,
        skip_images=args.skip_images,
        skip_labels=args.skip_labels,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
