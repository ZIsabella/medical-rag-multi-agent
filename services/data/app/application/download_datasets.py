from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "/data/raw"))
WORK_DIR = RAW_DATA_DIR / "_work"

PUBMEDQA_ORI_PQAL_URL = os.getenv(
    "PUBMEDQA_ORI_PQAL_URL",
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
)

MEDQUAD_REPO_ZIP_URL = os.getenv(
    "MEDQUAD_REPO_ZIP_URL",
    "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip",
)

PUBMEDQA_OUTPUT = RAW_DATA_DIR / "pubmedqa.json"
MEDQUAD_OUTPUT = RAW_DATA_DIR / "medquad-complete.xml"


def download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, open(output_path, "wb") as f:
        shutil.copyfileobj(response, f)

    if output_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {output_path}")

    return output_path


def build_pubmedqa_json(src_path: Path, out_path: Path) -> Path:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict[str, Any]] = []

    if isinstance(data, dict):
        items = data.items()
        for pmid, item in items:
            question = item.get("QUESTION", "")
            contexts = item.get("CONTEXTS", [])
            long_answer = item.get("LONG_ANSWER", "")

            if isinstance(contexts, list):
                contexts_text = "\n".join(str(x).strip() for x in contexts if str(x).strip())
            else:
                contexts_text = str(contexts).strip()

            records.append(
                {
                    "QUESTION": str(question).strip(),
                    "CONTEXTS": contexts_text,
                    "LONG_ANSWER": str(long_answer).strip(),
                    "PMID": str(pmid),
                }
            )
    elif isinstance(data, list):
        for item in data:
            question = str(item.get("QUESTION", "")).strip()
            contexts = item.get("CONTEXTS", [])
            long_answer = str(item.get("LONG_ANSWER", "")).strip()

            if isinstance(contexts, list):
                contexts_text = "\n".join(str(x).strip() for x in contexts if str(x).strip())
            else:
                contexts_text = str(contexts).strip()

            records.append(
                {
                    "QUESTION": question,
                    "CONTEXTS": contexts_text,
                    "LONG_ANSWER": long_answer,
                }
            )
    else:
        raise RuntimeError("Unsupported PubMedQA JSON structure.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return out_path


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def append_medquad_entry(parent: ET.Element, question: str, answer: str, context: str) -> None:
    entry = ET.SubElement(parent, "Entry")
    q = ET.SubElement(entry, "Question")
    a = ET.SubElement(entry, "Answer")
    c = ET.SubElement(entry, "Context")

    q.text = question
    a.text = answer
    c.text = context


def build_medquad_xml(repo_zip_path: Path, out_path: Path) -> Path:
    """
    Extracts the official MedQuAD repository and converts its
    Document/QAPairs/QAPair XML files into the unified format expected
    by MedQuADLoader:

    <MedQuAD>
        <Entry>
            <Question>...</Question>
            <Answer>...</Answer>
            <Context>...</Context>
        </Entry>
    </MedQuAD>
    """

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=WORK_DIR) as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(repo_zip_path, "r") as zip_file:
            zip_file.extractall(tmpdir_path)

        extracted_dirs = [
            path for path in tmpdir_path.iterdir()
            if path.is_dir()
        ]

        if not extracted_dirs:
            raise RuntimeError(
                "Could not find the extracted MedQuAD directory."
            )

        extracted_root = extracted_dirs[0]

        output_root = ET.Element("MedQuAD")
        total_xml_files = 0
        total_documents = 0
        total_qapairs = 0

        xml_files = sorted(extracted_root.rglob("*.xml"))

        if not xml_files:
            raise RuntimeError(
                "No XML files were found in the MedQuAD archive."
            )

        for xml_file in xml_files:
            total_xml_files += 1

            try:
                tree = ET.parse(xml_file)
                document_root = tree.getroot()
            except ET.ParseError as error:
                print(f"[Warning] Invalid XML skipped: {xml_file} | {error}")
                continue
            except OSError as error:
                print(f"[Warning] Cannot read XML skipped: {xml_file} | {error}")
                continue

            if document_root.tag != "Document":
                continue

            total_documents += 1

            focus = normalize_text(
                document_root.findtext("./Focus")
            )

            source = document_root.attrib.get("source", "")
            source_url = document_root.attrib.get("url", "")

            context_parts = [
                value for value in [
                    focus,
                    source,
                    source_url,
                ]
                if normalize_text(value)
            ]

            context = " | ".join(
                normalize_text(value)
                for value in context_parts
            )

            qapairs = document_root.findall("./QAPairs/QAPair")

            for qapair in qapairs:
                question = normalize_text(
                    qapair.findtext("./Question")
                )
                answer = normalize_text(
                    qapair.findtext("./Answer")
                )

                # Some MedQuAD subsets may not include answers.
                if not question or not answer:
                    continue

                append_medquad_entry(
                    parent=output_root,
                    question=question,
                    answer=answer,
                    context=context,
                )

                total_qapairs += 1

        if total_qapairs == 0:
            raise RuntimeError(
                "No valid MedQuAD QAPair records were collected. "
                f"xml_files={total_xml_files}, "
                f"documents={total_documents}"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        ET.indent(output_root, space=" ")

        ET.ElementTree(output_root).write(
            out_path,
            encoding="utf-8",
            xml_declaration=True,
        )

        print(
            "[MedQuAD] Conversion completed: "
            f"xml_files={total_xml_files}, "
            f"documents={total_documents}, "
            f"qapairs={total_qapairs}, "
            f"output={out_path}"
        )

    return out_path



def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading PubMedQA source: {PUBMEDQA_ORI_PQAL_URL}")
    pubmedqa_source = download_file(PUBMEDQA_ORI_PQAL_URL, WORK_DIR / "ori_pqal.json")

    print(f"[2/4] Building final PubMedQA file: {PUBMEDQA_OUTPUT}")
    build_pubmedqa_json(pubmedqa_source, PUBMEDQA_OUTPUT)

    print(f"[3/4] Downloading MedQuAD archive: {MEDQUAD_REPO_ZIP_URL}")
    medquad_zip = download_file(MEDQUAD_REPO_ZIP_URL, WORK_DIR / "medquad.zip")

    print(f"[4/4] Building final MedQuAD file: {MEDQUAD_OUTPUT}")
    build_medquad_xml(medquad_zip, MEDQUAD_OUTPUT)

    print("Done.")
    print(f"PubMedQA: {PUBMEDQA_OUTPUT}")
    print(f"MedQuAD:  {MEDQUAD_OUTPUT}")


if __name__ == "__main__":
    main()
