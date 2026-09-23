from pathlib import Path

medquad_dir = Path(r"data\raw\medquad")

xml_files = list(medquad_dir.rglob("*.xml"))
print(f"Total XML files: {len(xml_files)}")

if not xml_files:
    print("No XML found!")
else:
    first = xml_files[0]
    print(f"First file: {first.relative_to(medquad_dir)}")
    print(f"Size: {first.stat().st_size} bytes")

    import xml.etree.ElementTree as ET
    tree = ET.parse(first)
    root = tree.getroot()
    print(f"Root tag: {root.tag}")
    print(f"Root attribs: {root.attrib}")
    for child in root:
        print(f"  child: {child.tag}")
        for sub in list(child)[:3]:
            print(f"     -> {sub.tag}")
