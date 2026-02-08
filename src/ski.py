import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Pattern

PISTE_PATH = "/mnt/c/Users/DSavkovic/Downloads/ski/planet_pistes.osm/planet_pistes-osmium.osm"


def _local_name(tag: str) -> str:
    """Return the tag name without any namespace prefix."""
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _extract_tags(element: ET.Element) -> Dict[str, str]:
    """Collect tag key/value pairs from a way/relation, handling namespaces."""
    tags: Dict[str, str] = {}
    for child in element:
        if _local_name(child.tag) != "tag":
            continue
        key = child.attrib.get("k")
        value = child.attrib.get("v")
        if key and value:
            tags[key] = value
    return tags


def _build_search_pattern(piste_name: str) -> Pattern[str]:
    """Build a case-insensitive pattern for user-supplied piste names."""

    normalized = piste_name.strip()
    if not normalized:
        raise ValueError("Piste name must not be empty")
    return re.compile(re.escape(normalized), re.IGNORECASE)


def parse_piste_data(piste_name: str) -> Dict[str, Dict[str, Any]]:
    """Return a summary of every matching way/relation in the OSM file for the requested piste."""
    name_pattern = _build_search_pattern(piste_name)
    piste_data: Dict[str, Dict[str, Any]] = {}

    for event, element in ET.iterparse(PISTE_PATH, events=("end",)):
        tag_name = _local_name(element.tag)
        if tag_name not in ("way", "relation"):
            if tag_name in {"nd", "member", "tag"}:
                # let the parent clear its children once we process it
                continue
            element.clear()
            continue

        tags = _extract_tags(element)
        if not tags:
            element.clear()
            continue

        found_match = any(name_pattern.search(value) for value in tags.values())
        name_matches = bool(name_pattern.search(tags.get("name", "")))
        is_winter_sports = tags.get("landuse") == "winter_sports"
        is_site_piste = tags.get("site") == "piste"
        is_type_site = tags.get("type") == "site"

        if not (
            found_match
            or name_matches
            or (is_winter_sports and name_matches)
            or (tag_name == "relation" and (is_site_piste or is_type_site) and name_matches)
        ):
            element.clear()
            continue

        element_id = element.attrib.get("id")
        if not element_id:
            element.clear()
            continue

        summary: Dict[str, Any] = {"id": element_id, "type": tag_name, "tags": tags}
        if tag_name == "way":
            summary["nodes"] = [
                child.attrib["ref"]
                for child in element
                if _local_name(child.tag) == "nd" and "ref" in child.attrib
            ]
        else:
            summary["members"] = [
                {
                    "type": child.attrib.get("type"),
                    "ref": child.attrib.get("ref"),
                    "role": child.attrib.get("role", ""),
                }
                for child in element
                if _local_name(child.tag) == "member"
            ]
        piste_data[element_id] = summary
        element.clear()

    return piste_data


def main() -> None:
    """Parse the piste name argument and print how many matching elements were found."""
    parser = argparse.ArgumentParser(description="Count piste-related features in the OSM dump.")
    parser.add_argument("piste_name", help="Case-insensitive substring to match in piste tags")
    args = parser.parse_args()

    try:
        piste_data = parse_piste_data(args.piste_name)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Found {len(piste_data)} elements matching {args.piste_name!r}")
    print(f"piste data: {json.dumps(piste_data, indent=2)}")


if __name__ == "__main__":
    main()
