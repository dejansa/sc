import re
import xml.etree.ElementTree as ET
from typing import Any, Dict

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


def parse_piste_data() -> Dict[str, Dict[str, Any]]:
    """Return a Krvavec-focused summary of every matching way/relation in the OSM file."""
    name_pattern = re.compile(r"krvavec", re.IGNORECASE)
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

        found_krvavec = any(name_pattern.search(value) for value in tags.values())
        name_has_krvavec = bool(name_pattern.search(tags.get("name", "")))
        is_winter_sports = tags.get("landuse") == "winter_sports"
        is_site_piste = tags.get("site") == "piste"
        is_type_site = tags.get("type") == "site"

        if not (
            found_krvavec
            or name_has_krvavec
            or (is_winter_sports and name_has_krvavec)
            or (tag_name == "relation" and (is_site_piste or is_type_site) and name_has_krvavec)
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
    """Print the number of Krvavec-related features found in the piste file."""
    piste_data = parse_piste_data()
    print(f"Found {len(piste_data)} Krvavec elements")
    print(f"{piste_data=}")


if __name__ == "__main__":
    main()
