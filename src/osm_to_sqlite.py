"""
Script to parse OSM XML and copy data into SQLite database.

- Reads OSM XML file: /mnt/c/Users/DSavkovic/Downloads/ski/planet_pistes.osm/planet_pistes-osmium.osm
- Creates/uses SQLite DB: data/piste/data.db
- Schema matches normalized OSM structure (nodes, ways, relations, tags, members).
"""
import os
import sqlite3
import xml.etree.ElementTree as ET


# OSM Format
# An extract of the osm database is available daily here: planet_pistes.osm.gz
# https://www.opensnowmap.org/download/planet_pistes.osm.gz
# File timestamp: planet_pistes-state.txt
# https://www.opensnowmap.org/download/planet_pistes-state.txt

OSM_PATH = "/mnt/c/Users/DSavkovic/Downloads/ski/planet_pistes.osm/planet_pistes-osmium.osm"
PISTE_STATE_PATH = "/mnt/c/Users/DSavkovic/Downloads/ski/planet_pistes-state.txt"
DB_PATH = "data/piste/data.db"

SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY,
        version INTEGER,
        timestamp TEXT,
        uid INTEGER,
        user TEXT,
        changeset INTEGER,
        lat REAL,
        lon REAL
    );''',
    '''CREATE TABLE IF NOT EXISTS node_tags (
        node_id INTEGER,
        k TEXT,
        v TEXT,
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    );''',
    '''CREATE TABLE IF NOT EXISTS ways (
        id INTEGER PRIMARY KEY,
        version INTEGER,
        timestamp TEXT,
        uid INTEGER,
        user TEXT,
        changeset INTEGER
    );''',
    '''CREATE TABLE IF NOT EXISTS way_nodes (
        way_id INTEGER,
        node_id INTEGER,
        ord INTEGER,
        FOREIGN KEY (way_id) REFERENCES ways(id),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    );''',
    '''CREATE TABLE IF NOT EXISTS way_tags (
        way_id INTEGER,
        k TEXT,
        v TEXT,
        FOREIGN KEY (way_id) REFERENCES ways(id)
    );''',
    '''CREATE TABLE IF NOT EXISTS relations (
        id INTEGER PRIMARY KEY,
        version INTEGER,
        timestamp TEXT,
        uid INTEGER,
        user TEXT,
        changeset INTEGER
    );''',
    '''CREATE TABLE IF NOT EXISTS relation_members (
        relation_id INTEGER,
        member_type TEXT,
        member_ref INTEGER,
        role TEXT,
        ord INTEGER,
        FOREIGN KEY (relation_id) REFERENCES relations(id)
    );''',
    '''CREATE TABLE IF NOT EXISTS relation_tags (
        relation_id INTEGER,
        k TEXT,
        v TEXT,
        FOREIGN KEY (relation_id) REFERENCES relations(id)
    );''',
    '''CREATE VIEW way_with_tags AS
        SELECT
            w.id,
            w.version,
            w.timestamp,
            w.uid,
            w.user,
            w.changeset,
            wt.k AS tag_key,
            wt.v AS tag_value
        FROM ways AS w
        LEFT JOIN way_tags AS wt ON wt.way_id = w.id;'''
    '''CREATE VIEW relation_with_tags AS
        SELECT
            r.id,
            r.version,
            r.timestamp,
            r.uid,
            r.user,
            r.changeset,
            rt.k AS tag_key,
            rt.v AS tag_value
        FROM relations AS r
        LEFT JOIN relation_tags AS rt ON rt.relation_id = r.id;'''
    '''CREATE VIEW way_with_nodes AS
        SELECT
            w.id,
            wn.node_id,
            wn.ord
        FROM ways AS w
        LEFT JOIN way_nodes AS wn ON wn.way_id = w.id
    /* way_with_nodes(id,node_id,ord) */;'''
    '''CREATE VIEW relation_with_members AS
        SELECT
            r.id,
            rm.member_type,
            rm.member_ref,
            rm.role,
            rm.ord
        FROM relations AS r
        LEFT JOIN relation_members AS rm ON rm.relation_id = r.id;'''
]

def _local_name(tag):
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag

def _extract_tags(element):
    return [(child.attrib["k"], child.attrib["v"]) for child in element if _local_name(child.tag) == "tag" and "k" in child.attrib and "v" in child.attrib]

def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    conn.commit()

    # Parse XML
    ord_cache = {"way": {}, "relation": {}}
    for event, elem in ET.iterparse(OSM_PATH, events=("end",)):
        tag = _local_name(elem.tag)
        if tag not in {"node", "way", "relation"}:
            continue
        if tag == "node":
            cur.execute(
                "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(elem.attrib["id"]),
                    int(elem.attrib.get("version", 1)),
                    elem.attrib.get("timestamp"),
                    int(elem.attrib.get("uid", 0)),
                    elem.attrib.get("user"),
                    int(elem.attrib.get("changeset", 0)),
                    float(elem.attrib.get("lat", 0)),
                    float(elem.attrib.get("lon", 0)),
                )
            )
            for k, v in _extract_tags(elem):
                cur.execute("INSERT INTO node_tags VALUES (?, ?, ?)", (int(elem.attrib["id"]), k, v))
        elif tag == "way":
            cur.execute(
                "INSERT OR IGNORE INTO ways VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(elem.attrib["id"]),
                    int(elem.attrib.get("version", 1)),
                    elem.attrib.get("timestamp"),
                    int(elem.attrib.get("uid", 0)),
                    elem.attrib.get("user"),
                    int(elem.attrib.get("changeset", 0)),
                )
            )
            ord_cache["way"][elem.attrib["id"]] = 0
            for child in elem:
                if _local_name(child.tag) == "nd" and "ref" in child.attrib:
                    cur.execute(
                        "INSERT INTO way_nodes VALUES (?, ?, ?)",
                        (int(elem.attrib["id"]), int(child.attrib["ref"]), ord_cache["way"][elem.attrib["id"]])
                    )
                    ord_cache["way"][elem.attrib["id"]] += 1
            for k, v in _extract_tags(elem):
                cur.execute("INSERT INTO way_tags VALUES (?, ?, ?)", (int(elem.attrib["id"]), k, v))
        elif tag == "relation":
            cur.execute(
                "INSERT OR IGNORE INTO relations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(elem.attrib["id"]),
                    int(elem.attrib.get("version", 1)),
                    elem.attrib.get("timestamp"),
                    int(elem.attrib.get("uid", 0)),
                    elem.attrib.get("user"),
                    int(elem.attrib.get("changeset", 0)),
                )
            )
            ord_cache["relation"][elem.attrib["id"]] = 0
            for child in elem:
                if _local_name(child.tag) == "member":
                    cur.execute(
                        "INSERT INTO relation_members VALUES (?, ?, ?, ?, ?)",
                        (
                            int(elem.attrib["id"]),
                            child.attrib.get("type"),
                            int(child.attrib.get("ref", 0)),
                            child.attrib.get("role", ""),
                            ord_cache["relation"][elem.attrib["id"]]
                        )
                    )
                    ord_cache["relation"][elem.attrib["id"]] += 1
            for k, v in _extract_tags(elem):
                cur.execute("INSERT INTO relation_tags VALUES (?, ?, ?)", (int(elem.attrib["id"]), k, v))
        elem.clear()
    conn.commit()
    conn.close()
    print(f"Finished importing OSM XML to {DB_PATH}")

if __name__ == "__main__":
    main()
