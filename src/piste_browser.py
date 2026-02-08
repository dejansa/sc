import json
import argparse
import sqlite3
from typing import Dict

DB_PATH = "data/piste/data.db"


def get_resort_ways(resort_name: str) -> Dict[int, str]:
    """Return a map of relation IDs to matching names for the resort."""
    resort_ways = {}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, tag_value
        FROM relation_with_tags
        WHERE (tag_key='name' AND tag_value LIKE ?) OR (tag_key='name:en' AND tag_value LIKE ?)
        LIMIT 100
    ''', (f"%{resort_name}%",f"%{resort_name}%"))
    relations = cursor.fetchall()
    # conn.close()
    # print(f"{len(relations)=}")
    for relation in relations:
        # print("===============================")
        # print(f"{relation=}")
        resort_ways[relation[1]] = {"ways": []}
        # select data from relation_with_members
        conn = sqlite3.connect(DB_PATH)
        # cursor = conn.cursor()
        cursor.execute('''
            SELECT member_type, member_ref, role
            FROM relation_with_members
            WHERE member_type in ('way', 'node') AND id=?
        ''', (relation[0],))
        members = cursor.fetchall()
        for member in members:
            # print("  -------------------------------")
            # print(f"  {member=}")
            # resort_ways[relation[1]][f"{member[0]}s"][member[1]] = {"type": member[0], "role": member[2]}
            if member[0] == 'way':
                resort_ways[relation[1]]["ways"].append({"type": member[0], "nodes": []})
                cursor.execute('''
                    SELECT id, tag_key, tag_value
                    FROM way_with_tags
                    WHERE id=?
                ''', (member[1],))
                way_tags = cursor.fetchall()
                # print("    ---")
                for way_tag in way_tags:
                    # print(f"    {way_tag=}")
                    resort_ways[relation[1]]["ways"][-1][way_tag[1]] = way_tag[2]
                cursor.execute('''
                    SELECT *
                    FROM way_with_nodes
                    WHERE id=?
                ''', (member[1],))
                way_nodes = cursor.fetchall()
                # print("    ---")
                for way_node in way_nodes:
                    # print(f"    {way_node=}")
                    cursor.execute('''
                        SELECT id, lat, lon
                        FROM nodes
                        WHERE id=?
                    ''', (way_node[1],))
                    way_nodes = cursor.fetchall()
                    # print("      ---")
                    # print(f"      {way_nodes=}")
                    resort_ways[relation[1]]["ways"][-1]["nodes"].append({"lat": way_nodes[0][1], "lon": way_nodes[0][2]})
            elif member[0] == 'node':
                resort_ways[relation[1]]["ways"].append({"type": member[0], "role": member[2], "nodes": []})
                cursor.execute('''
                    SELECT id, lat, lon
                    FROM nodes
                    WHERE id=?
                ''', (member[1],))
                node_data = cursor.fetchall()
                # print(f"      {node_data=}")
                resort_ways[relation[1]]["ways"][-1]["nodes"].append({"lat": node_data[0][1], "lon": node_data[0][2]})

                # print(f"    {way_tags=}")
    conn.close()
    return resort_ways


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the piste browser script."""
    parser = argparse.ArgumentParser(description="List resort relation details from the piste database.")
    parser.add_argument(
        "-r",
        "--resort",
        default="Kopaonik",
        help="Resort name to query",
    )
    return parser.parse_args()


def main() -> None:
    """Run the primary CLI workflow."""
    args = parse_arguments()
    resort_ways = get_resort_ways(args.resort)
    # print(f"Resort ways for {args.resort}: {json.dumps(resort_ways, indent=2)}")
    for resort_name, resort_data in resort_ways.items():
        print(f"Resort: {resort_name}")
        for way in resort_data["ways"]:
            if "aerialway" not in way and "role" not in way:
                name = way.get("name")
                if name:
                    print(f"  {name}:")
                    print(f"    dificulty: {way.get('piste:difficulty', '??')} type: {way.get('piste:type', '??')}")
                    # print(f"    type: {way.get('piste:type', '??')}")
                    # tags = {k: v for k, v in way.items() if k not in ['type', 'nodes']}
                    # print(f"    Tags: {tags}")
                for node in way["nodes"]:
                    print(f"    Node: lat={node['lat']}, lon={node['lon']}")
        print()
        for way in resort_data["ways"]:
            if "aerialway" not in way and "role" not in way:
                name = way.get("name")
                if name is None:
                    print(f"    dificulty: {way.get('piste:difficulty', '??')} type: {way.get('piste:type', '??')}")
                    # print(f"    dificulty: {way.get('piste:difficulty', '??')}")
                    tags = {k: v for k, v in way.items() if k not in ['type', 'nodes']}
                    print(f"    Tags: {tags}")
                for node in way["nodes"]:
                    print(f"    Node: lat={node['lat']}, lon={node['lon']}")


                # print(f"    Tags: { {k: v for k, v in way.items() if k not in ['type', 'nodes']} }")

            # print(f"  Way type: {way['type']}")
            # if way['type'] == 'way':
            #     print(f"    Tags: { {k: v for k, v in way.items() if k not in ['type', 'nodes']} }")
            # for node in way["nodes"]:
            #     print(f"    Node: lat={node['lat']}, lon={node['lon']}")
    # for piste in pistes:
    #     print(piste, type(piste))


if __name__ == "__main__":
    main()
