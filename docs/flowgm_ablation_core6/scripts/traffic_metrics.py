#!/usr/bin/env python
"""Field-level metrics for generated traffic-record comparisons."""
import argparse, json, re

FIELDS = ['frame.len','eth.dst','eth.src','eth.type','ip.hdr_len','ip.dsfield','ip.len','ip.id','ip.flags','ip.frag_offset','ip.ttl','ip.proto','ip.src','ip.dst','tcp.srcport','tcp.dstport','tcp.hdr_len','tcp.seq','tcp.ack','tcp.flags','tcp.window_size','tcp.urgent_pointer']

def parse(s):
    out = {}
    for k in FIELDS:
        m = re.search(r'(?m)(?:^|, )' + re.escape(k) + r': ([^,\\n]+)', s)
        out[k] = m.group(1).strip() if m else None
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('file'); a = ap.parse_args()
    rows = json.load(open(a.file, encoding='utf-8')); hit = {k: 0 for k in FIELDS}; full = 0
    for row in rows:
        ref, pred = parse(row['reference']), parse(row['after'])
        for k in FIELDS: hit[k] += int(ref[k] is not None and pred[k] == ref[k])
        full += int(all(ref[k] is not None and pred[k] == ref[k] for k in FIELDS))
    n = len(rows) or 1
    print(json.dumps({'samples': len(rows), 'full_struct_exact': full/n,
                      'field_accuracy': {k: v/n for k,v in hit.items()}}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
