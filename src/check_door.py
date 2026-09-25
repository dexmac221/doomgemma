"""Reads freedoom2.wad (the one shipped with ViZDoom) and reports what kind of door sector 37
of MAP02 is, where the oracle and the models get stuck, and where the keys are.
Usage: python check_door.py (in the environment with vizdoom)."""
import struct, vizdoom, os
wad = open(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"), "rb").read()
n, off = struct.unpack("<ii", wad[4:12])
dirs = [(struct.unpack("<ii", wad[off+16*i:off+16*i+8]), wad[off+16*i+8:off+16*i+16].rstrip(b"\0").decode()) for i in range(n)]
i = [k for k, (_, nm) in enumerate(dirs) if nm == "MAP02"][0]
lump = lambda name: next(((p, s) for (p, s), nm in dirs[i+1:i+12] if nm == name))
p, s = lump("LINEDEFS"); lines = [struct.unpack("<7H", wad[p+14*k:p+14*k+14]) for k in range(s//14)]
p, s = lump("SIDEDEFS"); sides = [struct.unpack("<hh8s8s8sH", wad[p+30*k:p+30*k+30]) for k in range(s//30)]
p, s = lump("THINGS"); things = [struct.unpack("<5h", wad[p+10*k:p+10*k+10]) for k in range(s//10)]
NAMES = {1: "DR normal", 26: "DR blue key", 27: "DR yellow key", 28: "DR red key", 31: "D1", 32: "D1 blue key", 33: "D1 red key", 34: "D1 yellow key"}
sp = [l[3] for l in lines if l[6] != 65535 and sides[l[6]][-1] == 37]
print("sector 37, linedef specials:", [(x, NAMES.get(x, "?")) for x in sp])
print("keys on MAP02:", [({5: "blue", 6: "yellow", 13: "red"}[t[3]], t[0], t[1]) for t in things if t[3] in (5, 6, 13)])
