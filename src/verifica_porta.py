"""Legge freedoom2.wad (quello di ViZDoom) e dice che tipo di porta e' il settore 37 di MAP02,
dove l'oracolo e i modelli restano bloccati. Uso: python verifica_porta.py (nel venv con vizdoom)."""
import struct, vizdoom, os
wad = open(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"), "rb").read()
n, off = struct.unpack("<ii", wad[4:12])
dirs = [(struct.unpack("<ii", wad[off+16*i:off+16*i+8]), wad[off+16*i+8:off+16*i+16].rstrip(b"\0").decode()) for i in range(n)]
i = [k for k, (_, nm) in enumerate(dirs) if nm == "MAP02"][0]
lump = lambda name: next(((p, s) for (p, s), nm in dirs[i+1:i+12] if nm == name))
p, s = lump("LINEDEFS"); lines = [struct.unpack("<7H", wad[p+14*k:p+14*k+14]) for k in range(s//14)]
p, s = lump("SIDEDEFS"); sides = [struct.unpack("<hh8s8s8sH", wad[p+30*k:p+30*k+30]) for k in range(s//30)]
p, s = lump("THINGS"); things = [struct.unpack("<5h", wad[p+10*k:p+10*k+10]) for k in range(s//10)]
NOMI = {1: "DR normale", 26: "DR chiave blu", 27: "DR chiave gialla", 28: "DR chiave rossa", 31: "D1", 32: "D1 chiave blu", 33: "D1 chiave rossa", 34: "D1 chiave gialla"}
sp = [l[3] for l in lines if l[6] != 65535 and sides[l[6]][-1] == 37]
print("settore 37, special delle linee:", [(x, NOMI.get(x, "?")) for x in sp])
print("chiavi su MAP02:", [({5: "blu", 6: "gialla", 13: "rossa"}[t[3]], t[0], t[1]) for t in things if t[3] in (5, 6, 13)])
