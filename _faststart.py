import av
import os

BASE = "assets/animations/Positional_Encoding"
FILES = [
    "FourTermExpansio",
    "RotationRelativeAngle",
    "RotationPreservesLength",
    "RotationPreservesAngle",
]

def remux(stem):
    src = os.path.join(BASE, stem + ".mp4")
    tmp = os.path.join(BASE, stem + ".faststart.mp4")
    print(f"remux {stem}.mp4 ...", flush=True)

    inp = av.open(src)
    in_stream = inp.streams.video[0]
    out = av.open(tmp, "w", options={"movflags": "+faststart"})
    try:
        out_stream = out.add_stream_from_template(in_stream)
    except AttributeError:
        out_stream = out.add_stream(template=in_stream)

    for packet in inp.demux(in_stream):
        if packet.dts is None:
            continue
        packet.stream = out_stream
        out.mux(packet)
    out.close()
    inp.close()

    os.replace(tmp, src)
    print(f"  done ({os.path.getsize(src)/1e6:.1f} MB)", flush=True)

for s in FILES:
    remux(s)
print("ALL DONE", flush=True)
