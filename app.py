# --- adauga importuri sus, cu celelalte ---
from flask import Flask, request, jsonify, Response
import base64
# ------------------------------------------

# ... restul codului tău rămâne neschimbat ...

@app.route("/teapot/p.gif")
def ping_gif():
    """
    Image-beacon endpoint:
    - param 'u' = URL-ul imaginii (base64)
    Răspuns:
      width=1 px  -> CLEAN
      width=2 px  -> SUSPECT
      width=3 px  -> ERROR
    """
    try:
        u_b64 = request.args.get("u","")
        url = base64.b64decode(u_b64).decode("utf-8", "ignore")
        im = fetch_img(url)
        bw = to_bin(im, 168)

        best = 0.0
        for m in MASKS:
            s = match_score(bw, m)
            if s > best: best = s
            if best >= 0.74: break

        verdict = 2 if best >= 0.74 else 1  # 2 = suspect, 1 = clean

    except Exception:
        verdict = 3  # eroare de download / procesare

    # 1xN GIF transparent; returnăm lățimea = verdict
    out = Image.new("P", (verdict, 1), 0)
    buf = io.BytesIO()
    out.save(buf, format="GIF")
    b = buf.getvalue()
    return Response(
        b,
        mimetype="image/gif",
        headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
    )
