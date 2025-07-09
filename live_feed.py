import cv2
import torch
import timm
import numpy as np
import time

# depth net (vit-small)
midas = timm.create_model('vit_small_patch16_224',
                          pretrained=True).eval().to('mps')

# webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macbook cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)          # ask for 640x480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_gray = None
K = None # intrinsics (camera matrix) filled on first frame
colors = np.random.randint(64, 255, (1000, 3)).tolist()

fps_ema, t0 = 0.0, time.time() # fps monitor
poses_live = [] # per-frame pose rows


while True:
    ok, frame = cap.read()
    if not ok:
        break

    # depth 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if K is None:
        fx = fy = 0.9 * w
        K  = np.array([[fx, 0, w/2],
                       [0, fy, h/2],
                       [0,  0,   1]], np.float32)

    rgb_224 = cv2.resize(rgb, (224, 224), cv2.INTER_AREA)
    inp = (torch.from_numpy(rgb_224)
              .permute(2, 0, 1).float().div(255.)
              .unsqueeze(0).to('mps'))
    with torch.no_grad():
        pred = midas(inp)
    depth = cv2.resize(pred.squeeze().cpu().numpy(),
                       (w, h), cv2.INTER_CUBIC)

    # vo front-end
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    draw = frame.copy()

    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, 500, 0.01, 10) # cutting corners to half
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                             winSize=(15,15), maxLevel=2)
        if p1 is not None:
            p0 = p0[st == 1].astype(np.float32)
            p1 = p1[st == 1].astype(np.float32)

            # flow viz
            for i, (a, b) in enumerate(zip(p0, p1)):
                c = colors[i % len(colors)]
                a = tuple(a.astype(int).ravel()) # opencv wants integer pixels
                b = tuple(b.astype(int).ravel())
                cv2.line(draw, a, b, c, 1)
                cv2.circle(draw, b, 2, c, -1)

            if len(p0) >= 6:
                z = depth[p0[:, 1].astype(int), p0[:, 0].astype(int)]
                ok_mask = z > 0
                obj = np.column_stack([
                    (p0[ok_mask, 0] - K[0, 2]) * z[ok_mask] / K[0, 0],
                    (p0[ok_mask, 1] - K[1, 2]) * z[ok_mask] / K[1, 1],
                    z[ok_mask]
                ]).astype(np.float32)
                img = p1[ok_mask].astype(np.float32)

                if len(obj) >= 4:
                    ok_pose, rvec, tvec, _ = cv2.solvePnPRansac(
                        obj, img, K, None, reprojectionError=8.0,
                        iterationsCount=100, confidence=0.99
                    )
                    if ok_pose:
                        tx, ty, tz = tvec.flatten()
                        cv2.putText(draw,
                                    f"t: [{tx:.3f}, {ty:.3f}, {tz:.3f}] m",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

                        poses_live.append(
                            np.hstack([tvec.flatten(), rvec.flatten()])
                        )

    prev_gray = gray

    # fps diagnostics
    dt = time.time() - t0
    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
    cv2.putText(draw, f"{fps_ema:5.1f} fps", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    t0 = time.time()

    cv2.imshow("depth-vo live", draw)
    if cv2.waitKey(1) & 0xff in (27, ord('q')):  # esc or q to quit
        break

cap.release()
cv2.destroyAllWindows()

if poses_live:
    np.savetxt('poses_live.csv', np.vstack(poses_live), delimiter=',')
    print(f"saved {len(poses_live)} rows to poses_live.csv")
else:
    print("no poses captured, check lighting/texture/both")
