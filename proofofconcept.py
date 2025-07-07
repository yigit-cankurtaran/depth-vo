import cv2
import numpy as np
import timm
import torch

# ---- depth net (224-224 vit) ----
midas = timm.create_model('vit_small_patch16_224', pretrained=True).eval().to('mps')

cap   = cv2.VideoCapture('processed.mp4')
prev_gray = None
poses = []

# scale is good enough as seen by the graph and results
K = None

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # rgb -> depth -----------------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv reads BGR,pytorch and timm expect RGB
    h, w = rgb.shape[:2]
    if K is None:  # build camera matrix once
        fx = fy = 0.9 * w
        K  = np.array([[fx, 0, w/2],
                       [0, fy, h/2],
                       [0,  0,   1]], dtype=np.float32)

    rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    inp = torch.from_numpy(rgb_224).permute(2,0,1).float().div(255.).unsqueeze(0).to("mps")
    # permute because we want CHW instead of HWC
    with torch.no_grad():
        pred = midas(inp)
    depth = cv2.resize(pred.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, 1000, 0.01, 10) # grabbing corners
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None) # using lucas-kanade
        # p1 is predicted positions
        # st = 1 if track survives, 0 if it loses corner
        if p1 is None:
            prev_gray = gray
            continue

        p0 = p0[st==1].astype(np.float32)
        p1 = p1[st==1].astype(np.float32)
        if len(p0) < 6: # need 4, keep margin
            prev_gray = gray
            continue

        # prev_gray is None at the start, just store the frame
        # from the 2nd frame on we grab corners
        # p1 is the LK prediction of the new position
        # if not enough valid pairs, continue

        # depth -> 3-d (camera coords)
        z = depth[p0[:,1].astype(int), p0[:,0].astype(int)] # ,1 is ys, ,0 is xs
        valid = z > 0
        obj_pts = np.column_stack([ # each 2d corner + its depth turns to a 3d point in the camera frame
            (p0[valid,0] - K[0,2]) * z[valid] / K[0,0], # pinhole math
            (p0[valid,1] - K[1,2]) * z[valid] / K[1,1],
            z[valid]
        ]).astype(np.float32)

        img_pts = p1[valid].astype(np.float32)

        if len(obj_pts) >= 4:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj_pts, img_pts, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                confidence=0.99,
                iterationsCount=100
            ) # find the R,t pose that best reprojects the 3d points
            if ok: # flatten and concat the vectors into a 6 float row, append to poses
                poses.append(np.hstack([tvec.flatten(), rvec.flatten()]))

    prev_gray = gray # next loop, this frame becomes the previous one

# dump
if poses:
    np.savetxt('poses.csv', np.vstack(poses), delimiter=',')
    print("done, wrote", len(poses), "poses.")
else:
    print("no valid poses, check lighting / texture.")
