"""Interactive helper to validate detected quad pairs before triangulating a cube.

Workflow:
- detect quadrilateral contours in both images
- enumerate candidate quad pairs (all combinations)
- show them side-by-side with indices; user can cycle (n/p) and accept (a)
- on accept: compute metric P matrices (chessboard), triangulate the 8 corner points,
  compute PCA axes and per-axis extents, save NPZ/JSON and visualization

This is a small usability layer on top of the automatic heuristics to make the
measurement workflow reliable without full manual clicking.
"""
import os
import cv2
import numpy as np
import argparse
import json
try:
    import common_utils as cu
except Exception:
    from project_2.phase_2 import common_utils as cu





def detect_quads(img, min_area=1000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4,2)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) if hull is not None and len(hull) > 0 else area
            solidity = float(area) / (hull_area + 1e-8)
            x,y,w,h = cv2.boundingRect(approx)
            aspect = float(w) / (h + 1e-8)
            ordered = order_quad_points(pts)
            angles = quad_angles(ordered)
            quads.append({'pts': ordered, 'area': area, 'solidity': solidity, 'aspect': aspect, 'bbox': (x,y,w,h), 'angles': angles})
    # sort by descending area (likely base first)
    quads.sort(key=lambda q: q['area'], reverse=True)
    return quads


def quad_angles(q):
    def angle(a,b,c):
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    angles = []
    for i in range(4):
        prev = q[(i-1)%4]
        cur = q[i]
        nxt = q[(i+1)%4]
        angles.append(angle(prev, cur, nxt))
    return angles


def quad_similarity(q1, q2):
    def sides(q):
        s = []
        for i in range(4):
            a = q[i]
            b = q[(i+1)%4]
            s.append(np.linalg.norm(a-b))
        return np.array(s)
    s1 = sides(q1)
    s2 = sides(q2)
    side_score = 1.0 / (1.0 + np.mean(np.abs(s1/s1.mean() - s2/s2.mean())))
    a1 = np.array(quad_angles(q1))
    a2 = np.array(quad_angles(q2))
    angle_score = 1.0 / (1.0 + np.mean(np.abs(a1 - a2)) / 90.0)
    return float(0.6 * side_score + 0.4 * angle_score)


def order_quad_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def compute_P_from_chess(img_path, mtx, dist, cols, rows, square_size):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return None
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
    objp = []
    for r in range(rows):
        for c in range(cols):
            objp.append([c * square_size, r * square_size, 0.0])
    objp = np.array(objp, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    P = mtx.dot(np.hstack((R, tvec)))
    return {'P': P, 'R': R, 't': tvec, 'corners': corners2, 'objp': objp}



def pca_axes(points):
    C = points.mean(axis=0)
    X = points - C
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt.T
    return C, axes


def draw_quad(img, quad, color=(0,255,0), thickness=2):
    pts = quad.astype(int)
    for i in range(4):
        a = tuple(pts[i])
        b = tuple(pts[(i+1)%4])
        cv2.line(img, a, b, color, thickness)
    for i,p in enumerate(pts):
        cv2.circle(img, tuple(p), 4, color, -1)
        cv2.putText(img, str(i), tuple((p+np.array([5,-5]))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def interactive_cycle(img1_path, img2_path, quads1, quads2, mtx, dist, P1, P2, out_dir, base_name):
    # interactive_cycle now expects a pre-built candidate list; this function should
    # not build candidates itself in the refactored flow.
    raise RuntimeError('interactive_cycle should be called with a candidate list; use build_candidates() to build candidates first.')


def build_candidates(quads1, quads2, top_k=8):
    """Build and return the top_k candidate pairs as a list of (score, (i,j,k,l))."""
    candidates = []
    def pair_score(qa, qb):
        area_ratio = min(qa['area'], qb['area']) / (max(qa['area'], qb['area']) + 1e-8)
        if area_ratio < 0.05:
            return 0.0
        sim = quad_similarity(qa['pts'], qb['pts'])
        c1 = qa['pts'].mean(axis=0)
        c2 = qb['pts'].mean(axis=0)
        dist = np.linalg.norm(c1-c2)
        size = np.sqrt(max(qa['area'], qb['area']))
        proximity = 1.0 / (1.0 + dist / (size+1e-6))
        solidity_score = 0.5 * (qa.get('solidity',1.0) + qb.get('solidity',1.0))
        aspect_score = 1.0 - abs(qa.get('aspect',1.0) - qb.get('aspect',1.0)) / (max(qa.get('aspect',1.0), qb.get('aspect',1.0)) + 1e-8)
        return sim * proximity * area_ratio * solidity_score * (0.5 + 0.5 * aspect_score)

    for i in range(len(quads1)):
        for j in range(i+1, len(quads1)):
            score1 = pair_score(quads1[i], quads1[j])
            if score1 <= 0:
                continue
            for k in range(len(quads2)):
                for l in range(k+1, len(quads2)):
                    score2 = pair_score(quads2[k], quads2[l])
                    if score2 <= 0:
                        continue
                    ar1 = min(quads1[i]['area'], quads1[j]['area']) / (max(quads1[i]['area'], quads1[j]['area']) + 1e-8)
                    ar2 = min(quads2[k]['area'], quads2[l]['area']) / (max(quads2[k]['area'], quads2[l]['area']) + 1e-8)
                    cross = 1.0 / (1.0 + abs(ar1 - ar2) / (max(ar1, ar2) + 1e-8))
                    total_score = 0.5 * score1 + 0.5 * score2
                    total_score *= cross
                    candidates.append((total_score, (i,j,k,l)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return []
    if top_k is None:
        top_k = 8
    return candidates[:top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True)
    parser.add_argument('--img2', required=True)
    parser.add_argument('--calibration', default=os.path.join('..', 'phase_1', 'results', 'calibration_results.npz'))
    parser.add_argument('--board-cols', type=int, default=7)
    parser.add_argument('--board-rows', type=int, default=6)
    parser.add_argument('--square-size', type=float, default=0.025)
    parser.add_argument('--out-dir', default=os.path.join('results', 'cube_measure_interactive'))
    parser.add_argument('--min-area', type=int, default=2000)
    parser.add_argument('--top-k', type=int, default=8, help='Keep only top-K candidate pairs for interactive browsing')
    parser.add_argument('--list-only', action='store_true', help='List top-K candidates and save JSON, do not open GUI')
    parser.add_argument('--accept-rank', type=int, default=None, help='Non-interactively accept a ranked candidate (1-based) and save triangulation outputs')
    args = parser.parse_args()

    mtx, dist = cu.load_calibration(args.calibration)
    Pinfo1 = compute_P_from_chess(args.img1, mtx, dist, args.board_cols, args.board_rows, args.square_size)
    Pinfo2 = compute_P_from_chess(args.img2, mtx, dist, args.board_cols, args.board_rows, args.square_size)
    if Pinfo1 is None or Pinfo2 is None:
        print('Chessboard not found in both images; cannot proceed')
        return

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    quads1 = detect_quads(img1, min_area=args.min_area)
    quads2 = detect_quads(img2, min_area=args.min_area)
    if not quads1 or not quads2:
        print('No significant quadrilaterals found in one or both images; try lowering --min-area')
        return

    base_name = os.path.splitext(os.path.basename(args.img1))[0] + '_vs_' + os.path.splitext(os.path.basename(args.img2))[0]
    candidates = build_candidates(quads1, quads2, top_k=args.top_k)
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(os.path.abspath(args.out_dir), f'cube_candidates_{base_name}.json')
    # prepare a serializable summary
    summary = []
    for idx, (score, (i,j,k,l)) in enumerate(candidates):
        entry = {
            'rank': idx+1,
            'score': float(score),
            'img1_quads': {
                'i': int(i), 'j': int(j),
                'area_i': float(quads1[i]['area']), 'area_j': float(quads1[j]['area']),
                'solidity_i': float(quads1[i].get('solidity', 0.0)), 'solidity_j': float(quads1[j].get('solidity', 0.0)),
                'aspect_i': float(quads1[i].get('aspect', 0.0)), 'aspect_j': float(quads1[j].get('aspect', 0.0)),
                'angles_i': [float(a) for a in quads1[i].get('angles',[])], 'angles_j': [float(a) for a in quads1[j].get('angles',[])],
            },
            'img2_quads': {
                'k': int(k), 'l': int(l),
                'area_k': float(quads2[k]['area']), 'area_l': float(quads2[l]['area']),
                'solidity_k': float(quads2[k].get('solidity', 0.0)), 'solidity_l': float(quads2[l].get('solidity', 0.0)),
                'aspect_k': float(quads2[k].get('aspect', 0.0)), 'aspect_l': float(quads2[l].get('aspect', 0.0)),
                'angles_k': [float(a) for a in quads2[k].get('angles',[])], 'angles_l': [float(a) for a in quads2[l].get('angles',[])],
            }
        }
        summary.append(entry)

    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump({'pairs': summary}, fh, indent=2)

    print(f'Wrote top-{len(summary)} candidate summary to: {out_json}')
    for e in summary:
        print(f"Rank {e['rank']}: score={e['score']:.3f} img1_quads=({e['img1_quads']['i']},{e['img1_quads']['j']}) img2_quads=({e['img2_quads']['k']},{e['img2_quads']['l']})")

    if args.list_only:
        return

    # If accept-rank is provided, perform non-interactive accept and save outputs
    if args.accept_rank is not None:
        rank_idx = int(args.accept_rank) - 1
        if rank_idx < 0 or rank_idx >= len(candidates):
            print(f'accept-rank {args.accept_rank} out of range (1..{len(candidates)})')
            return
        score, (i,j,k,l) = candidates[rank_idx]
        pts_img1 = np.vstack((quads1[i]['pts'], quads1[j]['pts'])).astype(np.float32)
        pts_img2 = np.vstack((quads2[k]['pts'], quads2[l]['pts'])).astype(np.float32)
        pts3d = cu.triangulate_points(Pinfo1['P'], Pinfo2['P'], pts_img1.reshape(-1,1,2), pts_img2.reshape(-1,1,2), mtx, dist)
        C, axes = pca_axes(pts3d)
        proj = (pts3d - C).dot(axes)
        mins = proj.min(axis=0)
        maxs = proj.max(axis=0)
        dims = (maxs - mins)
        out = {
            'img1': args.img1,
            'img2': args.img2,
            'candidate_rank': int(args.accept_rank),
            'score': float(score),
            'num_points': int(pts3d.shape[0]),
            'centroid_m': C.tolist(),
            'axes': axes.tolist(),
            'dimensions_m': {'dim0': float(dims[0]), 'dim1': float(dims[1]), 'dim2': float(dims[2])}
        }
        npz_path = os.path.join(args.out_dir, f'cube_{base_name}.npz')
        json_path = os.path.join(args.out_dir, f'cube_{base_name}.json')
        np.savez(npz_path, pts3d=pts3d, pts_img1=pts_img1, pts_img2=pts_img2)
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=2)
        vis1_a = img1.copy()
        vis2_a = img2.copy()
        for i_pt, p in enumerate(pts_img1.reshape(-1,2)):
            cv2.circle(vis1_a, tuple(p.astype(int)), 5, (0,255,0), -1)
            cv2.putText(vis1_a, str(i_pt), tuple((p+np.array([5,-5])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        for i_pt, p in enumerate(pts_img2.reshape(-1,2)):
            cv2.circle(vis2_a, tuple(p.astype(int)), 5, (0,255,0), -1)
            cv2.putText(vis2_a, str(i_pt), tuple((p+np.array([5,-5])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(os.path.join(args.out_dir, f'cube_vis_{base_name}_1.jpg'), vis1_a)
        cv2.imwrite(os.path.join(args.out_dir, f'cube_vis_{base_name}_2.jpg'), vis2_a)

        # compute reprojection error back to images using Pinfo R/t
        def reproj_error(pts3d, R, tvec, mtx, dist, pts2d):
            rvec, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(pts3d, rvec, tvec, mtx, dist)
            proj = proj.reshape(-1,2)
            err = np.linalg.norm(proj - pts2d.reshape(-1,2), axis=1)
            return float(err.mean()), float(err.max())

        mean1, max1 = reproj_error(pts3d, Pinfo1['R'], Pinfo1['t'], mtx, dist, pts_img1)
        mean2, max2 = reproj_error(pts3d, Pinfo2['R'], Pinfo2['t'], mtx, dist, pts_img2)

        print('Saved measurement:', json_path, npz_path)
        print(f'Reprojection error img1 mean={mean1:.3f}px max={max1:.3f}px')
        print(f'Reprojection error img2 mean={mean2:.3f}px max={max2:.3f}px')
        print('Dimensions (meters):', out['dimensions_m'])
        return

    # interactive GUI path (best-first) -- build a simple browser using the saved candidates
    # Reconstruct candidate list for interactive display
    # Note: for now reuse the 'candidates' list and implement a basic GUI loop
    win = 'cube_interactive'
    window_created = False
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        window_created = True
        idx = 0
        while True:
            score, (i,j,k,l) = candidates[idx]
            vis1 = img1.copy()
            vis2 = img2.copy()
            draw_quad(vis1, quads1[i]['pts'], (0,255,0))
            draw_quad(vis1, quads1[j]['pts'], (0,0,255))
            draw_quad(vis2, quads2[k]['pts'], (0,255,0))
            draw_quad(vis2, quads2[l]['pts'], (0,0,255))
            h = max(vis1.shape[0], vis2.shape[0])
            w = vis1.shape[1] + vis2.shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:vis1.shape[0], :vis1.shape[1]] = vis1
            canvas[:vis2.shape[0], vis1.shape[1]:] = vis2
            cv2.putText(canvas, f'Rank {idx+1}/{len(candidates)} Score: {score:.3f} (n/p:nav, a:accept, q:quit)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(win, canvas)
            try:
                key = cv2.waitKey(0) & 0xFF
            except KeyboardInterrupt:
                print('\nInterrupted by user (KeyboardInterrupt). Closing GUI.')
                break
            if key == ord('n'):
                idx = (idx + 1) % len(candidates)
            elif key == ord('p'):
                idx = (idx - 1) % len(candidates)
            elif key == ord('q'):
                break
            elif key == ord('a'):
                # accept candidate and perform triangulation+save
                score, (i,j,k,l) = candidates[idx]
                pts_img1 = np.vstack((quads1[i]['pts'], quads1[j]['pts'])).astype(np.float32)
                pts_img2 = np.vstack((quads2[k]['pts'], quads2[l]['pts'])).astype(np.float32)
                pts3d = cu.triangulate_points(Pinfo1['P'], Pinfo2['P'], pts_img1.reshape(-1,1,2), pts_img2.reshape(-1,1,2), mtx, dist)
                C, axes = pca_axes(pts3d)
                proj = (pts3d - C).dot(axes)
                mins = proj.min(axis=0)
                maxs = proj.max(axis=0)
                dims = (maxs - mins)
                out = {
                    'img1': args.img1,
                    'img2': args.img2,
                    'candidate_idx': idx,
                    'num_points': int(pts3d.shape[0]),
                    'centroid_m': C.tolist(),
                    'axes': axes.tolist(),
                    'dimensions_m': {'dim0': float(dims[0]), 'dim1': float(dims[1]), 'dim2': float(dims[2])}
                }
                npz_path = os.path.join(args.out_dir, f'cube_{base_name}.npz')
                json_path = os.path.join(args.out_dir, f'cube_{base_name}.json')
                np.savez(npz_path, pts3d=pts3d, pts_img1=pts_img1, pts_img2=pts_img2)
                with open(json_path, 'w', encoding='utf-8') as fh:
                    json.dump(out, fh, indent=2)
                vis1_a = vis1.copy()
                vis2_a = vis2.copy()
                for i_pt, p in enumerate(pts_img1.reshape(-1,2)):
                    cv2.circle(vis1_a, tuple(p.astype(int)), 5, (0,255,0), -1)
                    cv2.putText(vis1_a, str(i_pt), tuple((p+np.array([5,-5])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for i_pt, p in enumerate(pts_img2.reshape(-1,2)):
                    cv2.circle(vis2_a, tuple(p.astype(int)), 5, (0,255,0), -1)
                    cv2.putText(vis2_a, str(i_pt), tuple((p+np.array([5,-5])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imwrite(os.path.join(args.out_dir, f'cube_vis_{base_name}_1.jpg'), vis1_a)
                cv2.imwrite(os.path.join(args.out_dir, f'cube_vis_{base_name}_2.jpg'), vis2_a)
                print('Saved measurement:', json_path, npz_path)
                break
    except Exception as exc:
        print('Error during interactive GUI:', exc)
    finally:
        if window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == '__main__':
    main()
