"""
Geometry-based transpedicular trajectory planner for vertebroplasty (3D Slicer).

Computes the maximum-clearance transpedicular corridor from a posterolateral entry
on the posterior elements to a target in the anterior vertebral body, given a CT and
a per-vertebra segmentation.

QUICK START (Slicer Python console)
-----------------------------------
    exec(open('/home/michelle/Downloads/transpedicular_planner.py').read())

    ct, seg = loadCase('/path/sub-xxx_ct.nii.gz', '/path/sub-xxx_seg-vert_msk.nii.gz')
    listSegments(ct, seg)
    p, res = planVertebra(ct, seg, 'Segment_11')
    clearPlans()

LAYOUT
------
planVertebra() switches to a custom three-panel layout:

    left         3D view - vertebra surface, needle cylinder, trajectory
    top right    slice reformatted ALONG the trajectory (needle seen in-plane)
    bottom right clearance-vs-depth chart

The reformatted slice plane is NOT drawn in the 3D view by default - it contains
the trajectory, so drawing it would hide the needle. Pass showSlicePlane=True if
you want it. restoreLayout() returns to the standard four-up.

CLEARANCE PLOT
--------------
X is depth from the bone entry point, so the curve reads left to right as the
needle advances. Two horizontal reference lines: needle radius (below it = cortex
breach) and needle radius + safety margin (below it = contained but no margin).
The dip is the pedicle isthmus - that minimum is the number the whole plan turns on.

COLOURING
---------
Trajectories are coloured by CONTAINMENT of the physical needle, not by the margin:

    green  clearance >= needleRadius + safetyMargin   comfortably inside
    amber  needleRadius <= clearance < required       inside, but no margin left
    red    clearance <  needleRadius                  cortex breached

Pass colorBy='margin' to collapse amber into red (strict two-state).

Other entry points:
    p, res = planVertebra(ct, seg, 'Segment_11', sides=('right',), needleDiameterMm=2.4)
    planAll(ct, seg)                      # sweep every segment, summary table only
    p.debugSweep(side='right')            # why candidates are being rejected
    p.evaluateLine(getNode('MyLine'))     # score a hand-drawn trajectory

NOTE: after editing this file, re-run exec() AND rebuild the planner object.
Existing objects keep pointing at the old class definition.

NOTE: assumes the CT volume node has no parent transform (or that it is hardened).
"""

import numpy as np
import vtk
import slicer
from scipy import ndimage, optimize


# ----------------------------------------------------------------------------- utils
def _mat(node, which):
    m = vtk.vtkMatrix4x4()
    getattr(node, f'Get{which}Matrix')(m)
    return slicer.util.arrayFromVTKMatrix(m)


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


INFEASIBLE = 1e6

GREEN = (0.20, 0.85, 0.35)
AMBER = (0.95, 0.70, 0.15)
RED = (0.95, 0.25, 0.20)
GREY = (0.55, 0.55, 0.55)


def _statusColor(r, colorBy='containment'):
    if not r['containsNeedle']:
        return RED, 'BREACH'
    if r['feasible']:
        return GREEN, 'OK'
    return (AMBER, 'NO MARGIN') if colorBy == 'containment' else (RED, 'NO MARGIN')


def _show2D(displayNode, visible=True, thickness=2):
    """SetSliceIntersectionVisibility/Thickness are deprecated; prefer the new API."""
    if hasattr(displayNode, 'SetVisibility2D'):
        displayNode.SetVisibility2D(visible)
    else:
        displayNode.SetSliceIntersectionVisibility(visible)
    if hasattr(displayNode, 'SetLineWidth'):
        displayNode.SetLineWidth(thickness)


# ----------------------------------------------------------------------------- planner
class TranspedicularPlanner:

    def __init__(self, ctNode, segNode, segmentId,
                 needleDiameterMm=3.05,      # 11G outer diameter
                 safetyMarginMm=1.0,         # required clearance beyond needle radius
                 tipToAnteriorCortexMm=8.0,  # tip depth: anterior third of the body
                 entryExcludeMm=4.0,         # ignore clearance this close to entry cortex
                 minCorridorMm=35.0,         # a real transpedicular path is 40-50 mm
                 minPosteriorSpanMm=10.0,    # path length behind the posterior body wall
                 cropPadMm=15.0,             # margin kept around the vertebra
                 verbose=True):

        self.ct, self.seg, self.segmentId = ctNode, segNode, segmentId
        self.needleDiameterMm = needleDiameterMm
        self.needleRadius = needleDiameterMm / 2.0
        self.safetyMarginMm = safetyMarginMm
        self.requiredClearance = self.needleRadius + safetyMarginMm
        self.tipDepth = tipToAnteriorCortexMm
        self.entryExcludeMm = entryExcludeMm
        self.minCorridorMm = minCorridorMm
        self.minPosteriorSpanMm = minPosteriorSpanMm
        self.cropPadMm = cropPadMm
        self.verbose = verbose

        # angular limits (deg) in the vertebra frame; soft-penalized, not hard bounds
        self.alphaRange = (0.0, 35.0)    # medial angulation
        self.betaMax = 20.0              # cranio-caudal angulation

        self._buildFields()
        self._buildFrame()
        self._buildLandmarks()

    # ------------------------------------------------------------------ image fields
    def _buildFields(self):
        """Crop to the vertebra + margin, then build the signed clearance field.

        Cropping matters: a full spine CT is ~100 Mvoxel and the distance transform
        over that takes minutes. All RAS<->IJK matrices below refer to the CROPPED
        subvolume. cropPadMm also bounds how far _skinEntry can search outward.
        """
        ijkToRasFull = _mat(self.ct, 'IJKToRAS')
        spacing = np.array(self.ct.GetSpacing())       # along i, j, k
        self.sampling_kji = spacing[::-1]              # array axes are [k, j, i]
        self.worstSpacingMm = float(spacing.max())

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.seg, self.segmentId, self.ct) > 0
        if not mask.any():
            raise ValueError(f'Segment {self.segmentId!r} is empty on the CT grid.')

        kji = np.argwhere(mask)
        pad = np.ceil(self.cropPadMm / self.sampling_kji).astype(int)
        lo = np.maximum(kji.min(0) - pad, 0)
        hi = np.minimum(kji.max(0) + 1 + pad, np.array(mask.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        if self.verbose:
            dims = tuple(int(v) for v in (hi - lo))
            print(f'crop {mask.shape} -> {dims} ({np.prod(hi - lo) / 1e6:.1f} Mvoxel)')

        # shift the IJK origin to the crop corner; ijkToRas acts on [i, j, k, 1]
        shift = np.eye(4)
        shift[:3, 3] = [lo[2], lo[1], lo[0]]
        self.ijkToRas = ijkToRasFull @ shift
        self.rasToIjk = np.linalg.inv(self.ijkToRas)

        self.ctArr = slicer.util.arrayFromVolume(self.ct)[sl].astype(np.float32)

        # 3D fill closes trabecular voids but leaves the vertebral foramen open,
        # since the canal is a through-channel connected to background.
        self.vert = ndimage.binary_fill_holes(mask[sl])

        # Signed clearance field: >0 inside the vertebra = distance to cortex (mm).
        inside = ndimage.distance_transform_edt(self.vert, sampling=self.sampling_kji)
        outside = ndimage.distance_transform_edt(~self.vert, sampling=self.sampling_kji)
        self.edt = (inside - outside).astype(np.float32)

    def _sample(self, field, pts, cval=-1e3):
        """Trilinear sample a [k,j,i] array at an (N,3) array of RAS points."""
        pts = np.atleast_2d(pts)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        ijk = (self.rasToIjk @ h.T).T[:, :3]
        coords = np.vstack([ijk[:, 2], ijk[:, 1], ijk[:, 0]])   # k, j, i
        return ndimage.map_coordinates(field, coords, order=1,
                                       mode='constant', cval=cval)

    def _maskPointsRAS(self, mask, maxPoints=8000, seed=0):
        kji = np.argwhere(mask)
        if len(kji) > maxPoints:
            idx = np.random.default_rng(seed).choice(len(kji), maxPoints, replace=False)
            kji = kji[idx]
        h = np.hstack([kji[:, ::-1].astype(float), np.ones((len(kji), 1))])
        return (self.ijkToRas @ h.T).T[:, :3]

    # ------------------------------------------------------------------ local frame
    def _buildFrame(self):
        P = self._maskPointsRAS(self.vert)
        c = P.mean(0)
        maskF = self.vert.astype(np.float32)

        # --- midsagittal symmetry plane: maximize overlap of mask with its mirror.
        # n(a,b) = normalize([1, a, b]) starts at RAS +R; d is the plane offset.
        def negOverlap(x):
            a, b, d = x
            n = _unit([1.0, a, b])
            proj = (P - c) @ n - d
            mirrored = P - 2.0 * proj[:, None] * n[None, :]
            return -float(self._sample(maskF, mirrored, cval=0.0).mean())

        res = optimize.minimize(negOverlap, [0.0, 0.0, 0.0], method='Powell',
                                options=dict(xtol=1e-3, ftol=1e-4, maxiter=200))
        a, b, d = res.x
        ml = _unit([1.0, a, b])                     # points to patient RIGHT
        self.symmetryOverlap = -res.fun

        # --- SI: smallest-variance inertia axis (vertebra is flattest cranio-caudally)
        w, V = np.linalg.eigh(np.cov((P - c).T))
        si = V[:, 0]
        if si[2] < 0:
            si = -si                                # disambiguate toward RAS +S
        si = _unit(si - (si @ ml) * ml)

        # --- AP: complete the frame, orient anteriorly.
        ap = _unit(np.cross(si, ml))
        if ap[1] < 0:
            ap = -ap                                # RAS +A
            si = -si
        # sanity: spinous process is a long posterior tail -> negative skew along +ap
        skew = float((((P - c) @ ap) ** 3).mean())
        warnings = []
        if skew > 0:
            warnings.append('Anterior direction ambiguous (mass skew points anteriorly).')
        if self.symmetryOverlap < 0.80:
            warnings.append(f'Weak symmetry fit ({self.symmetryOverlap:.2f}); segment '
                            'may be truncated by the field of view.')
        self.frameWarning = ' '.join(warnings) if warnings else None

        self.origin, self.ml, self.ap, self.si = c, ml, ap, si

    def _buildLandmarks(self):
        """Anchor on the vertebral body, NOT the whole-vertebra centroid.

        The centroid of a full vertebra mask (body + pedicles + laminae + spinous
        process) lands inside the spinal canal, which is outside the mask, so any
        ray marched from there exits immediately. The global argmax of the distance
        field is the centre of the largest inscribed sphere: the body core.

        apPosteriorWall is expressed relative to self.origin. It is normally a small
        number of either sign - it just says where the body's back wall sits relative
        to the whole-vertebra centroid, which itself lies around the canal.
        """
        k, j, i = np.unravel_index(np.argmax(self.edt), self.edt.shape)
        bc = (self.ijkToRas @ np.array([i, j, k, 1.0]))[:3]
        # snap to midsagittal: a symmetric mask's centroid lies on its symmetry plane
        bc = bc - ((bc - self.origin) @ self.ml) * self.ml
        self.bodyCenter = bc

        s = np.arange(0.0, 60.0, 0.25)
        dist = self._sample(self.edt, bc[None, :] - s[:, None] * self.ap[None, :])
        if dist[0] <= 0:
            raise RuntimeError('Body anchor is outside the mask - check the ap sign.')
        out = np.nonzero(dist <= 0.0)[0]
        if not len(out):
            raise RuntimeError('No posterior body wall found within 60 mm.')
        self.apPosteriorWall = float(((bc - s[out[0]] * self.ap) - self.origin) @ self.ap)
        self.bodyDepthPosteriorMm = float(s[out[0]])

    # ------------------------------------------------------------------ geometry
    def _target(self, mlOff, siOff):
        """Anterior-body target: march anteriorly, back off from the cortex."""
        base = self.bodyCenter + mlOff * self.ml + siOff * self.si
        s = np.arange(0.0, 70.0, 0.25)
        dist = self._sample(self.edt, base[None, :] + s[:, None] * self.ap[None, :])
        if dist[0] <= 0:
            return None
        out = np.nonzero(dist <= 0.0)[0]
        if not len(out):
            return None
        return base + (s[out[0]] - self.tipDepth) * self.ap

    def _direction(self, alphaDeg, betaDeg, sideSign):
        """Unit vector from target back toward the entry point."""
        a, b = np.radians(alphaDeg), np.radians(betaDeg)
        return _unit(-self.ap
                     + np.tan(a) * sideSign * self.ml
                     + np.tan(b) * self.si)

    def _corridor(self, T, d, maxLenMm=120.0, stepMm=0.25):
        """March from target to entry; return clearance profile and exit point.

        posteriorSpanMm is the length of path lying behind the posterior body wall,
        i.e. inside the posterior elements. That region IS the pedicle, so this is
        what distinguishes a true transpedicular corridor from a path that simply
        exits through the lateral wall of the vertebral body - the latter scores
        high clearance (fat cancellous bone all the way) but never enters a pedicle.

        breachLengthMm is how much of the path has less clearance than the needle
        radius, i.e. how much of the needle surface is outside the cortex.

        The returned profile is ordered ENTRY -> TARGET so it plots left to right
        in the direction the needle actually advances.
        """
        if T is None:
            return None
        s = np.arange(0.0, maxLenMm, stepMm)
        pts = T[None, :] + s[:, None] * d[None, :]
        dist = self._sample(self.edt, pts)
        if dist[0] <= 0:
            return None
        out = np.nonzero(dist <= 0.0)[0]
        if not len(out):
            return None
        iExit = out[0]
        length = float(s[iExit])
        if length < self.minCorridorMm:
            return None

        apCoord = (pts[:iExit] - self.origin) @ self.ap
        posteriorSpan = float(np.count_nonzero(apCoord < self.apPosteriorWall) * stepMm)

        iEnd = max(1, iExit - int(self.entryExcludeMm / stepMm))
        scored = dist[:iEnd]
        iMin = int(np.argmin(scored))
        breachLen = float(np.count_nonzero(scored < self.needleRadius) * stepMm)

        depthFromEntry = (length - s[:iExit])[::-1]
        clearanceFromEntry = dist[:iExit][::-1]

        return dict(entry=T + s[iExit] * d,
                    length=length,
                    clearance=float(scored.min()),
                    narrowestAtMm=float(s[iMin]),
                    narrowestDepthMm=float(length - s[iMin]),
                    narrowestPoint=T + s[iMin] * d,
                    breachLengthMm=breachLen,
                    posteriorSpanMm=posteriorSpan,
                    profile=(depthFromEntry, clearanceFromEntry))

    def _evaluate(self, x, sideSign):
        """Return (cost, target, direction, corridor). Cost is INFEASIBLE if rejected."""
        alpha, beta, mlOff, siOff = x
        T = self._target(mlOff, siOff)
        if T is None:
            return INFEASIBLE, None, None, None
        d = self._direction(alpha, beta, sideSign)
        c = self._corridor(T, d)
        if c is None:
            return INFEASIBLE, T, d, None
        # entry must lie on the posterior elements, not the lateral body wall
        if (c['entry'] - self.origin) @ self.ap > self.apPosteriorWall:
            return INFEASIBLE, T, d, None
        # and the path must actually traverse the pedicle, not just end near it
        if c['posteriorSpanMm'] < self.minPosteriorSpanMm:
            return INFEASIBLE, T, d, None

        cost = -c['clearance']
        lo, hi = self.alphaRange
        cost += 0.05 * (max(0.0, alpha - hi) ** 2 + max(0.0, lo - alpha) ** 2)
        cost += 0.05 * max(0.0, abs(beta) - self.betaMax) ** 2
        cost += 0.02 * (mlOff ** 2 + siOff ** 2)     # stay near the nominal target
        return cost, T, d, c

    def plan(self, side='left', mlBound=6.0, siBound=4.0):
        """side: 'left'/'right' pedicle (patient anatomical side).

        The cost surface has large flat INFEASIBLE plateaus, and Powell can finish
        standing on one - res.x is therefore not reliably a feasible point. So the
        best FEASIBLE point actually evaluated is tracked explicitly and used.
        """
        sideSign = +1.0 if side == 'right' else -1.0
        best = dict(cost=np.inf, x=None)

        def cost(x):
            v = self._evaluate(x, sideSign)[0]
            if v < best['cost']:
                best['cost'], best['x'] = v, np.array(x, dtype=float)
            return v

        # coarse sweep over the angular cone
        for alpha in np.arange(-5.0, 45.0, 2.0):
            for beta in np.arange(-25.0, 25.1, 2.0):
                cost(np.array([alpha, beta, 0.0, 0.0]))
        if best['x'] is None or best['cost'] >= INFEASIBLE:
            raise RuntimeError(
                f'No corridor found on the {side} side. Inspect with '
                f'p.debugSweep(side="{side}") to see which constraint is binding.')

        # refine all 4 DOF; `best` keeps the answer regardless of where Powell stops
        optimize.minimize(cost, best['x'].copy(), method='Powell',
                          bounds=[(-10, 50), (-30, 30),
                                  (-mlBound, mlBound), (-siBound, siBound)],
                          options=dict(xtol=1e-3, ftol=1e-4, maxiter=500))

        alpha, beta, mlOff, siOff = best['x']
        _, T, d, c = self._evaluate(best['x'], sideSign)
        if c is None:
            raise RuntimeError(f'Internal error: best point on the {side} side is '
                               'not reproducible. Please report the segment id.')

        E = c['entry']
        r = dict(side=side, segmentId=self.segmentId,
                 target=T, entry=E, direction=_unit(T - E),
                 clearance=c['clearance'], corridorLengthMm=c['length'],
                 posteriorSpanMm=c['posteriorSpanMm'],
                 narrowestAtMm=c['narrowestAtMm'],
                 narrowestDepthMm=c['narrowestDepthMm'],
                 narrowestPoint=c['narrowestPoint'],
                 breachLengthMm=c['breachLengthMm'],
                 axialAngleDeg=float(alpha), sagittalAngleDeg=float(beta),
                 targetOffsetMlMm=float(mlOff), targetOffsetSiMm=float(siOff),
                 profile=c['profile'],
                 needleDiameterMm=self.needleDiameterMm,
                 needleRadius=self.needleRadius,
                 safetyMarginMm=self.safetyMarginMm,
                 requiredClearance=self.requiredClearance,
                 containsNeedle=c['clearance'] >= self.needleRadius,
                 feasible=c['clearance'] >= self.requiredClearance,
                 breachMm=max(0.0, self.needleRadius - c['clearance']),
                 marginDeficitMm=max(0.0, self.requiredClearance - c['clearance']),
                 skinEntry=self._skinEntry(E, _unit(E - T)))
        r['pose'] = self._pose(E, r['direction'])
        return r

    def debugSweep(self, side='left'):
        """Report why candidates were rejected across the coarse angular sweep."""
        sideSign = +1.0 if side == 'right' else -1.0
        tally = dict(noTarget=0, noCorridor=0, tooShort=0,
                     entryTooAnterior=0, notThroughPedicle=0, accepted=0)
        bestClear, bestAt = 0.0, None
        for alpha in np.arange(-5.0, 45.0, 2.0):
            for beta in np.arange(-25.0, 25.1, 2.0):
                T = self._target(0.0, 0.0)
                if T is None:
                    tally['noTarget'] += 1
                    continue
                d = self._direction(alpha, beta, sideSign)
                s = np.arange(0.0, 120.0, 0.25)
                pts = T[None, :] + s[:, None] * d[None, :]
                dist = self._sample(self.edt, pts)
                out = np.nonzero(dist <= 0.0)[0]
                if dist[0] <= 0 or not len(out):
                    tally['noCorridor'] += 1
                    continue
                if s[out[0]] < self.minCorridorMm:
                    tally['tooShort'] += 1
                    continue
                c = self._corridor(T, d)
                if (c['entry'] - self.origin) @ self.ap > self.apPosteriorWall:
                    tally['entryTooAnterior'] += 1
                    continue
                if c['posteriorSpanMm'] < self.minPosteriorSpanMm:
                    tally['notThroughPedicle'] += 1
                    continue
                tally['accepted'] += 1
                if c['clearance'] > bestClear:
                    bestClear, bestAt = c['clearance'], (alpha, beta)
        print(f'--- coarse sweep, {side} side ---')
        for k, v in tally.items():
            print(f'  {k:20s} {v}')
        if bestAt:
            print(f'  best clearance {bestClear:.2f} mm at '
                  f'alpha={bestAt[0]:.0f} beta={bestAt[1]:.0f}')
        return tally

    # ------------------------------------------------------------------ outputs
    def _skinEntry(self, E, outward, maxLenMm=250.0, airHU=-500.0):
        """Bounded by cropPadMm: returns None if the skin is outside the crop."""
        s = np.arange(0.0, maxLenMm, 0.5)
        pts = E[None, :] + s[:, None] * outward[None, :]
        hu = self._sample(self.ctArr, pts, cval=-1000.0)
        air = hu < airHU
        run, need = 0, 20                      # first >=10 mm of continuous air
        for i, a in enumerate(air):
            run = run + 1 if a else 0
            if run >= need:
                return E + s[i - need + 1] * outward
        return None

    def _pose(self, E, insertionDir):
        """4x4 RAS pose: origin at entry, +Z along insertion."""
        z = _unit(insertionDir)
        x = self.ml - (self.ml @ z) * z
        if np.linalg.norm(x) < 1e-6:
            x = self.si - (self.si @ z) * z
        x = _unit(x)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, np.cross(z, x), z, E
        return M

    def createNodes(self, r, name='Trajectory', makeTransform=True,
                    colorBy='containment'):
        col, _ = _statusColor(r, colorBy)
        line = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', name)
        start = r['skinEntry'] if r['skinEntry'] is not None else r['entry']
        line.AddControlPoint(vtk.vtkVector3d(*start), 'entry')
        line.AddControlPoint(vtk.vtkVector3d(*r['target']), 'target')
        dn = line.GetDisplayNode()
        dn.SetSelectedColor(*col)
        dn.SetColor(*col)

        if makeTransform:
            t = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode',
                                                   name + '_pose')
            m = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    m.SetElement(i, j, r['pose'][i, j])
            t.SetMatrixTransformToParent(m)
        return line

    def createNeedleModel(self, r, name='Needle', colorBy='containment'):
        """Cylinder at the true needle diameter, coloured by containment.

        A line has no thickness, so a 2 mm and a 5 mm clearance look identical until
        the real width is rendered.
        """
        col, _ = _statusColor(r, colorBy)
        E, T = r['entry'], r['target']
        axis = T - E
        L = float(np.linalg.norm(axis))

        cyl = vtk.vtkCylinderSource()
        cyl.SetRadius(self.needleRadius)
        cyl.SetHeight(L)
        cyl.SetResolution(32)
        cyl.Update()

        z = axis / L
        y = np.array([0.0, 1.0, 0.0])          # vtkCylinderSource is +Y aligned
        v = np.cross(y, z)
        cth = float(np.dot(y, z))
        if np.linalg.norm(v) < 1e-8:
            R = np.eye(3) * (1.0 if cth > 0 else -1.0)
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + K + K @ K * (1.0 / (1.0 + cth))

        M = vtk.vtkMatrix4x4()
        mid = (E + T) / 2.0
        for i in range(3):
            for j in range(3):
                M.SetElement(i, j, R[i, j])
            M.SetElement(i, 3, mid[i])

        tf = vtk.vtkTransform()
        tf.SetMatrix(M)
        tfp = vtk.vtkTransformPolyDataFilter()
        tfp.SetInputData(cyl.GetOutput())
        tfp.SetTransform(tf)
        tfp.Update()

        model = slicer.modules.models.logic().AddModel(tfp.GetOutput())
        model.SetName(name)
        dn = model.GetDisplayNode()
        dn.SetColor(*col)
        dn.SetOpacity(0.8)
        _show2D(dn, True, 2)
        return model

    def markNarrowest(self, r, name='Isthmus'):
        """Fiducial at the tightest point of the corridor - where a breach starts."""
        f = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', name)
        f.AddControlPoint(vtk.vtkVector3d(*r['narrowestPoint']),
                          f"{r['clearance']:.2f} mm")
        col, _ = _statusColor(r)
        f.GetDisplayNode().SetSelectedColor(*col)
        return f

    def plotProfile(self, r, name=None):
        """Clearance chart for a single side."""
        return plotClearance(self, {r['side']: r},
                             name=name or f"{r['segmentId']} clearance")

    def report(self, r, colorBy='containment'):
        _, status = _statusColor(r, colorBy)
        insertion = float(np.linalg.norm(r['target'] - r['entry']))
        lines = [
            f"--- {r['segmentId']} / {r['side']} transpedicular corridor ---",
            f"  status                  : {status}",
            f"  min clearance to cortex : {r['clearance']:.2f} mm  "
            f"(needle radius {r['needleRadius']:.2f}, "
            f"required {r['requiredClearance']:.2f})",
        ]
        if not r['containsNeedle']:
            lines.append(f"  cortex breach           : {r['breachMm']:.2f} mm deep, "
                         f"over {r['breachLengthMm']:.1f} mm of path")
        elif not r['feasible']:
            lines.append(f"  needle fully contained, but {r['marginDeficitMm']:.2f} mm "
                         f"short of the safety margin")
        lines += [
            f"  narrowest point         : {r['narrowestDepthMm']:.1f} mm from entry",
            f"  bone insertion depth    : {insertion:.1f} mm",
            f"  span behind body wall   : {r['posteriorSpanMm']:.1f} mm "
            f"(pedicle traversal)",
            f"  axial (medial) angle    : {r['axialAngleDeg']:.1f} deg",
            f"  sagittal angle          : {r['sagittalAngleDeg']:.1f} deg",
            f"  entry (RAS)             : "
            f"({r['entry'][0]:.1f}, {r['entry'][1]:.1f}, {r['entry'][2]:.1f})",
            f"  target (RAS)            : "
            f"({r['target'][0]:.1f}, {r['target'][1]:.1f}, {r['target'][2]:.1f})",
        ]
        if r['skinEntry'] is not None:
            standoff = float(np.linalg.norm(r['entry'] - r['skinEntry']))
            lines.append(f"  skin-to-bone standoff   : {standoff:.1f} mm")
        else:
            lines.append("  skin entry              : not found (raise cropPadMm)")
        if abs(r['sagittalAngleDeg']) > self.betaMax - 1.0:
            lines.append(f"  NOTE: sagittal angle is at the {self.betaMax:.0f} deg "
                         "limit; the optimum may lie outside it.")
        if r['clearance'] < self.worstSpacingMm:
            lines.append(f"  NOTE: clearance is below the coarsest voxel dimension "
                         f"({self.worstSpacingMm:.2f} mm); treat it as approximate.")
        if self.frameWarning:
            lines.append(f"  WARNING: {self.frameWarning}")
        print('\n'.join(lines))

    # ------------------------------------------------------------------ validation
    def evaluateLine(self, lineNode):
        """Score an existing (e.g. clinician-drawn) markups line with the same metric."""
        E = np.array(lineNode.GetNthControlPointPositionWorld(0))
        T = np.array(lineNode.GetNthControlPointPositionWorld(1))
        c = self._corridor(T, _unit(E - T))
        if c is None:
            return dict(containsNeedle=False, feasible=False, clearance=None,
                        note='Path does not lie inside the vertebra.')
        return dict(clearance=c['clearance'],
                    containsNeedle=c['clearance'] >= self.needleRadius,
                    feasible=c['clearance'] >= self.requiredClearance,
                    breachMm=max(0.0, self.needleRadius - c['clearance']),
                    breachLengthMm=c['breachLengthMm'],
                    corridorLengthMm=c['length'],
                    posteriorSpanMm=c['posteriorSpanMm'])

    def reformatToTrajectory(self, r, sliceViewName='Red'):
        """Reformat a slice view to contain the trajectory, so the needle is seen
        in-plane along its whole length rather than as a dot crossing axial slices."""
        w = slicer.app.layoutManager().sliceWidget(sliceViewName)
        if w is None:
            return
        n = r['direction']
        t = _unit(np.cross(n, self.si)) if abs(n @ self.si) < 0.95 \
            else _unit(np.cross(n, self.ml))
        w.mrmlSliceNode().SetSliceToRASByNTP(*n, *t, *r['target'], 0)


# =========================================================================== plotting
def _plotSeries(chart, name, x, y, color, markers=False):
    """Add one series to a chart, backed by its own table node.

    One table per series because sides have different corridor lengths, so their
    x-vectors do not align and cannot share a table.
    """
    table = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', name + ' table')
    slicer.util.updateTableFromArray(
        table, np.column_stack([np.asarray(x, float), np.asarray(y, float)]),
        ['depth (mm)', 'clearance (mm)'])

    s = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotSeriesNode', name)
    s.SetAndObserveTableNodeID(table.GetID())
    s.SetXColumnName('depth (mm)')
    s.SetYColumnName('clearance (mm)')
    s.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
    if markers:
        s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
        s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
        s.SetMarkerSize(11)
    else:
        s.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        s.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        s.SetLineWidth(3)
    s.SetColor(*color)
    chart.AddAndObservePlotSeriesNodeID(s.GetID())
    return s


def plotClearance(planner, results, name='Clearance'):
    """Build a clearance-vs-depth chart. Does not change the layout; setupViews does.

    results: a single result dict, or {side: result} as returned by planVertebra.
    """
    if isinstance(results, dict) and 'clearance' in results:
        results = {results['side']: results}
    if not results:
        return None

    chart = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotChartNode', name)
    chart.SetTitle(f'{name}  (needle {planner.needleDiameterMm:.2f} mm OD)')
    chart.SetXAxisTitle('depth from bone entry (mm)')
    chart.SetYAxisTitle('clearance to cortex (mm)')

    maxDepth = 0.0
    for side, r in results.items():
        x, y = r['profile']
        maxDepth = max(maxDepth, float(x.max()))
        col, _ = _statusColor(r)
        _plotSeries(chart, f'{name} {side}', x, y, col)
        _plotSeries(chart, f'{name} {side} isthmus',
                    [r['narrowestDepthMm']], [r['clearance']], col, markers=True)

    ref = np.array([0.0, maxDepth])
    _plotSeries(chart, f'{name} needle radius',
                ref, np.full(2, planner.needleRadius), RED)
    _plotSeries(chart, f'{name} required',
                ref, np.full(2, planner.requiredClearance), GREY)
    return chart


# =========================================================================== views
PLANNER_LAYOUT_ID = 701

_PLANNER_LAYOUT_XML = """
<layout type="horizontal" split="true">
  <item splitSize="600">
    <view class="vtkMRMLViewNode" singletontag="1">
      <property name="viewlabel" action="default">1</property>
    </view>
  </item>
  <item splitSize="400">
    <layout type="vertical" split="true">
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">R</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLPlotViewNode" singletontag="PlotView1">
          <property name="viewlabel" action="default">P</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""


def _registerLayout():
    lm = slicer.app.layoutManager()
    node = lm.layoutLogic().GetLayoutNode()
    if not node.IsLayoutDescription(PLANNER_LAYOUT_ID):
        node.AddLayoutDescription(PLANNER_LAYOUT_ID, _PLANNER_LAYOUT_XML)
    return lm


def restoreLayout():
    """Back to the standard four-up."""
    slicer.app.layoutManager().setLayout(
        slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)


def hideSlicePlanesIn3D():
    """Turn off slice-plane rendering in the 3D view for every slice.

    The reformatted plane contains the trajectory, so leaving it on occludes
    exactly the thing you are trying to inspect.
    """
    lm = slicer.app.layoutManager()
    for name in lm.sliceViewNames():
        w = lm.sliceWidget(name)
        if w is not None:
            w.mrmlSliceNode().SetSliceVisible(False)


def showVertebraSurface(planner, opacity=0.35, buildSurface=True):
    """Render only the planned vertebra, semi-transparent so the needle shows through.

    buildSurface=False skips closed-surface conversion, which is slow in a Debug
    build because it converts every segment, not just the visible one. The surface
    persists once built, so you only pay it the first time per case.
    """
    seg = planner.seg
    dn = seg.GetDisplayNode()
    if dn is None:
        seg.CreateDefaultDisplayNodes()
        dn = seg.GetDisplayNode()
    if buildSurface:
        seg.CreateClosedSurfaceRepresentation()

    s = seg.GetSegmentation()
    for i in range(s.GetNumberOfSegments()):
        sid = s.GetNthSegmentID(i)
        show = (sid == planner.segmentId)
        dn.SetSegmentVisibility(sid, show)
        if show:
            dn.SetSegmentOpacity3D(sid, opacity)
    dn.SetVisibility(True)
    return dn


def frameCamera(planner, viewFrom='superior', distanceMm=260.0):
    """Point the 3D camera at the vertebra.

    'superior' looks down the SI axis, which is the view where medial angulation is
    directly readable - the same projection you read an axial slice in.
    'posterior' looks along +AP, the fluoro AP view.
    """
    lm = slicer.app.layoutManager()
    if lm.threeDWidget(0) is None:
        return None
    viewNode = lm.threeDWidget(0).mrmlViewNode()
    cam = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
    if cam is None:
        return None

    focal = planner.bodyCenter
    if viewFrom == 'posterior':
        pos, up = focal - distanceMm * planner.ap, planner.si
    else:
        pos, up = focal + distanceMm * planner.si, planner.ap

    cam.SetFocalPoint(*focal)
    cam.SetPosition(*pos)
    cam.SetViewUp(*up)
    cam.ResetClippingRange()
    return cam


def setupViews(planner, results, chart=None, viewFrom='superior',
               buildSurface=True, sliceViewName='Red', showSlicePlane=False):
    """Three-panel layout: 3D + trajectory-reformatted slice + clearance chart."""
    lm = _registerLayout()
    lm.setLayout(PLANNER_LAYOUT_ID)
    slicer.app.processEvents()

    showVertebraSurface(planner, buildSurface=buildSurface)

    if results:
        first = list(results.values())[0] if isinstance(results, dict) else results
        planner.reformatToTrajectory(first, sliceViewName)

    if showSlicePlane:
        w = lm.sliceWidget(sliceViewName)
        if w is not None:
            w.mrmlSliceNode().SetSliceVisible(True)
    else:
        hideSlicePlanesIn3D()

    if chart is not None:
        pw = lm.plotWidget(0)
        if pw is not None:
            pw.mrmlPlotViewNode().SetPlotChartNodeID(chart.GetID())

    frameCamera(planner, viewFrom=viewFrom)


# =========================================================================== driver
def loadCase(ctPath, maskPath=None):
    """Load a CT (+ optional multi-label vertebra mask) and return (ct, seg).

    A multi-label NIfTI loaded via loadSegmentation becomes one segment per integer.
    Segment names are NOT preserved - they come out 'Segment_1', 'Segment_2', ... in
    file order, which is not necessarily anatomical order. Use listSegments() to sort
    them inferior -> superior and identify levels by size and spacing.
    """
    ct = slicer.util.loadVolume(ctPath)
    seg = None
    if maskPath is not None:
        seg = slicer.util.loadSegmentation(maskPath)
        seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
    print(f'CT  {ct.GetName()}  spacing={tuple(round(v, 3) for v in ct.GetSpacing())}'
          f'  dims={ct.GetImageData().GetDimensions()}')
    if seg is not None:
        print(f'SEG {seg.GetName()}  '
              f'{seg.GetSegmentation().GetNumberOfSegments()} segments')
    return ct, seg


def listSegments(ctNode, segNode):
    """Inventory every segment, sorted inferior -> superior. Returns a list of dicts.

    Level identification: lumbar vertebrae run ~35-50 cm3 with 33-38 mm centroid
    spacing; thoracic run ~20-30 cm3 with 16-22 mm spacing. Counting up from the
    sacrum is the only fully reliable method. A segment whose S extent touches the
    CT bound is truncated by the field of view - do not plan on it.
    """
    ijkToRas = _mat(ctNode, 'IJKToRAS')
    sx, sy, sz = ctNode.GetSpacing()
    voxCm3 = sx * sy * sz / 1000.0
    b = [0.0] * 6
    ctNode.GetRASBounds(b)

    s = segNode.GetSegmentation()
    rows = []
    for i in range(s.GetNumberOfSegments()):
        sid = s.GetNthSegmentID(i)
        m = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, sid, ctNode) > 0
        kji = np.argwhere(m)
        if not len(kji):
            continue
        h = np.hstack([kji[:, ::-1].astype(float), np.ones((len(kji), 1))])
        ras = (ijkToRas @ h.T).T[:, :3]
        rows.append(dict(segmentId=sid, name=s.GetSegment(sid).GetName(),
                         volumeCm3=len(kji) * voxCm3,
                         centroidS=float(ras[:, 2].mean()),
                         sMin=float(ras[:, 2].min()), sMax=float(ras[:, 2].max())))

    rows.sort(key=lambda r: r['centroidS'])
    print(f'CT S range: {b[4]:.1f} .. {b[5]:.1f}    (inferior -> superior)')
    prev = None
    for r in rows:
        gap = '' if prev is None else f'{r["centroidS"] - prev:5.1f} mm'
        trunc = ' TRUNCATED' if (r['sMin'] <= b[4] + 1 or r['sMax'] >= b[5] - 1) else ''
        print(f'  {r["segmentId"]:<12} {r["volumeCm3"]:5.1f} cm3  '
              f'S={r["centroidS"]:7.1f}  gap={gap:>9}{trunc}')
        prev = r['centroidS']
    return rows


def planVertebra(ctNode, segNode, segmentId, sides=('left', 'right'),
                 makeNodes=True, makeNeedle=True, markIsthmus=True,
                 makePlot=True, setupLayout=True, buildSurface=True,
                 viewFrom='superior', showSlicePlane=False,
                 colorBy='containment', **kwargs):
    """Plan one segment, report, build nodes, and arrange the three-panel view.

    Extra kwargs pass through to TranspedicularPlanner (needleDiameterMm,
    safetyMarginMm, cropPadMm, minPosteriorSpanMm, ...).
    Returns (planner, {side: result}).
    """
    p = TranspedicularPlanner(ctNode, segNode, segmentId, **kwargs)
    print(f'frame: symmetry={p.symmetryOverlap:.3f}  '
          f'bodyDepth={p.bodyDepthPosteriorMm:.1f} mm  '
          f'postWall={p.apPosteriorWall:.1f} mm')

    results = {}
    for side in sides:
        try:
            r = p.plan(side=side)
        except RuntimeError as e:
            print(f'{segmentId} / {side}: {e}')
            p.debugSweep(side=side)
            continue
        results[side] = r
        p.report(r, colorBy=colorBy)
        if makeNodes:
            p.createNodes(r, name=f'{segmentId}_{side}', colorBy=colorBy)
        if makeNeedle:
            p.createNeedleModel(r, name=f'{segmentId}_{side}_needle', colorBy=colorBy)
        if markIsthmus:
            p.markNarrowest(r, name=f'{segmentId}_{side}_isthmus')

    chart = plotClearance(p, results, name=f'{segmentId} clearance') \
        if (makePlot and results) else None
    if setupLayout and results:
        setupViews(p, results, chart, viewFrom=viewFrom, buildSurface=buildSurface,
                   showSlicePlane=showSlicePlane)
    return p, results


def planAll(ctNode, segNode, sides=('left', 'right'), **kwargs):
    """Sweep every segment and print a one-line summary per side. No nodes created."""
    s = segNode.GetSegmentation()
    out = []
    for i in range(s.GetNumberOfSegments()):
        sid = s.GetNthSegmentID(i)
        try:
            p = TranspedicularPlanner(ctNode, segNode, sid, verbose=False, **kwargs)
        except Exception as e:
            print(f'{sid:<12} planner failed: {e}')
            continue
        for side in sides:
            try:
                r = p.plan(side=side)
            except RuntimeError:
                print(f'{sid:<12} {side:<5} no corridor')
                continue
            out.append(r)
            _, status = _statusColor(r)
            print(f'{sid:<12} {side:<5} clearance {r["clearance"]:5.2f} mm  '
                  f'depth {r["corridorLengthMm"]:5.1f} mm  '
                  f'pedicle {r["posteriorSpanMm"]:5.1f} mm  '
                  f'alpha {r["axialAngleDeg"]:5.1f}  {status}')
    return out


def clearPlans():
    """Remove trajectories, fiducials, needle models, pose transforms, and charts."""
    n = 0
    for cls in ('vtkMRMLMarkupsLineNode', 'vtkMRMLMarkupsFiducialNode',
                'vtkMRMLPlotChartNode', 'vtkMRMLPlotSeriesNode'):
        for node in slicer.util.getNodesByClass(cls):
            slicer.mrmlScene.RemoveNode(node)
            n += 1
    for node in slicer.util.getNodesByClass('vtkMRMLTableNode'):
        if node.GetName().endswith(' table'):
            slicer.mrmlScene.RemoveNode(node)
            n += 1
    for node in slicer.util.getNodesByClass('vtkMRMLModelNode'):
        if node.GetName().endswith('_needle'):
            slicer.mrmlScene.RemoveNode(node)
            n += 1
    for node in slicer.util.getNodesByClass('vtkMRMLLinearTransformNode'):
        if node.GetName().endswith('_pose'):
            slicer.mrmlScene.RemoveNode(node)
            n += 1
    print(f'removed {n} nodes')
