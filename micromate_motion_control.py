""" 
Execute the following steps in the Slicer Python Console:

1. Load this motion control python script

2. Place the robot, solve for the generated trajectory, and create motion plan

3. Preview and execute plan (preview and execute robot joint movement following out = runMicromate...,
    and then needle movement following planSide...)

4. Retract needle and reset robot state

5. Replace 'left' with 'right' (or vice versa) in the function calls to repeat process for other side of vertebra


runMicromate and planSide parameters:
    'left' and 'right': side of vertebra to plan for and execute on
    stage:
        'all': single trajectory, with all joints moving at once (including needle)
        'move': X, Y, and tilt (no needle insertion)
        'insert': needle insertion only (following 'move')

planHome parameters:
    retract_only:
        True: only retracts needle, robot joints remain stationary
        no argument: robot joints also return to home


exec(open('~/ros2_ws/src/src/slicer_ros2_module/micromate_motion_control.py').read())

out = runMicromate(res, 'left', stage='move')

planSide(res, 'left', stage='insert')

planHome(retract_only=True)
planHome()
"""



# ROBOT PLACEMENT PARAMETERS
# --------------------------
OFFSET_MM = -32.0 # needle tip distance from skin entry point
ROLL_DEG = 0.0 # roll about the needle axis
PLACEMENT_ORIENTATION = 0.6 # between 0 and 1; 0 = robot is flat/perpendicular to the spine, 1 = needle aligned with trajectory
LATERAL_FRAC = 0.8 # between 0 and 1; 0 = needle is centred on the vertebra, 1 = robot is over the skin entry's side position
STOP_SHORT_MM = 0.0 # distance that needle will stop short of full trajectory target
# --------------------------



import numpy as np
from scipy import optimize
import slicer, vtk


# Micromate and needle forward kinematics
_K = 30.7692307692
_AX = np.array([0.0, 0.998152941013715, -0.0607511839034574])
_TIP = np.array([-0.00119933, 0.00784393, 0.12887725, 1.0])

def _T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    M = np.eye(4); M[:3, :3] = Rz @ Ry @ Rx; M[:3, 3] = xyz
    return M

def _rot(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    M = np.eye(4); M[:3, :3] = np.eye(3) + np.sin(q) * Kx + (1 - np.cos(q)) * Kx @ Kx
    return M

def _tr(axis, q):
    M = np.eye(4); M[:3, 3] = np.asarray(axis, float) * q
    return M

def micromateFK(q):
    # q = [top_base_x, top_base_y, top_arm_x, top_arm_y, needle_insert]
    # return: 4x4 pose of needle_needle_insert_link in the robot root frame
    bx, by, ax, ay, ins = q
    M = _T((0, 0, 0.036)) @ _tr((1, 0, 0), bx) @ _tr((0, 1, 0), by)
    M = M @ _T((0.14989, 0.020603, -0.0211))
    M = M @ _T((0, 0.02, 0.01)) @ _rot((1, 0, 0), -_K * ay)
    M = M @ _T((0.1287, 0.000686, 0.010686), (0, 0, 1.5708))
    M = M @ _T((0, 0.02675, -0.01275), (-1.5708, -0.14185, -1.5708)) @ _rot((0, 0, 1), _K * ax)
    M = M @ _T((0.0276220163006711, -0.00314931749452428, 0.129115729986363),
               (-3.08080403835634, 0, 0.141852856920061))
    return M @ _tr(_AX, ins)

def needleTipAndDir(q):
    F = micromateFK(q)
    a = -(F[:3, :3] @ _AX)
    return (F @ _TIP)[:3], a / np.linalg.norm(a)

def solveMicromate(r, rootToWorld, stopShortMm=0.0, maxStrokeMm=75.0, verbose=True):
    Minv = np.linalg.inv(np.asarray(rootToWorld, float))
    toRobot = lambda p: (Minv @ np.append(np.asarray(p, float), 1.0))[:3] / 1000.0
    tgt = toRobot(r['target'])
    d = Minv[:3, :3] @ np.asarray(r['direction'], float); d /= np.linalg.norm(d)
    tgt = tgt - d * stopShortMm / 1000.0
    lo = np.array([-0.019] * 4 + [-maxStrokeMm / 1000.0]); hi = np.array([0.019] * 4 + [0.0])

    def resid(q):
        tip, a = needleTipAndDir(q)
        return np.concatenate([(tip - tgt) * 1000.0, 50.0 * (a - d)])

    best = None
    for x0 in ([0, 0, 0, 0, -0.05], [0.01, 0.01, 0.01, 0.01, -0.05], [-0.01, -0.01, -0.01, -0.01, -0.05]):
        s_ = optimize.least_squares(resid, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
        if best is None or s_.cost < best.cost:
            best = s_
    q = best.x
    tip, a = needleTipAndDir(q)
    posErr = float(np.linalg.norm(tip - tgt) * 1000.0)
    angErr = float(np.degrees(np.arccos(np.clip(a @ d, -1, 1))))

    # Stroke needed, measured from the fully retracted tip along the needle line
    tip0, a0 = needleTipAndDir(list(q[:4]) + [0.0])
    needMm = float((tgt - tip0) @ a0 * 1000.0)
    skinMm = None
    if r.get('skinEntry') is not None:
        skinMm = float((toRobot(r['skinEntry']) - tip0) @ a0 * 1000.0)
    atBound = [n for n, v, l, h in zip(['top_base_x', 'top_base_y', 'top_arm_x', 'top_arm_y', 'needle_insert'], q, lo, hi)
               if min(v - l, h - v) < 1e-5]
    groupOrder = [float(q[0]), float(q[1]), float(q[4]), float(q[2]), float(q[3])]
    out = dict(q=q, groupOrder=groupOrder, posErrMm=posErr, angErrDeg=angErr, strokeNeededMm=needMm, retractedToSkinMm=skinMm,
               atBound=atBound, ok=(posErr < 0.5 and angErr < 0.5))
    if verbose:
        n = ['top_base_x', 'top_base_y', 'top_arm_x', 'top_arm_y', 'needle_insert']
        print("joint solution (m):", {k: round(float(v), 5) for k, v in zip(n, q)})
        print(f"tip error {posErr:.2f} mm | direction error {angErr:.2f} deg | reachable: {out['ok']}")
        print(f"stroke needed: {needMm:.1f} mm ({maxStrokeMm:.0f} usable)")
        # if skinMm is not None:
        #     print(f"retracted tip is {skinMm:.1f} mm from the skin entry "
        #           f"({'outside the skin' if skinMm >= 0 else 'INSIDE the skin: move the robot back'})")
        if atBound:
            print("at a joint limit:", atBound)
        # print("xy_needle group order [bx, by, insert, arm_x, arm_y]:", [round(v, 5) for v in groupOrder])
        # if out['ok']:
        #     print("\n# --- commands (run in a terminal, XY first, then needle) ---")
        #     print("ros2 topic pub --once /xy_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
        #           f"\"{{joint_names: [top_base_x, top_base_y, top_arm_x, top_arm_y], points: [{{positions: "
        #           f"[{q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}], time_from_start: {{sec: 5}}}}]}}\"")
        #     print("ros2 topic pub --once /needle_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
        #           f"\"{{joint_names: [needle_needle_insert], points: [{{positions: [{q[4]:.5f}], time_from_start: {{sec: 8}}}}]}}\"")
        # else:
        #     print("\nNot reachable as planned, so no commands printed. See the notes on stroke and placement.")
    return out


def planFromSolver(out, stage='all'):
    # import slicer
    if not out.get('ok'):
        print("Solver result is not reachable; not planning."); return None
    motionLogic = slicer.util.getModuleLogic("ROS2MotionControl")
    w = slicer.util.getModuleWidget("ROS2MotionControl")
    goal = list(out['groupOrder'])
    if stage == 'move':
        goal[2] = 0.0
    elif stage not in ('all', 'insert'):
        print("stage must be 'all', 'move' or 'insert'"); return None
    motionLogic.last_ik_solution = goal
    w.onPlanButton()
    td = getattr(w, 'trajectoryData', None)
    if td is None or not td.GetJointTrajectory().GetPoints():
        print("No trajectory was loaded. Check the Generator is MoveIt and read the move_group terminal."); return None
    final = [round(float(v), 5) for v in td.GetJointTrajectory().GetPoints()[-1].GetPositions()]
    print("trajectory joints:", list(td.GetJointTrajectory().GetJointNames()))
    print("planned final    :", final)
    print("solver wants     :", [round(v, 5) for v in goal])
    if stage in ('move', 'insert'):
        names = list(td.GetJointTrajectory().GetJointNames())
        P = np.array([list(pt.GetPositions()) for pt in td.GetJointTrajectory().GetPoints()])
        ii = names.index('needle_needle_insert')
        others = [k for k in range(len(names)) if k != ii]
        if stage == 'move':
            dev = float(np.abs(P[:, ii]).max()); what = "insertion"
        else:
            dev = float(np.abs(P[:, others] - P[0, others]).max()); what = "XY/tilt joints"
        print(f"stage '{stage}': max {what} deviation along the path = {dev*1000:.3f} mm")
        if dev > 5e-4:
            print("Path is not cleanly separated; do not execute, see the numbers above.")
    return td


# def fixTrajectoryPreview():
#     """Run inside Slicer, once, before planning. Display-only fix for the Motion Control scrubber/Preview.

#     The planned trajectory is in MoveIt group order (top_base_x, top_base_y, needle_needle_insert,
#     top_arm_x, top_arm_y). Slicer's robot chain is (top_base_x, top_base_y, needle_mount,
#     needle_bottom_revolute, needle_needle_insert), where the two tilt joints mimic the arm joints.
#     While a trajectory owns the goal robot, this remaps by joint name before the FK update.
#     It changes nothing that is planned or executed. Undo with unfixTrajectoryPreview()."""
#     import slicer
#     w = slicer.util.getModuleWidget("ROS2MotionControl")
#     logic = w.logic
#     if '_origGoalFK' in logic.__dict__:
#         print("Preview fix already active."); return
#     logic.__dict__['_origGoalFK'] = logic.updategoalTransformsFromJointsKDL   # bound method, original
#     K = 30.7692307692                                                         # mimic multiplier (URDF)

#     def patched(robotmodel, joint_values):
#         vals = list(joint_values)
#         try:
#             td = getattr(w, 'trajectoryData', None)
#             if getattr(w, '_goalVisualOwner', None) == "trajectory" and td is not None and robotmodel:
#                 tn = list(td.GetJointTrajectory().GetJointNames())
#                 if len(vals) == len(tn) and 'top_arm_x' in tn and 'top_arm_y' in tn:
#                     m = dict(zip(tn, vals))
#                     out = []
#                     for name in list(robotmodel.GetJoints()):
#                         if name == 'needle_mount':
#                             out.append(-K * m['top_arm_y'])
#                         elif name == 'needle_bottom_revolute':
#                             out.append(K * m['top_arm_x'])
#                         else:
#                             out.append(m[name])
#                     vals = out
#         except Exception as e:
#             print("preview remap skipped:", e)
#         return logic.__dict__['_origGoalFK'](robotmodel, vals)

#     logic.updategoalTransformsFromJointsKDL = patched
#     print("Preview fix active (display only).")


# def unfixTrajectoryPreview():
#     import slicer
#     logic = slicer.util.getModuleWidget("ROS2MotionControl").logic
#     logic.__dict__.pop('updategoalTransformsFromJointsKDL', None)
#     logic.__dict__.pop('_origGoalFK', None)
#     print("Preview fix removed.")



_ctx = {}


def _segPointsRAS(segId, step=3):
    arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg, segId, ct)
    idx = np.argwhere(arr > 0)[::step]
    m = vtk.vtkMatrix4x4(); ct.GetIJKToRASMatrix(m)
    ijk1 = np.c_[idx[:, ::-1], np.ones(len(idx))]
    return (slicer.util.arrayFromVTKMatrix(m) @ ijk1.T).T[:, :3]


def _spineFrame():
    ids = list(seg.GetSegmentation().GetSegmentIDs())
    cents = np.array([_segPointsRAS(i, step=8).mean(axis=0) for i in ids])
    S = np.linalg.svd(cents - cents.mean(axis=0))[2][0]
    if S @ np.array([0, 0, 1.0]) < 0: S = -S
    A = np.array([0, 1.0, 0]); A = A - (A @ S) * S; A /= np.linalg.norm(A)
    return S, A, np.cross(S, A)


def _axisAndAnchor(dt, skin, segId, blend, frac):
    if blend >= 1.0 and frac >= 1.0:
        return dt, skin
    S, A, L = _spineFrame()
    cen = _segPointsRAS(segId).mean(axis=0)
    d = (1.0 - blend) * A + blend * dt; d /= np.linalg.norm(d)
    lat = (skin - cen) @ L
    return d, skin - (1.0 - frac) * lat * L


def setupMicromate(res, side='left', offset_mm=None, roll_deg=None, blend=None, frac=None):
    offset_mm = OFFSET_MM if offset_mm is None else offset_mm
    roll_deg = ROLL_DEG if roll_deg is None else roll_deg
    blend = PLACEMENT_ORIENTATION if blend is None else blend
    frac = LATERAL_FRAC if frac is None else frac
    motionLogic = slicer.util.getModuleLogic("ROS2MotionControl")
    paramNode = motionLogic.getParameterNode()
    robotNode = slicer.mrmlScene.GetNodeByID(paramNode.robotNodeID)
    motionControlNode = slicer.mrmlScene.GetNodeByID(paramNode.motionControlNodeID)
    if robotNode is None or motionControlNode is None:
        raise RuntimeError("Add New Robot, then press Use in Motion Control, then rerun.")
    rootLink, tipLink = robotNode.FindRootAndTipLinks()[:2]
    rootTransform = motionLogic.findRobotTransforms(rootLink, goal=True)
    r = res[side]
    dt = np.array(r['direction'], float); dt /= np.linalg.norm(dt)
    skin = np.array(r['skinEntry'] if r['skinEntry'] is not None else r['entry'], float)
    d, skin = _axisAndAnchor(dt, skin, r['segmentId'], blend, frac)
    jointNames = list(robotNode.GetJoints())
    fk = vtk.vtkMatrix4x4()
    robotNode.ComputeKDLFK([0.0] * len(jointNames), fk, "needle_needle_insert_link")
    F = slicer.util.arrayFromVTKMatrix(fk)
    if np.linalg.norm(F[:3, 3]) < 5.0:
        F[:3, 3] *= 1000.0
    tip_link = np.append(np.array([-0.00119933, 0.00784393, 0.12887725]) * 1000.0, 1.0)
    P0 = (F @ tip_link)[:3]
    a = -(F[:3, :3] @ np.array([0.0, 0.998152941, -0.060751184])); a /= np.linalg.norm(a)
    v = np.cross(a, d); c = float(a @ d)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))
    tf = vtk.vtkTransform(); tf.RotateWXYZ(roll_deg, *d)
    R = slicer.util.arrayFromVTKMatrix(tf.GetMatrix())[:3, :3] @ R
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = (skin - offset_mm * d) - R @ P0
    tn = slicer.mrmlScene.GetFirstNodeByName('RobotToCT')
    if tn is None:
        tn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode', 'RobotToCT')
    tn.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(M))
    rootTransform.SetAndObserveTransformNodeID(tn.GetID())
    motionLogic.findRobotTransforms(rootLink, goal=False).SetAndObserveTransformNodeID(tn.GetID())
    _ctx['M'] = M
    # fixTrajectoryPreview()
    print(f"Robot placed for '{side}' (offset {offset_mm} mm, roll {roll_deg} deg, orientation {blend}, lateral frac {frac}).")
    return M


def planSide(res, side='left', stop_short_mm=None, stage='all'):
    stop_short_mm = STOP_SHORT_MM if stop_short_mm is None else stop_short_mm
    out = solveMicromate(res[side], _ctx['M'], stopShortMm=stop_short_mm)
    _ctx['out'] = out
    planFromSolver(out, stage)
    return out


def planHome(retract_only=False):
    goal = [0.0] * 5
    if retract_only:
        goal = list(_ctx['out']['groupOrder']); goal[2] = 0.0
    return planFromSolver({'ok': True, 'groupOrder': goal})


def runMicromate(res, side='left', **kw):
    setupMicromate(res, side, kw.get('offset_mm'), kw.get('roll_deg'), kw.get('blend'), kw.get('frac'))
    return planSide(res, side, kw.get('stop_short_mm'), kw.get('stage', 'all'))
