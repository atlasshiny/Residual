import json, os
from pxr import Usd, UsdGeom, Gf, Vt, Sdf

d = json.load(open("viewer/viewer_data.json"))
n = len(d["t"])
os.makedirs("out", exist_ok=True)
stage = Usd.Stage.CreateNew("out/artemis2.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1000.0)       # scene units are km
stage.SetTimeCodesPerSecond(24)
stage.SetStartTimeCode(0); stage.SetEndTimeCode(n - 1)
UsdGeom.Xform.Define(stage, "/World")

def sphere(path, radius, color, track=None):
    s = UsdGeom.Sphere.Define(stage, path)
    s.GetRadiusAttr().Set(radius)
    s.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    op = UsdGeom.Xformable(s).AddTranslateOp()
    if track is None:
        return s
    for i, p in enumerate(track):
        op.Set(Gf.Vec3d(*p), i)
    return s

e = d["earth"]
sphere("/World/Earth", 6371.0, (0.17, 0.37, 0.72)).GetPrim()
UsdGeom.Xformable(stage.GetPrimAtPath("/World/Earth")).GetOrderedXformOps()[0].Set(Gf.Vec3d(*e))
sphere("/World/Moon", 1737.4, (0.6, 0.6, 0.58), d["moon"])
orion = sphere("/World/Orion", 250.0, (1.0, 0.7, 0.28), d["paths"]["truth"])   # enlarged marker
orion.GetPrim().CreateAttribute("note", Sdf.ValueTypeNames.String).Set("radius is a visibility marker, not true size")

# mission time per sample, so tools can map time code back to seconds
attr = stage.GetPrimAtPath("/World").CreateAttribute("missionSeconds", Sdf.ValueTypeNames.Double)
for i, t in enumerate(d["t"]):
    attr.Set(float(t), i)

colors = {"truth": (1, 1, 1), "mlp": (0.15, 0.85, 1), "pinn": (1, 0.6, 0.18), "sindy": (0.9, 0.36, 0.9)}
for k, pts in d["paths"].items():
    c = UsdGeom.BasisCurves.Define(stage, f"/World/Trails/{k}")
    c.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    c.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))
    c.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    c.GetWidthsAttr().Set(Vt.FloatArray([120.0]))
    c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    c.GetDisplayColorAttr().Set([Gf.Vec3f(*colors.get(k, (0.5, 1, 0.5)))])

stage.GetRootLayer().Save()
print("wrote out/artemis2.usda,", n, "time samples")