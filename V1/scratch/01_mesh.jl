using Gmsh, GridapGmsh

function build_room_mesh(path::String; h::Float64=0.05,
                         exit_y::Tuple{Float64,Float64}=(0.45, 0.55))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room")
    g = gmsh.model.geo
    p1 = g.addPoint(0.0, 0.0, 0.0, h)
    p2 = g.addPoint(1.0, 0.0, 0.0, h)
    p3 = g.addPoint(1.0, exit_y[1], 0.0, h)
    p4 = g.addPoint(1.0, exit_y[2], 0.0, h)
    p5 = g.addPoint(1.0, 1.0, 0.0, h)
    p6 = g.addPoint(0.0, 1.0, 0.0, h)

    l1 = g.addLine(p1, p2)
    l2 = g.addLine(p2, p3)
    l3 = g.addLine(p3, p4)   # exit
    l4 = g.addLine(p4, p5)
    l5 = g.addLine(p5, p6)
    l6 = g.addLine(p6, p1)

    cl = g.addCurveLoop([l1, l2, l3, l4, l5, l6])
    s  = g.addPlaneSurface([cl])
    g.synchronize()

    gmsh.model.addPhysicalGroup(1, [l1, l2, l4, l5, l6], -1, "wall")
    gmsh.model.addPhysicalGroup(1, [l3],                  -1, "exit")
    gmsh.model.addPhysicalGroup(2, [s],                   -1, "domain")

    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

path = joinpath(@__DIR__, "room.msh")
build_room_mesh(path; h=0.06)
model = GmshDiscreteModel(path)
println("Model loaded.")
labels = get_face_labeling(model)
println("Tags: ", labels.tag_to_name)
