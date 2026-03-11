######################################################################
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
######################################################################
# Define constants
Model_origin_x = 0.
Model_origin_y = 0.

Model_enddim_x = 100.0
Model_enddim_y = 06.0

GlobalElementSize = 1.0
######################################################################
######################################################################
# Calculate constants
Model_size_x = Model_enddim_x - Model_origin_x
Model_size_y = Model_enddim_y - Model_origin_y

modelcentre_x = (Model_origin_x + Model_size_x)/2
modelcentre_y = (Model_origin_y + Model_size_y)/2

# left edge
LE_centre_x = Model_origin_x
LE_centre_y = (Model_origin_y + Model_enddim_y)/2
# right edge
RE_centre_x = Model_enddim_x
RE_centre_y = LE_centre_y
# bottom edge
BE_centre_x = (Model_origin_x + Model_enddim_x)/2
BE_centre_y = Model_origin_y
# top edge
TE_centre_x = (Model_origin_x + Model_enddim_x)/2
TE_centre_y = Model_enddim_y

RE_Displacement = TotalStrain_x*Model_size_x
######################################################################
model  = mdb.models['Model-1']
######################################################################
model.ConstrainedSketch(name='__profile__', sheetSize=200.0)
sketch = model.sketches['__profile__']
######################################################################
sketch.rectangle(point1 = (Model_origin_x, Model_origin_y), point2 = (Model_enddim_x, Model_enddim_y))
model.Part(dimensionality=TWO_D_PLANAR, name='partname', type=DEFORMABLE_BODY)
model.parts['partname'].BaseShell(sketch=sketch)

#del sketch
######################################################################
model.ConstrainedSketch(gridSpacing=2.69, name='__profile__', 
    sheetSize=107.7, transform=
    model.parts['partname'].MakeSketchTransform(
    sketchPlane=model.parts['partname'].faces[0], 
    sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0)))
######################################################################
sketch = model.sketches['__profile__']
######################################################################
part = model.parts['partname']
######################################################################
part.projectReferencesOntoSketch(filter = COPLANAR_EDGES, sketch = sketch)
######################################################################
sketch = model.sketches['__profile__']
######################################################################
# Create base sets
part.Set(faces = part.faces.findAt(((modelcentre_x, modelcentre_y, 0.),)), name = 'FullFace')
part.Set(edges = part.edges.findAt(((LE_centre_x, LE_centre_y,0.),)), name = 'FullEdge_x-')
part.Set(edges = part.edges.findAt(((RE_centre_x, RE_centre_y,0.),)), name = 'FullEdge_x+')
part.Set(edges = part.edges.findAt(((BE_centre_x, BE_centre_y,0.),)), name = 'FullEdge_y-')
part.Set(edges = part.edges.findAt(((TE_centre_x, TE_centre_y,0.),)), name = 'FullEdge_y+')
part.Set(vertices = part.vertices.findAt(((Model_origin_x, Model_origin_y,0.),)), name = 'FullVertex_BL')
######################################################################
######################################################################
######################################################################
xdata_filename = '_x_data_tess_2dvor.txt'
ydata_filename = '_y_data_tess_2dvor.txt'
cdata_filename = '_c_data_tess_2dvor.txt'
######################################################################
######################################################################
######################################################################
import numpy as np

x = np.loadtxt(xdata_filename, delimiter=",")
y = np.loadtxt(ydata_filename, delimiter=",")
c = np.loadtxt(cdata_filename, delimiter=",")

x = np.round(x, 4)
y = np.round(y, 4)
######################################################################
# Create the partitions
sketch = model.sketches['__profile__']
part.projectReferencesOntoSketch(filter = COPLANAR_EDGES, sketch = sketch)

for row in range(0, x.shape[0]):
	tempx = x[row]
	tempy = y[row]
	for col in range(0, x.shape[1]):
		tempx = tempx[tempx!=999]
		tempy = tempy[tempy!=999]
	for ii in range(0, tempx.shape[0]-1):
		sketch.Line(point1=(tempx[ii], tempy[ii]), point2=(tempx[ii+1], tempy[ii+1]))
	sketch.Line(point1=(tempx[tempx.shape[0]-1], tempy[tempx.shape[0]-1]), point2=(tempx[0], tempy[0]))
	print tempx
	print tempy
#
part.PartitionFaceBySketch(faces = part.faces.findAt(((modelcentre_x, modelcentre_y, 0), ), ), sketch = sketch)
######################################################################
# create set names
for row in range(0, c.shape[0]):
	#part.DatumPointByCoordinate(coords = (c[row,0], c[row,1], 0.0))
	setname = 'Grain_Num_' + str(row)
	part.Set(faces = part.faces.findAt(((c[row,0], c[row,1], 0.),)), name = setname)
######################################################################
# Create materials
# Create sections
# Assign sections
for grn in range(0, c.shape[0]):
	# create materials
	matsetname = 'Mat_Grain_Num' + str(grn)
	model.Material(name = matsetname)
	model.materials[matsetname].UserMaterial(mechanicalConstants = (0.0, ))
	# create section
	secsetname = 'Sec_Grain_Num' + str(grn)
	model.HomogeneousSolidSection(material = matsetname, name = secsetname, thickness = None)
	# assign section
	grainsetname = 'Grain_Num_' + str(grn)
	part.SectionAssignment(offset = 0.0, offsetField = '', offsetType = MIDDLE_SURFACE, region = 
		part.sets[grainsetname], sectionName = secsetname, thicknessAssignment = FROM_SECTION)
######################################################################
# Create assembly
model.rootAssembly.DatumCsysByDefault(CARTESIAN)
model.rootAssembly.Instance(dependent=ON, name='partname', part=mdb.models['Model-1'].parts['partname'])
######################################################################
# Create time steps
model.StaticStep(initialInc=Time_Step1_initial_incr, maxInc=Time_Step1_maximum_incr, minInc=Time_Step1_minimum_incr, 
    name='Step-1', nlgeom=ON, previous='Initial', timePeriod=Time_Step1_total_time)
######################################################################
model = mdb.models['Model-1']
part  = model.parts['partname']
######################################################################
# Apply displacement boundary conditions
# for the constrained edge
model.DisplacementBC(amplitude = UNSET, createStepName = 'Initial', distributionType = UNIFORM, fieldName = '', localCsys = None, name = 
    'BC_constrained_x-', region = model.rootAssembly.instances['partname'].sets['FullEdge_x-'], u1 = SET, u2=SET, ur3=SET)
######################################################################
# Apply displacement boundary conditions
# for the pulled/pushed edge
model.DisplacementBC(amplitude = UNSET, createStepName = 'Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name = 
    'BC_displaced_x_+', region = model.rootAssembly.instances['partname'].sets['FullEdge_x+'] , u1 = RE_Displacement, u2 = UNSET, ur3 = UNSET)
######################################################################
# Set element type

# TRI---LINEAR----1st order accuracy
#for row in range(0, c.shape[0]):
#	part.setElementType(elemTypes=(ElemType(
#		elemCode=CPS4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
#		hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
#		elemCode=CPS3, elemLibrary=STANDARD, secondOrderAccuracy=ON, 
#		distortionControl=ON, lengthRatio=0.100000001490116)), regions=(
#		part.faces.findAt(((c[row,0], c[row,1], 0.),)), ))

# TRI---QUADRATIC----2nd order accuracy
for row in range(0, c.shape[0]):	
	part.setElementType(elemTypes=(ElemType(
		elemCode=CPS8R, elemLibrary=STANDARD), ElemType(elemCode=CPS6M, 
		elemLibrary=STANDARD, secondOrderAccuracy=ON, distortionControl=ON, 
		lengthRatio=0.100000001490116)), regions=(
		part.faces.findAt(((c[row,0], c[row,1], 0.),)), ))
######################################################################
# Assign mesh controls
#part.setMeshControls(allowMapped=False, elemShape=TRI, regions=part.faces.getSequenceFromMask(('[#200000 ]', ), ))
for row in range(0, c.shape[0]):
	part.setMeshControls(allowMapped=False, elemShape=TRI, regions=part.faces.findAt(((c[row,0], c[row,1], 0.),)))
######################################################################
# set element seed length - global
part.seedPart(deviationFactor = 0.1, minSizeFactor = 0.1, size = GlobalElementSize)
######################################################################
# Mesh the individual faces
for row in range(0, c.shape[0]):
	part.generateMesh(regions = part.faces.findAt(((c[row,0], c[row,1], 0.),)))
######################################################################
mdb.models['Model-1'].rootAssembly.regenerate()
instance = model.rootAssembly.instances['partname']
TotalNumOfNodes = len(instance.nodes)
TotalNumOfElem  = len(instance.elements)
usethis = 0

if usethis==1:
	if str(instance.elements[1].type)=='CPS4R' or str(instance.elements[1].type)=='CPS4':
		Total_DOF = TotalNumOfNodes*4
	elif str(instance.elements[1].type)=='CPS8R' or str(instance.elements[1].type)=='CPS8':
		Total_DOF = TotalNumOfNodes*8

print('%d Elements --- %d Nodes --- '% (TotalNumOfElem, TotalNumOfNodes))
######################################################################