??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b5??

o

input_dataPlaceholder*
dtype0*
shape:??*(
_output_shapes
:??
j
yolov3_tiny/ShapeConst*
dtype0*%
valueB"   ?   ?      *
_output_shapes
:
i
yolov3_tiny/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!yolov3_tiny/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!yolov3_tiny/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
yolov3_tiny/strided_sliceStridedSliceyolov3_tiny/Shapeyolov3_tiny/strided_slice/stack!yolov3_tiny/strided_slice/stack_1!yolov3_tiny/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Fyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/minConst*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
valueB
 *???*
dtype0*
_output_shapes
: 
?
Fyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/maxConst*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
valueB
 *??>*
dtype0*
_output_shapes
: 
?
Pyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformHyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/shape*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
dtype0*
seed2 *

seed *&
_output_shapes
:
?
Fyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/subSubFyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/maxFyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
_output_shapes
: 
?
Fyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/mulMulPyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/RandomUniformFyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*&
_output_shapes
:
?
Byolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniformAddFyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/mulFyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*&
_output_shapes
:
?
'yolov3_tiny/darknet19_body/Conv/weights
VariableV2*
dtype0*
	container *
shape:*
shared_name *:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*&
_output_shapes
:
?
.yolov3_tiny/darknet19_body/Conv/weights/AssignAssign'yolov3_tiny/darknet19_body/Conv/weightsByolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
validate_shape(*
use_locking(*&
_output_shapes
:
?
,yolov3_tiny/darknet19_body/Conv/weights/readIdentity'yolov3_tiny/darknet19_body/Conv/weights*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*&
_output_shapes
:
~
-yolov3_tiny/darknet19_body/Conv/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
&yolov3_tiny/darknet19_body/Conv/Conv2DConv2D
input_data,yolov3_tiny/darknet19_body/Conv/weights/read*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*(
_output_shapes
:??
?
@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Initializer/onesConst*
dtype0*B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
valueB*  ??*
_output_shapes
:
?
/yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma
VariableV2*
dtype0*
	container *
shape:*
shared_name *B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
_output_shapes
:
?
6yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/AssignAssign/yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Initializer/ones*
validate_shape(*
use_locking(*
T0*B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
_output_shapes
:
?
4yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/readIdentity/yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
T0*B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
_output_shapes
:
?
@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Initializer/zerosConst*A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
?
.yolov3_tiny/darknet19_body/Conv/BatchNorm/beta
VariableV2*
dtype0*
	container *
shape:*
shared_name *A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
_output_shapes
:
?
5yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/AssignAssign.yolov3_tiny/darknet19_body/Conv/BatchNorm/beta@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
validate_shape(*
_output_shapes
:
?
3yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/readIdentity.yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
T0*A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
_output_shapes
:
?
Gyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
?
5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean
VariableV2*
shared_name *H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
dtype0*
	container *
shape:*
_output_shapes
:
?
<yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/AssignAssign5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanGyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:
?
:yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/readIdentity5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
T0*H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
_output_shapes
:
?
Jyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Initializer/onesConst*L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
valueB*  ??*
dtype0*
_output_shapes
:
?
9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes
:
?
@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/AssignAssign9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceJyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Initializer/ones*
T0*L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
:
?
>yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/readIdentity9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
T0*L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
_output_shapes
:
?
8yolov3_tiny/darknet19_body/Conv/BatchNorm/FusedBatchNormFusedBatchNorm&yolov3_tiny/darknet19_body/Conv/Conv2D4yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/read3yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/read:yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/read>yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*
is_training( *@
_output_shapes.
,:??::::
t
/yolov3_tiny/darknet19_body/Conv/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
)yolov3_tiny/darknet19_body/Conv/LeakyRelu	LeakyRelu8yolov3_tiny/darknet19_body/Conv/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*(
_output_shapes
:??
?
,yolov3_tiny/darknet19_body/MaxPool2D/MaxPoolMaxPool)yolov3_tiny/darknet19_body/Conv/LeakyRelu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*
T0*&
_output_shapes
:pp
?
Jyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*%
valueB"             *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/minConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*
valueB
 *?[??*
dtype0*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/maxConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*
valueB
 *?[?=*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*
dtype0*
seed2 *

seed *&
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
Dyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
)yolov3_tiny/darknet19_body/Conv_1/weights
VariableV2*
dtype0*
	container *
shape: *
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
0yolov3_tiny/darknet19_body/Conv_1/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_1/weightsDyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
.yolov3_tiny/darknet19_body/Conv_1/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_1/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
/yolov3_tiny/darknet19_body/Conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_1/Conv2DConv2D,yolov3_tiny/darknet19_body/MaxPool2D/MaxPool.yolov3_tiny/darknet19_body/Conv_1/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*&
_output_shapes
:pp 
?
Byolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Initializer/onesConst*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
valueB *  ??*
dtype0*
_output_shapes
: 
?
1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma
VariableV2*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
?
8yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
validate_shape(*
_output_shapes
: 
?
6yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
_output_shapes
: 
?
Byolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Initializer/zerosConst*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
?
0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta
VariableV2*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
?
7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
?
5yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
_output_shapes
: 
?
Iyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Initializer/zerosConst*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
?
7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean
VariableV2*
shape: *
shared_name *J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes
: 
?
>yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Initializer/zeros*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes
: 
?
<yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
_output_shapes
: 
?
Lyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/Initializer/onesConst*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
valueB *  ??*
dtype0*
_output_shapes
: 
?
;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance
VariableV2*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
?
Byolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: 
?
@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
_output_shapes
: 
?
:yolov3_tiny/darknet19_body/Conv_1/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_1/Conv2D6yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*
is_training( *>
_output_shapes,
*:pp : : : : 
v
1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_1/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_1/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*&
_output_shapes
:pp 
?
.yolov3_tiny/darknet19_body/MaxPool2D_1/MaxPoolMaxPool+yolov3_tiny/darknet19_body/Conv_1/LeakyRelu*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*&
_output_shapes
:88 
?
Jyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*%
valueB"          @   *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/minConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*
valueB
 *????*
dtype0*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/maxConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
Hyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
Dyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
)yolov3_tiny/darknet19_body/Conv_2/weights
VariableV2*
dtype0*
	container *
shape: @*
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
0yolov3_tiny/darknet19_body/Conv_2/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_2/weightsDyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*
validate_shape(*
use_locking(*&
_output_shapes
: @
?
.yolov3_tiny/darknet19_body/Conv_2/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_2/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
/yolov3_tiny/darknet19_body/Conv_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_2/Conv2DConv2D.yolov3_tiny/darknet19_body/MaxPool2D_1/MaxPool.yolov3_tiny/darknet19_body/Conv_2/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *&
_output_shapes
:88@
?
Byolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Initializer/onesConst*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma
VariableV2*
dtype0*
	container *
shape:@*
shared_name *D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
_output_shapes
:@
?
8yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
_output_shapes
:@
?
6yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
_output_shapes
:@
?
Byolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Initializer/zerosConst*
dtype0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
valueB@*    *
_output_shapes
:@
?
0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta
VariableV2*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
dtype0*
	container *
shape:@*
shared_name *
_output_shapes
:@
?
7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
5yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
_output_shapes
:@
?
Iyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Initializer/zerosConst*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
?
7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean
VariableV2*
shape:@*
shared_name *J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes
:@
?
>yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Initializer/zeros*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes
:@
?
<yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
_output_shapes
:@
?
Lyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/Initializer/onesConst*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance
VariableV2*
dtype0*
	container *
shape:@*
shared_name *N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
_output_shapes
:@
?
Byolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/Initializer/ones*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
:@
?
@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
_output_shapes
:@
?
:yolov3_tiny/darknet19_body/Conv_2/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_2/Conv2D6yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%??'7*>
_output_shapes,
*:88@:@:@:@:@
v
1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/ConstConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_2/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_2/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*&
_output_shapes
:88@
?
.yolov3_tiny/darknet19_body/MaxPool2D_2/MaxPoolMaxPool+yolov3_tiny/darknet19_body/Conv_2/LeakyRelu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*&
_output_shapes
:@
?
Jyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*%
valueB"      @   ?   *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/minConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
valueB
 *?[q?*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/maxConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
valueB
 *?[q=*
dtype0*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*'
_output_shapes
:@?
?
Hyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*'
_output_shapes
:@?
?
Dyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*'
_output_shapes
:@?
?
)yolov3_tiny/darknet19_body/Conv_3/weights
VariableV2*
shape:@?*
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
dtype0*
	container *'
_output_shapes
:@?
?
0yolov3_tiny/darknet19_body/Conv_3/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_3/weightsDyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
validate_shape(*'
_output_shapes
:@?
?
.yolov3_tiny/darknet19_body/Conv_3/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_3/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*'
_output_shapes
:@?
?
/yolov3_tiny/darknet19_body/Conv_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_3/Conv2DConv2D.yolov3_tiny/darknet19_body/MaxPool2D_2/MaxPool.yolov3_tiny/darknet19_body/Conv_3/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
	dilations
*
T0*'
_output_shapes
:?
?
Byolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Initializer/onesConst*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma
VariableV2*
shape:?*
shared_name *D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:?
?
8yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
6yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Initializer/zerosConst*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta
VariableV2*
dtype0*
	container *
shape:?*
shared_name *C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Initializer/zeros*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
5yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
_output_shapes	
:?
?
Iyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Initializer/zerosConst*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean
VariableV2*
shape:?*
shared_name *J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:?
?
>yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Initializer/zeros*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
<yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
_output_shapes	
:?
?
Lyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance
VariableV2*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
_output_shapes	
:?
?
@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
_output_shapes	
:?
?
:yolov3_tiny/darknet19_body/Conv_3/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_3/Conv2D6yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%??'7*
T0*C
_output_shapes1
/:?:?:?:?:?
v
1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_3/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_3/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
.yolov3_tiny/darknet19_body/MaxPool2D_3/MaxPoolMaxPool+yolov3_tiny/darknet19_body/Conv_3/LeakyRelu*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*'
_output_shapes
:?
?
Jyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*%
valueB"      ?      *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/minConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*
valueB
 *??*?*
dtype0*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/maxConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*
valueB
 *??*=*
dtype0*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*
dtype0*
seed2 *

seed *(
_output_shapes
:??
?
Hyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
Dyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
)yolov3_tiny/darknet19_body/Conv_4/weights
VariableV2*
dtype0*
	container *
shape:??*
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
0yolov3_tiny/darknet19_body/Conv_4/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_4/weightsDyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
.yolov3_tiny/darknet19_body/Conv_4/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_4/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
/yolov3_tiny/darknet19_body/Conv_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_4/Conv2DConv2D.yolov3_tiny/darknet19_body/MaxPool2D_3/MaxPool.yolov3_tiny/darknet19_body/Conv_4/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *'
_output_shapes
:?
?
Byolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Initializer/onesConst*
dtype0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
valueB?*  ??*
_output_shapes	
:?
?
1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma
VariableV2*
dtype0*
	container *
shape:?*
shared_name *D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
_output_shapes	
:?
?
8yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Initializer/ones*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
6yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Initializer/zerosConst*
dtype0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
valueB?*    *
_output_shapes	
:?
?
0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta
VariableV2*
shape:?*
shared_name *C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
5yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
_output_shapes	
:?
?
Iyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
valueB?*    *
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean
VariableV2*
shared_name *J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
>yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
_output_shapes	
:?
?
<yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
_output_shapes	
:?
?
Lyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/Initializer/onesConst*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance
VariableV2*
shape:?*
shared_name *N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
_output_shapes	
:?
?
:yolov3_tiny/darknet19_body/Conv_4/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_4/Conv2D6yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*
is_training( *C
_output_shapes1
/:?:?:?:?:?
v
1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/ConstConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_4/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_4/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
.yolov3_tiny/darknet19_body/MaxPool2D_4/MaxPoolMaxPool+yolov3_tiny/darknet19_body/Conv_4/LeakyRelu*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*'
_output_shapes
:?
?
Jyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/minConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
valueB
 *?[??*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/maxConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
valueB
 *?[?<*
dtype0*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/shape*

seed *
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
dtype0*
seed2 *(
_output_shapes
:??
?
Hyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*(
_output_shapes
:??
?
Dyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*(
_output_shapes
:??
?
)yolov3_tiny/darknet19_body/Conv_5/weights
VariableV2*
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
dtype0*
	container *
shape:??*(
_output_shapes
:??
?
0yolov3_tiny/darknet19_body/Conv_5/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_5/weightsDyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*(
_output_shapes
:??
?
.yolov3_tiny/darknet19_body/Conv_5/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_5/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*(
_output_shapes
:??
?
/yolov3_tiny/darknet19_body/Conv_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_5/Conv2DConv2D.yolov3_tiny/darknet19_body/MaxPool2D_4/MaxPool.yolov3_tiny/darknet19_body/Conv_5/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*'
_output_shapes
:?
?
Byolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Initializer/onesConst*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma
VariableV2*
dtype0*
	container *
shape:?*
shared_name *D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
_output_shapes	
:?
?
8yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
6yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Initializer/zerosConst*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta
VariableV2*
shared_name *C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
_output_shapes	
:?
?
5yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
_output_shapes	
:?
?
Iyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Initializer/zerosConst*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean
VariableV2*
shape:?*
shared_name *J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:?
?
>yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
<yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
_output_shapes	
:?
?
Lyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance
VariableV2*
dtype0*
	container *
shape:?*
shared_name *N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
_output_shapes	
:?
?
:yolov3_tiny/darknet19_body/Conv_5/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_5/Conv2D6yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%??'7*
T0*C
_output_shapes1
/:?:?:?:?:?
v
1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_5/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_5/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
.yolov3_tiny/darknet19_body/MaxPool2D_5/MaxPoolMaxPool+yolov3_tiny/darknet19_body/Conv_5/LeakyRelu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*'
_output_shapes
:?
?
Jyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/minConst*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
valueB
 *????*
dtype0*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/maxConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
valueB
 *???<*
_output_shapes
: 
?
Ryolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
dtype0*
seed2 *

seed *(
_output_shapes
:??
?
Hyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/subSubHyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/maxHyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
_output_shapes
: 
?
Hyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/mulMulRyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*(
_output_shapes
:??
?
Dyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniformAddHyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/mulHyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*(
_output_shapes
:??
?
)yolov3_tiny/darknet19_body/Conv_6/weights
VariableV2*
shape:??*
shared_name *<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
dtype0*
	container *(
_output_shapes
:??
?
0yolov3_tiny/darknet19_body/Conv_6/weights/AssignAssign)yolov3_tiny/darknet19_body/Conv_6/weightsDyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*(
_output_shapes
:??
?
.yolov3_tiny/darknet19_body/Conv_6/weights/readIdentity)yolov3_tiny/darknet19_body/Conv_6/weights*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*(
_output_shapes
:??
?
/yolov3_tiny/darknet19_body/Conv_6/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/darknet19_body/Conv_6/Conv2DConv2D.yolov3_tiny/darknet19_body/MaxPool2D_5/MaxPool.yolov3_tiny/darknet19_body/Conv_6/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:?
?
Ryolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
valueB:?*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
valueB
 *  ??*
_output_shapes
: 
?
Byolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/onesFillRyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones/shape_as_tensorHyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones/Const*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*

index_type0*
_output_shapes	
:?
?
1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma
VariableV2*
dtype0*
	container *
shape:?*
shared_name *D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
_output_shapes	
:?
?
8yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/AssignAssign1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammaByolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
6yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/readIdentity1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
_output_shapes	
:?
?
Ryolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Hyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros/ConstConst*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Byolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zerosFillRyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros/shape_as_tensorHyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros/Const*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta
VariableV2*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/AssignAssign0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betaByolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
5yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/readIdentity0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
_output_shapes	
:?
?
Yyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
valueB:?*
_output_shapes
:
?
Oyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
valueB
 *    *
_output_shapes
: 
?
Iyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zerosFillYyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorOyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean
VariableV2*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
>yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanIyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
<yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/readIdentity7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
_output_shapes	
:?
?
\yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
Ryolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones/ConstConst*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Lyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/onesFill\yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorRyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones/Const*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance
VariableV2*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
Byolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceLyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/readIdentity;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
_output_shapes	
:?
?
:yolov3_tiny/darknet19_body/Conv_6/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/darknet19_body/Conv_6/Conv2D6yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/read5yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/read<yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/read@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%??'7*C
_output_shapes1
/:?:?:?:?:?
v
1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
+yolov3_tiny/darknet19_body/Conv_6/LeakyRelu	LeakyRelu:yolov3_tiny/darknet19_body/Conv_6/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
Jyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/shapeConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*%
valueB"            *
_output_shapes
:
?
Hyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/minConst*
dtype0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
valueB
 *?7??*
_output_shapes
: 
?
Hyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/maxConst*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
valueB
 *?7?=*
dtype0*
_output_shapes
: 
?
Ryolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformJyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
dtype0*
seed2 *

seed *(
_output_shapes
:??
?
Hyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/subSubHyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/maxHyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
_output_shapes
: 
?
Hyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/mulMulRyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/RandomUniformHyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*(
_output_shapes
:??
?
Dyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniformAddHyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/mulHyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*(
_output_shapes
:??
?
)yolov3_tiny/yolov3_tiny_head/Conv/weights
VariableV2*
dtype0*
	container *
shape:??*
shared_name *<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*(
_output_shapes
:??
?
0yolov3_tiny/yolov3_tiny_head/Conv/weights/AssignAssign)yolov3_tiny/yolov3_tiny_head/Conv/weightsDyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
validate_shape(*
use_locking(*(
_output_shapes
:??
?
.yolov3_tiny/yolov3_tiny_head/Conv/weights/readIdentity)yolov3_tiny/yolov3_tiny_head/Conv/weights*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*(
_output_shapes
:??
?
/yolov3_tiny/yolov3_tiny_head/Conv/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(yolov3_tiny/yolov3_tiny_head/Conv/Conv2DConv2D+yolov3_tiny/darknet19_body/Conv_6/LeakyRelu.yolov3_tiny/yolov3_tiny_head/Conv/weights/read*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*'
_output_shapes
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Initializer/onesConst*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma
VariableV2*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
8yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/AssignAssign1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammaByolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Initializer/ones*
T0*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
6yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/readIdentity1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
T0*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
_output_shapes	
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Initializer/zerosConst*
dtype0*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
valueB?*    *
_output_shapes	
:?
?
0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta
VariableV2*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/AssignAssign0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betaByolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
_output_shapes	
:?
?
5yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/readIdentity0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
T0*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
_output_shapes	
:?
?
Iyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Initializer/zerosConst*J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean
VariableV2*
shared_name *J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
>yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/AssignAssign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanIyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Initializer/zeros*
T0*J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/readIdentity7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
T0*J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
_output_shapes	
:?
?
Lyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/Initializer/onesConst*N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance
VariableV2*
shape:?*
shared_name *N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/AssignAssign;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceLyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/readIdentity;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
T0*N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
_output_shapes	
:?
?
:yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/FusedBatchNormFusedBatchNorm(yolov3_tiny/yolov3_tiny_head/Conv/Conv2D6yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/read5yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/read<yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/read@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%??'7*C
_output_shapes1
/:?:?:?:?:?
v
1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/ConstConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
?
+yolov3_tiny/yolov3_tiny_head/Conv/LeakyRelu	LeakyRelu:yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
Lyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Jyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/minConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
valueB
 *?[??*
dtype0*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
valueB
 *?[?<*
dtype0*
_output_shapes
: 
?
Tyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformLyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/shape*

seed *
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
dtype0*
seed2 *(
_output_shapes
:??
?
Jyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/subSubJyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/maxJyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/mulMulTyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/RandomUniformJyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
Fyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniformAddJyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/mulJyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
+yolov3_tiny/yolov3_tiny_head/Conv_1/weights
VariableV2*
dtype0*
	container *
shape:??*
shared_name *>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
2yolov3_tiny/yolov3_tiny_head/Conv_1/weights/AssignAssign+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsFyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
0yolov3_tiny/yolov3_tiny_head/Conv_1/weights/readIdentity+yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
1yolov3_tiny/yolov3_tiny_head/Conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_1/Conv2DConv2D+yolov3_tiny/yolov3_tiny_head/Conv/LeakyRelu0yolov3_tiny/yolov3_tiny_head/Conv_1/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
	dilations
*
T0*'
_output_shapes
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Initializer/onesConst*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma
VariableV2*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?
?
:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/AssignAssign3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammaDyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
8yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/readIdentity3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Initializer/zerosConst*
dtype0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
valueB?*    *
_output_shapes	
:?
?
2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta
VariableV2*
shared_name *E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/AssignAssign2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betaDyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
_output_shapes	
:?
?
7yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/readIdentity2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
_output_shapes	
:?
?
Kyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Initializer/zerosConst*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean
VariableV2*
shared_name *L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/AssignAssign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanKyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
>yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/readIdentity9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Nyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/Initializer/onesConst*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance
VariableV2*
shared_name *P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/AssignAssign=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceNyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/readIdentity=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
_output_shapes	
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/FusedBatchNormFusedBatchNorm*yolov3_tiny/yolov3_tiny_head/Conv_1/Conv2D8yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/read7yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/read>yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/readByolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*
is_training( *C
_output_shapes1
/:?:?:?:?:?
x
3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/ConstConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
?
-yolov3_tiny/yolov3_tiny_head/Conv_1/LeakyRelu	LeakyRelu<yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
Lyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/shapeConst*
dtype0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*%
valueB"            *
_output_shapes
:
?
Jyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/minConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
valueB
 *??ٽ*
dtype0*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/maxConst*
dtype0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
valueB
 *???=*
_output_shapes
: 
?
Tyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformLyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
dtype0*
seed2 *

seed *'
_output_shapes
:?
?
Jyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/subSubJyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/maxJyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/mulMulTyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/RandomUniformJyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*'
_output_shapes
:?
?
Fyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniformAddJyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/mulJyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*'
_output_shapes
:?
?
+yolov3_tiny/yolov3_tiny_head/Conv_2/weights
VariableV2*
shared_name *>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
dtype0*
	container *
shape:?*'
_output_shapes
:?
?
2yolov3_tiny/yolov3_tiny_head/Conv_2/weights/AssignAssign+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsFyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
validate_shape(*
use_locking(*'
_output_shapes
:?
?
0yolov3_tiny/yolov3_tiny_head/Conv_2/weights/readIdentity+yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*'
_output_shapes
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Initializer/zerosConst*
dtype0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
valueB*    *
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_2/biases
VariableV2*
shared_name *=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
dtype0*
	container *
shape:*
_output_shapes
:
?
1yolov3_tiny/yolov3_tiny_head/Conv_2/biases/AssignAssign*yolov3_tiny/yolov3_tiny_head/Conv_2/biases<yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Initializer/zeros*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
validate_shape(*
use_locking(*
_output_shapes
:
?
/yolov3_tiny/yolov3_tiny_head/Conv_2/biases/readIdentity*yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
_output_shapes
:
?
1yolov3_tiny/yolov3_tiny_head/Conv_2/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_2/Conv2DConv2D-yolov3_tiny/yolov3_tiny_head/Conv_1/LeakyRelu0yolov3_tiny/yolov3_tiny_head/Conv_2/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
:
?
+yolov3_tiny/yolov3_tiny_head/Conv_2/BiasAddBiasAdd*yolov3_tiny/yolov3_tiny_head/Conv_2/Conv2D/yolov3_tiny/yolov3_tiny_head/Conv_2/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/feature_map_1Identity+yolov3_tiny/yolov3_tiny_head/Conv_2/BiasAdd*
T0*&
_output_shapes
:
?
Lyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*%
valueB"         ?   *
dtype0*
_output_shapes
:
?
Jyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/minConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
Tyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/RandomUniformRandomUniformLyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
Jyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/subSubJyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/maxJyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/mulMulTyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/RandomUniformJyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
Fyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniformAddJyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/mulJyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
+yolov3_tiny/yolov3_tiny_head/Conv_3/weights
VariableV2*
dtype0*
	container *
shape:??*
shared_name *>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
2yolov3_tiny/yolov3_tiny_head/Conv_3/weights/AssignAssign+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsFyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
0yolov3_tiny/yolov3_tiny_head/Conv_3/weights/readIdentity+yolov3_tiny/yolov3_tiny_head/Conv_3/weights*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
1yolov3_tiny/yolov3_tiny_head/Conv_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_3/Conv2DConv2D+yolov3_tiny/yolov3_tiny_head/Conv/LeakyRelu0yolov3_tiny/yolov3_tiny_head/Conv_3/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Initializer/onesConst*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma
VariableV2*
shared_name *F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
dtype0*
	container *
shape:?*
_output_shapes	
:?
?
:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/AssignAssign3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammaDyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
8yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/readIdentity3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Initializer/zerosConst*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta
VariableV2*
shape:?*
shared_name *E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/AssignAssign2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betaDyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
_output_shapes	
:?
?
7yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/readIdentity2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
_output_shapes	
:?
?
Kyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Initializer/zerosConst*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean
VariableV2*
shape:?*
shared_name *L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:?
?
@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/AssignAssign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanKyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
>yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/readIdentity9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
_output_shapes	
:?
?
Nyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance
VariableV2*
shape:?*
shared_name *P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/AssignAssign=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceNyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/readIdentity=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
_output_shapes	
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/FusedBatchNormFusedBatchNorm*yolov3_tiny/yolov3_tiny_head/Conv_3/Conv2D8yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/read7yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/read>yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/readByolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%??'7*C
_output_shapes1
/:?:?:?:?:?
x
3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
-yolov3_tiny/yolov3_tiny_head/Conv_3/LeakyRelu	LeakyRelu<yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
|
+yolov3_tiny/yolov3_tiny_head/upsampled/sizeConst*
dtype0*
valueB"      *
_output_shapes
:
?
&yolov3_tiny/yolov3_tiny_head/upsampledResizeNearestNeighbor-yolov3_tiny/yolov3_tiny_head/Conv_3/LeakyRelu+yolov3_tiny/yolov3_tiny_head/upsampled/size*
align_corners(*
half_pixel_centers( *
T0*'
_output_shapes
:?
j
(yolov3_tiny/yolov3_tiny_head/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
#yolov3_tiny/yolov3_tiny_head/concatConcatV2&yolov3_tiny/yolov3_tiny_head/upsampled.yolov3_tiny/darknet19_body/MaxPool2D_3/MaxPool(yolov3_tiny/yolov3_tiny_head/concat/axis*
N*

Tidx0*
T0*'
_output_shapes
:?
?
Lyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/shapeConst*
dtype0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*%
valueB"            *
_output_shapes
:
?
Jyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/minConst*
dtype0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
valueB
 *:??*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
valueB
 *:?=*
dtype0*
_output_shapes
: 
?
Tyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/RandomUniformRandomUniformLyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
dtype0*
seed2 *

seed *(
_output_shapes
:??
?
Jyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/subSubJyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/maxJyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/mulMulTyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/RandomUniformJyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
Fyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniformAddJyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/mulJyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
+yolov3_tiny/yolov3_tiny_head/Conv_4/weights
VariableV2*
dtype0*
	container *
shape:??*
shared_name *>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
2yolov3_tiny/yolov3_tiny_head/Conv_4/weights/AssignAssign+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsFyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
validate_shape(*(
_output_shapes
:??
?
0yolov3_tiny/yolov3_tiny_head/Conv_4/weights/readIdentity+yolov3_tiny/yolov3_tiny_head/Conv_4/weights*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
1yolov3_tiny/yolov3_tiny_head/Conv_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_4/Conv2DConv2D#yolov3_tiny/yolov3_tiny_head/concat0yolov3_tiny/yolov3_tiny_head/Conv_4/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Initializer/onesConst*
dtype0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
valueB?*  ??*
_output_shapes	
:?
?
3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma
VariableV2*
shape:?*
shared_name *F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:?
?
:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/AssignAssign3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammaDyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Initializer/ones*
validate_shape(*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
_output_shapes	
:?
?
8yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/readIdentity3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Initializer/zerosConst*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta
VariableV2*
shape:?*
shared_name *E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/AssignAssign2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betaDyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Initializer/zeros*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
7yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/readIdentity2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
_output_shapes	
:?
?
Kyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Initializer/zerosConst*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean
VariableV2*
shape:?*
shared_name *L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:?
?
@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/AssignAssign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanKyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
>yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/readIdentity9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
_output_shapes	
:?
?
Nyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/Initializer/onesConst*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance
VariableV2*
dtype0*
	container *
shape:?*
shared_name *P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
_output_shapes	
:?
?
Dyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/AssignAssign=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceNyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
_output_shapes	
:?
?
Byolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/readIdentity=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
_output_shapes	
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/FusedBatchNormFusedBatchNorm*yolov3_tiny/yolov3_tiny_head/Conv_4/Conv2D8yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/read7yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/read>yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/readByolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%??'7*
T0*C
_output_shapes1
/:?:?:?:?:?
x
3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
?
-yolov3_tiny/yolov3_tiny_head/Conv_4/LeakyRelu	LeakyRelu<yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/FusedBatchNorm*
T0*
alpha%???=*'
_output_shapes
:?
?
Lyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Jyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/minConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
valueB
 *ԇ?*
dtype0*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
valueB
 *ԇ>*
dtype0*
_output_shapes
: 
?
Tyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/RandomUniformRandomUniformLyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*'
_output_shapes
:?
?
Jyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/subSubJyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/maxJyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
_output_shapes
: 
?
Jyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/mulMulTyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/RandomUniformJyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*'
_output_shapes
:?
?
Fyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniformAddJyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/mulJyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*'
_output_shapes
:?
?
+yolov3_tiny/yolov3_tiny_head/Conv_5/weights
VariableV2*
shape:?*
shared_name *>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
dtype0*
	container *'
_output_shapes
:?
?
2yolov3_tiny/yolov3_tiny_head/Conv_5/weights/AssignAssign+yolov3_tiny/yolov3_tiny_head/Conv_5/weightsFyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
validate_shape(*
use_locking(*'
_output_shapes
:?
?
0yolov3_tiny/yolov3_tiny_head/Conv_5/weights/readIdentity+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*'
_output_shapes
:?
?
<yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Initializer/zerosConst*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
valueB*    *
dtype0*
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_5/biases
VariableV2*
shared_name *=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
dtype0*
	container *
shape:*
_output_shapes
:
?
1yolov3_tiny/yolov3_tiny_head/Conv_5/biases/AssignAssign*yolov3_tiny/yolov3_tiny_head/Conv_5/biases<yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Initializer/zeros*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
validate_shape(*
use_locking(*
_output_shapes
:
?
/yolov3_tiny/yolov3_tiny_head/Conv_5/biases/readIdentity*yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
_output_shapes
:
?
1yolov3_tiny/yolov3_tiny_head/Conv_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/Conv_5/Conv2DConv2D-yolov3_tiny/yolov3_tiny_head/Conv_4/LeakyRelu0yolov3_tiny/yolov3_tiny_head/Conv_5/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
?
+yolov3_tiny/yolov3_tiny_head/Conv_5/BiasAddBiasAdd*yolov3_tiny/yolov3_tiny_head/Conv_5/Conv2D/yolov3_tiny/yolov3_tiny_head/Conv_5/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:
?
*yolov3_tiny/yolov3_tiny_head/feature_map_2Identity+yolov3_tiny/yolov3_tiny_head/Conv_5/BiasAdd*
T0*&
_output_shapes
:
e
yolov3_tiny/save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
?
yolov3_tiny/save/filenamePlaceholderWithDefaultyolov3_tiny/save/filename/input*
dtype0*
shape: *
_output_shapes
: 
}
yolov3_tiny/save/ConstPlaceholderWithDefaultyolov3_tiny/save/filename*
dtype0*
shape: *
_output_shapes
: 
?
$yolov3_tiny/save/SaveV2/tensor_namesConst*
dtype0*?
value?B?;B.yolov3_tiny/darknet19_body/Conv/BatchNorm/betaB/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammaB5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanB9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceB'yolov3_tiny/darknet19_body/Conv/weightsB0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_1/weightsB0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_2/weightsB0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_3/weightsB0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_4/weightsB0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_5/weightsB0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_6/weightsB0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betaB1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammaB7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanB;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceB)yolov3_tiny/yolov3_tiny_head/Conv/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
_output_shapes
:;
?
(yolov3_tiny/save/SaveV2/shape_and_slicesConst*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:;
?
yolov3_tiny/save/SaveV2SaveV2yolov3_tiny/save/Const$yolov3_tiny/save/SaveV2/tensor_names(yolov3_tiny/save/SaveV2/shape_and_slices.yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance'yolov3_tiny/darknet19_body/Conv/weights0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_1/weights0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_2/weights0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_3/weights0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_4/weights0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_5/weights0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_6/weights0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance)yolov3_tiny/yolov3_tiny_head/Conv/weights2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_1/weights*yolov3_tiny/yolov3_tiny_head/Conv_2/biases+yolov3_tiny/yolov3_tiny_head/Conv_2/weights2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_3/weights2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_4/weights*yolov3_tiny/yolov3_tiny_head/Conv_5/biases+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*I
dtypes?
=2;
?
#yolov3_tiny/save/control_dependencyIdentityyolov3_tiny/save/Const^yolov3_tiny/save/SaveV2*
T0*)
_class
loc:@yolov3_tiny/save/Const*
_output_shapes
: 
?
'yolov3_tiny/save/RestoreV2/tensor_namesConst*?
value?B?;B.yolov3_tiny/darknet19_body/Conv/BatchNorm/betaB/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammaB5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanB9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceB'yolov3_tiny/darknet19_body/Conv/weightsB0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_1/weightsB0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_2/weightsB0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_3/weightsB0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_4/weightsB0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_5/weightsB0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_6/weightsB0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betaB1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammaB7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanB;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceB)yolov3_tiny/yolov3_tiny_head/Conv/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
dtype0*
_output_shapes
:;
?
+yolov3_tiny/save/RestoreV2/shape_and_slicesConst*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:;
?
yolov3_tiny/save/RestoreV2	RestoreV2yolov3_tiny/save/Const'yolov3_tiny/save/RestoreV2/tensor_names+yolov3_tiny/save/RestoreV2/shape_and_slices*I
dtypes?
=2;*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
yolov3_tiny/save/AssignAssign.yolov3_tiny/darknet19_body/Conv/BatchNorm/betayolov3_tiny/save/RestoreV2*
use_locking(*
T0*A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save/Assign_1Assign/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammayolov3_tiny/save/RestoreV2:1*
use_locking(*
T0*B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save/Assign_2Assign5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:2*
use_locking(*
T0*H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save/Assign_3Assign9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:3*
validate_shape(*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
_output_shapes
:
?
yolov3_tiny/save/Assign_4Assign'yolov3_tiny/darknet19_body/Conv/weightsyolov3_tiny/save/RestoreV2:4*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*
validate_shape(*
use_locking(*&
_output_shapes
:
?
yolov3_tiny/save/Assign_5Assign0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betayolov3_tiny/save/RestoreV2:5*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
?
yolov3_tiny/save/Assign_6Assign1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammayolov3_tiny/save/RestoreV2:6*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
_output_shapes
: 
?
yolov3_tiny/save/Assign_7Assign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:7*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
: 
?
yolov3_tiny/save/Assign_8Assign;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:8*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
: 
?
yolov3_tiny/save/Assign_9Assign)yolov3_tiny/darknet19_body/Conv_1/weightsyolov3_tiny/save/RestoreV2:9*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*&
_output_shapes
: 
?
yolov3_tiny/save/Assign_10Assign0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betayolov3_tiny/save/RestoreV2:10*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
yolov3_tiny/save/Assign_11Assign1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammayolov3_tiny/save/RestoreV2:11*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
yolov3_tiny/save/Assign_12Assign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:12*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
_output_shapes
:@
?
yolov3_tiny/save/Assign_13Assign;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:13*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
yolov3_tiny/save/Assign_14Assign)yolov3_tiny/darknet19_body/Conv_2/weightsyolov3_tiny/save/RestoreV2:14*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*&
_output_shapes
: @
?
yolov3_tiny/save/Assign_15Assign0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betayolov3_tiny/save/RestoreV2:15*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_16Assign1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammayolov3_tiny/save/RestoreV2:16*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_17Assign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:17*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_18Assign;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:18*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_19Assign)yolov3_tiny/darknet19_body/Conv_3/weightsyolov3_tiny/save/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
validate_shape(*'
_output_shapes
:@?
?
yolov3_tiny/save/Assign_20Assign0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betayolov3_tiny/save/RestoreV2:20*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_21Assign1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammayolov3_tiny/save/RestoreV2:21*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_22Assign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:22*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_23Assign;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:23*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_24Assign)yolov3_tiny/darknet19_body/Conv_4/weightsyolov3_tiny/save/RestoreV2:24*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*
validate_shape(*
use_locking(*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_25Assign0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betayolov3_tiny/save/RestoreV2:25*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_26Assign1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammayolov3_tiny/save/RestoreV2:26*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_27Assign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:27*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_28Assign;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:28*
validate_shape(*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_29Assign)yolov3_tiny/darknet19_body/Conv_5/weightsyolov3_tiny/save/RestoreV2:29*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*
validate_shape(*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_30Assign0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betayolov3_tiny/save/RestoreV2:30*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_31Assign1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammayolov3_tiny/save/RestoreV2:31*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_32Assign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:32*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_33Assign;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:33*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_34Assign)yolov3_tiny/darknet19_body/Conv_6/weightsyolov3_tiny/save/RestoreV2:34*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*
validate_shape(*
use_locking(*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_35Assign0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betayolov3_tiny/save/RestoreV2:35*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_36Assign1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammayolov3_tiny/save/RestoreV2:36*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_37Assign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:37*
T0*J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_38Assign;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:38*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_39Assign)yolov3_tiny/yolov3_tiny_head/Conv/weightsyolov3_tiny/save/RestoreV2:39*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
validate_shape(*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_40Assign2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betayolov3_tiny/save/RestoreV2:40*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_41Assign3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammayolov3_tiny/save/RestoreV2:41*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_42Assign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:42*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_43Assign=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:43*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_44Assign+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsyolov3_tiny/save/RestoreV2:44*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_45Assign*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesyolov3_tiny/save/RestoreV2:45*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
validate_shape(*
use_locking(*
_output_shapes
:
?
yolov3_tiny/save/Assign_46Assign+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsyolov3_tiny/save/RestoreV2:46*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
validate_shape(*
use_locking(*'
_output_shapes
:?
?
yolov3_tiny/save/Assign_47Assign2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betayolov3_tiny/save/RestoreV2:47*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_48Assign3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammayolov3_tiny/save/RestoreV2:48*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_49Assign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:49*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_50Assign=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:50*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_51Assign+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsyolov3_tiny/save/RestoreV2:51*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_52Assign2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betayolov3_tiny/save/RestoreV2:52*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_53Assign3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammayolov3_tiny/save/RestoreV2:53*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_54Assign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanyolov3_tiny/save/RestoreV2:54*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_55Assign=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceyolov3_tiny/save/RestoreV2:55*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save/Assign_56Assign+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsyolov3_tiny/save/RestoreV2:56*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
yolov3_tiny/save/Assign_57Assign*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesyolov3_tiny/save/RestoreV2:57*
use_locking(*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save/Assign_58Assign+yolov3_tiny/yolov3_tiny_head/Conv_5/weightsyolov3_tiny/save/RestoreV2:58*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
validate_shape(*'
_output_shapes
:?
?
yolov3_tiny/save/restore_allNoOp^yolov3_tiny/save/Assign^yolov3_tiny/save/Assign_1^yolov3_tiny/save/Assign_10^yolov3_tiny/save/Assign_11^yolov3_tiny/save/Assign_12^yolov3_tiny/save/Assign_13^yolov3_tiny/save/Assign_14^yolov3_tiny/save/Assign_15^yolov3_tiny/save/Assign_16^yolov3_tiny/save/Assign_17^yolov3_tiny/save/Assign_18^yolov3_tiny/save/Assign_19^yolov3_tiny/save/Assign_2^yolov3_tiny/save/Assign_20^yolov3_tiny/save/Assign_21^yolov3_tiny/save/Assign_22^yolov3_tiny/save/Assign_23^yolov3_tiny/save/Assign_24^yolov3_tiny/save/Assign_25^yolov3_tiny/save/Assign_26^yolov3_tiny/save/Assign_27^yolov3_tiny/save/Assign_28^yolov3_tiny/save/Assign_29^yolov3_tiny/save/Assign_3^yolov3_tiny/save/Assign_30^yolov3_tiny/save/Assign_31^yolov3_tiny/save/Assign_32^yolov3_tiny/save/Assign_33^yolov3_tiny/save/Assign_34^yolov3_tiny/save/Assign_35^yolov3_tiny/save/Assign_36^yolov3_tiny/save/Assign_37^yolov3_tiny/save/Assign_38^yolov3_tiny/save/Assign_39^yolov3_tiny/save/Assign_4^yolov3_tiny/save/Assign_40^yolov3_tiny/save/Assign_41^yolov3_tiny/save/Assign_42^yolov3_tiny/save/Assign_43^yolov3_tiny/save/Assign_44^yolov3_tiny/save/Assign_45^yolov3_tiny/save/Assign_46^yolov3_tiny/save/Assign_47^yolov3_tiny/save/Assign_48^yolov3_tiny/save/Assign_49^yolov3_tiny/save/Assign_5^yolov3_tiny/save/Assign_50^yolov3_tiny/save/Assign_51^yolov3_tiny/save/Assign_52^yolov3_tiny/save/Assign_53^yolov3_tiny/save/Assign_54^yolov3_tiny/save/Assign_55^yolov3_tiny/save/Assign_56^yolov3_tiny/save/Assign_57^yolov3_tiny/save/Assign_58^yolov3_tiny/save/Assign_6^yolov3_tiny/save/Assign_7^yolov3_tiny/save/Assign_8^yolov3_tiny/save/Assign_9
g
!yolov3_tiny/save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
yolov3_tiny/save_1/filenamePlaceholderWithDefault!yolov3_tiny/save_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
?
yolov3_tiny/save_1/ConstPlaceholderWithDefaultyolov3_tiny/save_1/filename*
shape: *
dtype0*
_output_shapes
: 
?
&yolov3_tiny/save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_af5c71c3102340d4931d74533e0ce851/part*
dtype0*
_output_shapes
: 
?
yolov3_tiny/save_1/StringJoin
StringJoinyolov3_tiny/save_1/Const&yolov3_tiny/save_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
_
yolov3_tiny/save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
j
(yolov3_tiny/save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
"yolov3_tiny/save_1/ShardedFilenameShardedFilenameyolov3_tiny/save_1/StringJoin(yolov3_tiny/save_1/ShardedFilename/shardyolov3_tiny/save_1/num_shards*
_output_shapes
: 
?
&yolov3_tiny/save_1/SaveV2/tensor_namesConst*?
value?B?;B.yolov3_tiny/darknet19_body/Conv/BatchNorm/betaB/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammaB5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanB9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceB'yolov3_tiny/darknet19_body/Conv/weightsB0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_1/weightsB0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_2/weightsB0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_3/weightsB0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_4/weightsB0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_5/weightsB0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_6/weightsB0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betaB1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammaB7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanB;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceB)yolov3_tiny/yolov3_tiny_head/Conv/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
dtype0*
_output_shapes
:;
?
*yolov3_tiny/save_1/SaveV2/shape_and_slicesConst*
dtype0*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:;
?
yolov3_tiny/save_1/SaveV2SaveV2"yolov3_tiny/save_1/ShardedFilename&yolov3_tiny/save_1/SaveV2/tensor_names*yolov3_tiny/save_1/SaveV2/shape_and_slices.yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance'yolov3_tiny/darknet19_body/Conv/weights0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_1/weights0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_2/weights0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_3/weights0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_4/weights0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_5/weights0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance)yolov3_tiny/darknet19_body/Conv_6/weights0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance)yolov3_tiny/yolov3_tiny_head/Conv/weights2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_1/weights*yolov3_tiny/yolov3_tiny_head/Conv_2/biases+yolov3_tiny/yolov3_tiny_head/Conv_2/weights2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_3/weights2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance+yolov3_tiny/yolov3_tiny_head/Conv_4/weights*yolov3_tiny/yolov3_tiny_head/Conv_5/biases+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*I
dtypes?
=2;
?
%yolov3_tiny/save_1/control_dependencyIdentity"yolov3_tiny/save_1/ShardedFilename^yolov3_tiny/save_1/SaveV2*
T0*5
_class+
)'loc:@yolov3_tiny/save_1/ShardedFilename*
_output_shapes
: 
?
9yolov3_tiny/save_1/MergeV2Checkpoints/checkpoint_prefixesPack"yolov3_tiny/save_1/ShardedFilename&^yolov3_tiny/save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
?
%yolov3_tiny/save_1/MergeV2CheckpointsMergeV2Checkpoints9yolov3_tiny/save_1/MergeV2Checkpoints/checkpoint_prefixesyolov3_tiny/save_1/Const*
delete_old_dirs(
?
yolov3_tiny/save_1/IdentityIdentityyolov3_tiny/save_1/Const&^yolov3_tiny/save_1/MergeV2Checkpoints&^yolov3_tiny/save_1/control_dependency*
T0*
_output_shapes
: 
?
)yolov3_tiny/save_1/RestoreV2/tensor_namesConst*?
value?B?;B.yolov3_tiny/darknet19_body/Conv/BatchNorm/betaB/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammaB5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanB9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceB'yolov3_tiny/darknet19_body/Conv/weightsB0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_1/weightsB0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_2/weightsB0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_3/weightsB0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_4/weightsB0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_5/weightsB0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betaB1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammaB7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanB;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceB)yolov3_tiny/darknet19_body/Conv_6/weightsB0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betaB1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammaB7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanB;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceB)yolov3_tiny/yolov3_tiny_head/Conv/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsB2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betaB3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammaB9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanB=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceB+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsB*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesB+yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
dtype0*
_output_shapes
:;
?
-yolov3_tiny/save_1/RestoreV2/shape_and_slicesConst*?
value?B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:;
?
yolov3_tiny/save_1/RestoreV2	RestoreV2yolov3_tiny/save_1/Const)yolov3_tiny/save_1/RestoreV2/tensor_names-yolov3_tiny/save_1/RestoreV2/shape_and_slices*I
dtypes?
=2;*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
yolov3_tiny/save_1/AssignAssign.yolov3_tiny/darknet19_body/Conv/BatchNorm/betayolov3_tiny/save_1/RestoreV2*
use_locking(*
T0*A
_class7
53loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/beta*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_1Assign/yolov3_tiny/darknet19_body/Conv/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:1*
use_locking(*
T0*B
_class8
64loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_2Assign5yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:2*
T0*H
_class>
<:loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_3Assign9yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:3*
validate_shape(*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_4Assign'yolov3_tiny/darknet19_body/Conv/weightsyolov3_tiny/save_1/RestoreV2:4*
validate_shape(*
use_locking(*
T0*:
_class0
.,loc:@yolov3_tiny/darknet19_body/Conv/weights*&
_output_shapes
:
?
yolov3_tiny/save_1/Assign_5Assign0yolov3_tiny/darknet19_body/Conv_1/BatchNorm/betayolov3_tiny/save_1/RestoreV2:5*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta*
_output_shapes
: 
?
yolov3_tiny/save_1/Assign_6Assign1yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:6*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma*
_output_shapes
: 
?
yolov3_tiny/save_1/Assign_7Assign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:7*
validate_shape(*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean*
_output_shapes
: 
?
yolov3_tiny/save_1/Assign_8Assign;yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:8*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
: 
?
yolov3_tiny/save_1/Assign_9Assign)yolov3_tiny/darknet19_body/Conv_1/weightsyolov3_tiny/save_1/RestoreV2:9*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_1/weights*
validate_shape(*
use_locking(*&
_output_shapes
: 
?
yolov3_tiny/save_1/Assign_10Assign0yolov3_tiny/darknet19_body/Conv_2/BatchNorm/betayolov3_tiny/save_1/RestoreV2:10*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta*
_output_shapes
:@
?
yolov3_tiny/save_1/Assign_11Assign1yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:11*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes
:@
?
yolov3_tiny/save_1/Assign_12Assign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:12*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
yolov3_tiny/save_1/Assign_13Assign;yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:13*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
yolov3_tiny/save_1/Assign_14Assign)yolov3_tiny/darknet19_body/Conv_2/weightsyolov3_tiny/save_1/RestoreV2:14*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_2/weights*
validate_shape(*
use_locking(*&
_output_shapes
: @
?
yolov3_tiny/save_1/Assign_15Assign0yolov3_tiny/darknet19_body/Conv_3/BatchNorm/betayolov3_tiny/save_1/RestoreV2:15*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_16Assign1yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:16*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_17Assign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:17*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_18Assign;yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:18*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_19Assign)yolov3_tiny/darknet19_body/Conv_3/weightsyolov3_tiny/save_1/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_3/weights*
validate_shape(*'
_output_shapes
:@?
?
yolov3_tiny/save_1/Assign_20Assign0yolov3_tiny/darknet19_body/Conv_4/BatchNorm/betayolov3_tiny/save_1/RestoreV2:20*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_21Assign1yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:21*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_22Assign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:22*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_23Assign;yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:23*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_24Assign)yolov3_tiny/darknet19_body/Conv_4/weightsyolov3_tiny/save_1/RestoreV2:24*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_4/weights*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_25Assign0yolov3_tiny/darknet19_body/Conv_5/BatchNorm/betayolov3_tiny/save_1/RestoreV2:25*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_26Assign1yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:26*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_27Assign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:27*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_28Assign;yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:28*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_29Assign)yolov3_tiny/darknet19_body/Conv_5/weightsyolov3_tiny/save_1/RestoreV2:29*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_5/weights*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_30Assign0yolov3_tiny/darknet19_body/Conv_6/BatchNorm/betayolov3_tiny/save_1/RestoreV2:30*
T0*C
_class9
75loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_31Assign1yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:31*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_32Assign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:32*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_33Assign;yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:33*
T0*N
_classD
B@loc:@yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_34Assign)yolov3_tiny/darknet19_body/Conv_6/weightsyolov3_tiny/save_1/RestoreV2:34*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/darknet19_body/Conv_6/weights*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_35Assign0yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/betayolov3_tiny/save_1/RestoreV2:35*
use_locking(*
T0*C
_class9
75loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_36Assign1yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:36*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_37Assign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:37*
use_locking(*
T0*J
_class@
><loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_38Assign;yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:38*
use_locking(*
T0*N
_classD
B@loc:@yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_39Assign)yolov3_tiny/yolov3_tiny_head/Conv/weightsyolov3_tiny/save_1/RestoreV2:39*
use_locking(*
T0*<
_class2
0.loc:@yolov3_tiny/yolov3_tiny_head/Conv/weights*
validate_shape(*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_40Assign2yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/betayolov3_tiny/save_1/RestoreV2:40*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_41Assign3yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:41*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_42Assign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:42*
use_locking(*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_43Assign=yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:43*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_44Assign+yolov3_tiny/yolov3_tiny_head/Conv_1/weightsyolov3_tiny/save_1/RestoreV2:44*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_1/weights*
validate_shape(*
use_locking(*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_45Assign*yolov3_tiny/yolov3_tiny_head/Conv_2/biasesyolov3_tiny/save_1/RestoreV2:45*
use_locking(*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/biases*
validate_shape(*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_46Assign+yolov3_tiny/yolov3_tiny_head/Conv_2/weightsyolov3_tiny/save_1/RestoreV2:46*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_2/weights*
validate_shape(*'
_output_shapes
:?
?
yolov3_tiny/save_1/Assign_47Assign2yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/betayolov3_tiny/save_1/RestoreV2:47*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_48Assign3yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:48*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_49Assign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:49*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_50Assign=yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:50*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_51Assign+yolov3_tiny/yolov3_tiny_head/Conv_3/weightsyolov3_tiny/save_1/RestoreV2:51*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_3/weights*
validate_shape(*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_52Assign2yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/betayolov3_tiny/save_1/RestoreV2:52*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_53Assign3yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gammayolov3_tiny/save_1/RestoreV2:53*
use_locking(*
T0*F
_class<
:8loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_54Assign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_meanyolov3_tiny/save_1/RestoreV2:54*
T0*L
_classB
@>loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_55Assign=yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_varianceyolov3_tiny/save_1/RestoreV2:55*
validate_shape(*
use_locking(*
T0*P
_classF
DBloc:@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance*
_output_shapes	
:?
?
yolov3_tiny/save_1/Assign_56Assign+yolov3_tiny/yolov3_tiny_head/Conv_4/weightsyolov3_tiny/save_1/RestoreV2:56*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_4/weights*(
_output_shapes
:??
?
yolov3_tiny/save_1/Assign_57Assign*yolov3_tiny/yolov3_tiny_head/Conv_5/biasesyolov3_tiny/save_1/RestoreV2:57*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/biases*
_output_shapes
:
?
yolov3_tiny/save_1/Assign_58Assign+yolov3_tiny/yolov3_tiny_head/Conv_5/weightsyolov3_tiny/save_1/RestoreV2:58*
T0*>
_class4
20loc:@yolov3_tiny/yolov3_tiny_head/Conv_5/weights*
validate_shape(*
use_locking(*'
_output_shapes
:?
?
 yolov3_tiny/save_1/restore_shardNoOp^yolov3_tiny/save_1/Assign^yolov3_tiny/save_1/Assign_1^yolov3_tiny/save_1/Assign_10^yolov3_tiny/save_1/Assign_11^yolov3_tiny/save_1/Assign_12^yolov3_tiny/save_1/Assign_13^yolov3_tiny/save_1/Assign_14^yolov3_tiny/save_1/Assign_15^yolov3_tiny/save_1/Assign_16^yolov3_tiny/save_1/Assign_17^yolov3_tiny/save_1/Assign_18^yolov3_tiny/save_1/Assign_19^yolov3_tiny/save_1/Assign_2^yolov3_tiny/save_1/Assign_20^yolov3_tiny/save_1/Assign_21^yolov3_tiny/save_1/Assign_22^yolov3_tiny/save_1/Assign_23^yolov3_tiny/save_1/Assign_24^yolov3_tiny/save_1/Assign_25^yolov3_tiny/save_1/Assign_26^yolov3_tiny/save_1/Assign_27^yolov3_tiny/save_1/Assign_28^yolov3_tiny/save_1/Assign_29^yolov3_tiny/save_1/Assign_3^yolov3_tiny/save_1/Assign_30^yolov3_tiny/save_1/Assign_31^yolov3_tiny/save_1/Assign_32^yolov3_tiny/save_1/Assign_33^yolov3_tiny/save_1/Assign_34^yolov3_tiny/save_1/Assign_35^yolov3_tiny/save_1/Assign_36^yolov3_tiny/save_1/Assign_37^yolov3_tiny/save_1/Assign_38^yolov3_tiny/save_1/Assign_39^yolov3_tiny/save_1/Assign_4^yolov3_tiny/save_1/Assign_40^yolov3_tiny/save_1/Assign_41^yolov3_tiny/save_1/Assign_42^yolov3_tiny/save_1/Assign_43^yolov3_tiny/save_1/Assign_44^yolov3_tiny/save_1/Assign_45^yolov3_tiny/save_1/Assign_46^yolov3_tiny/save_1/Assign_47^yolov3_tiny/save_1/Assign_48^yolov3_tiny/save_1/Assign_49^yolov3_tiny/save_1/Assign_5^yolov3_tiny/save_1/Assign_50^yolov3_tiny/save_1/Assign_51^yolov3_tiny/save_1/Assign_52^yolov3_tiny/save_1/Assign_53^yolov3_tiny/save_1/Assign_54^yolov3_tiny/save_1/Assign_55^yolov3_tiny/save_1/Assign_56^yolov3_tiny/save_1/Assign_57^yolov3_tiny/save_1/Assign_58^yolov3_tiny/save_1/Assign_6^yolov3_tiny/save_1/Assign_7^yolov3_tiny/save_1/Assign_8^yolov3_tiny/save_1/Assign_9
I
yolov3_tiny/save_1/restore_allNoOp!^yolov3_tiny/save_1/restore_shard "&f
yolov3_tiny/save_1/Const:0yolov3_tiny/save_1/Identity:0yolov3_tiny/save_1/restore_all (5 @F8"?D
trainable_variables?D?D
?
)yolov3_tiny/darknet19_body/Conv/weights:0.yolov3_tiny/darknet19_body/Conv/weights/Assign.yolov3_tiny/darknet19_body/Conv/weights/read:02Dyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform:08
?
1yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma:06yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Assign6yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Initializer/ones:08
?
0yolov3_tiny/darknet19_body/Conv/BatchNorm/beta:05yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Assign5yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_1/weights:00yolov3_tiny/darknet19_body/Conv_1/weights/Assign0yolov3_tiny/darknet19_body/Conv_1/weights/read:02Fyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_2/weights:00yolov3_tiny/darknet19_body/Conv_2/weights/Assign0yolov3_tiny/darknet19_body/Conv_2/weights/read:02Fyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_3/weights:00yolov3_tiny/darknet19_body/Conv_3/weights/Assign0yolov3_tiny/darknet19_body/Conv_3/weights/read:02Fyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_4/weights:00yolov3_tiny/darknet19_body/Conv_4/weights/Assign0yolov3_tiny/darknet19_body/Conv_4/weights/read:02Fyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_5/weights:00yolov3_tiny/darknet19_body/Conv_5/weights/Assign0yolov3_tiny/darknet19_body/Conv_5/weights/read:02Fyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/darknet19_body/Conv_6/weights:00yolov3_tiny/darknet19_body/Conv_6/weights/Assign0yolov3_tiny/darknet19_body/Conv_6/weights/read:02Fyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros:08
?
+yolov3_tiny/yolov3_tiny_head/Conv/weights:00yolov3_tiny/yolov3_tiny_head/Conv/weights/Assign0yolov3_tiny/yolov3_tiny_head/Conv/weights/read:02Fyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform:08
?
3yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma:08yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Assign8yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta:07yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Assign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_1/weights:02yolov3_tiny/yolov3_tiny_head/Conv_1/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_1/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_2/weights:02yolov3_tiny/yolov3_tiny_head/Conv_2/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_2/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_2/biases:01yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_2/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_3/weights:02yolov3_tiny/yolov3_tiny_head/Conv_3/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_3/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_4/weights:02yolov3_tiny/yolov3_tiny_head/Conv_4/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_4/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_5/weights:02yolov3_tiny/yolov3_tiny_head/Conv_5/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_5/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_5/biases:01yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_5/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Initializer/zeros:08"?t
	variables?t?t
?
)yolov3_tiny/darknet19_body/Conv/weights:0.yolov3_tiny/darknet19_body/Conv/weights/Assign.yolov3_tiny/darknet19_body/Conv/weights/read:02Dyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform:08
?
1yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma:06yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Assign6yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Initializer/ones:08
?
0yolov3_tiny/darknet19_body/Conv/BatchNorm/beta:05yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Assign5yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Initializer/zeros:08
?
7yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean:0<yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Assign<yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/read:02Iyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Initializer/zeros:0@H
?
;yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance:0@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Assign@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/read:02Lyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_1/weights:00yolov3_tiny/darknet19_body/Conv_1/weights/Assign0yolov3_tiny/darknet19_body/Conv_1/weights/read:02Fyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_2/weights:00yolov3_tiny/darknet19_body/Conv_2/weights/Assign0yolov3_tiny/darknet19_body/Conv_2/weights/read:02Fyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_3/weights:00yolov3_tiny/darknet19_body/Conv_3/weights/Assign0yolov3_tiny/darknet19_body/Conv_3/weights/read:02Fyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_4/weights:00yolov3_tiny/darknet19_body/Conv_4/weights/Assign0yolov3_tiny/darknet19_body/Conv_4/weights/read:02Fyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_5/weights:00yolov3_tiny/darknet19_body/Conv_5/weights/Assign0yolov3_tiny/darknet19_body/Conv_5/weights/read:02Fyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_6/weights:00yolov3_tiny/darknet19_body/Conv_6/weights/Assign0yolov3_tiny/darknet19_body/Conv_6/weights/read:02Fyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/yolov3_tiny_head/Conv/weights:00yolov3_tiny/yolov3_tiny_head/Conv/weights/Assign0yolov3_tiny/yolov3_tiny_head/Conv/weights/read:02Fyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform:08
?
3yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma:08yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Assign8yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta:07yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Assign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean:0>yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Assign>yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/read:02Kyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance:0Byolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/AssignByolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/read:02Nyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_1/weights:02yolov3_tiny/yolov3_tiny_head/Conv_1/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_1/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_2/weights:02yolov3_tiny/yolov3_tiny_head/Conv_2/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_2/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_2/biases:01yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_2/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_3/weights:02yolov3_tiny/yolov3_tiny_head/Conv_3/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_3/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_4/weights:02yolov3_tiny/yolov3_tiny_head/Conv_4/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_4/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_5/weights:02yolov3_tiny/yolov3_tiny_head/Conv_5/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_5/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_5/biases:01yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_5/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Initializer/zeros:08"?t
model_variables?t?t
?
)yolov3_tiny/darknet19_body/Conv/weights:0.yolov3_tiny/darknet19_body/Conv/weights/Assign.yolov3_tiny/darknet19_body/Conv/weights/read:02Dyolov3_tiny/darknet19_body/Conv/weights/Initializer/random_uniform:08
?
1yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma:06yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Assign6yolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/gamma/Initializer/ones:08
?
0yolov3_tiny/darknet19_body/Conv/BatchNorm/beta:05yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Assign5yolov3_tiny/darknet19_body/Conv/BatchNorm/beta/read:02Byolov3_tiny/darknet19_body/Conv/BatchNorm/beta/Initializer/zeros:08
?
7yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean:0<yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Assign<yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/read:02Iyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_mean/Initializer/zeros:0@H
?
;yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance:0@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Assign@yolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/read:02Lyolov3_tiny/darknet19_body/Conv/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_1/weights:00yolov3_tiny/darknet19_body/Conv_1/weights/Assign0yolov3_tiny/darknet19_body/Conv_1/weights/read:02Fyolov3_tiny/darknet19_body/Conv_1/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_1/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_2/weights:00yolov3_tiny/darknet19_body/Conv_2/weights/Assign0yolov3_tiny/darknet19_body/Conv_2/weights/read:02Fyolov3_tiny/darknet19_body/Conv_2/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_2/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_2/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_3/weights:00yolov3_tiny/darknet19_body/Conv_3/weights/Assign0yolov3_tiny/darknet19_body/Conv_3/weights/read:02Fyolov3_tiny/darknet19_body/Conv_3/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_3/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_4/weights:00yolov3_tiny/darknet19_body/Conv_4/weights/Assign0yolov3_tiny/darknet19_body/Conv_4/weights/read:02Fyolov3_tiny/darknet19_body/Conv_4/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_4/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_5/weights:00yolov3_tiny/darknet19_body/Conv_5/weights/Assign0yolov3_tiny/darknet19_body/Conv_5/weights/read:02Fyolov3_tiny/darknet19_body/Conv_5/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_5/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_5/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/darknet19_body/Conv_6/weights:00yolov3_tiny/darknet19_body/Conv_6/weights/Assign0yolov3_tiny/darknet19_body/Conv_6/weights/read:02Fyolov3_tiny/darknet19_body/Conv_6/weights/Initializer/random_uniform:08
?
3yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma:08yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Assign8yolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta:07yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Assign7yolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/read:02Dyolov3_tiny/darknet19_body/Conv_6/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean:0>yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Assign>yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/read:02Kyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance:0Byolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/AssignByolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/read:02Nyolov3_tiny/darknet19_body/Conv_6/BatchNorm/moving_variance/Initializer/ones:0@H
?
+yolov3_tiny/yolov3_tiny_head/Conv/weights:00yolov3_tiny/yolov3_tiny_head/Conv/weights/Assign0yolov3_tiny/yolov3_tiny_head/Conv/weights/read:02Fyolov3_tiny/yolov3_tiny_head/Conv/weights/Initializer/random_uniform:08
?
3yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma:08yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Assign8yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/gamma/Initializer/ones:08
?
2yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta:07yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Assign7yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/read:02Dyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/beta/Initializer/zeros:08
?
9yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean:0>yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Assign>yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/read:02Kyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_mean/Initializer/zeros:0@H
?
=yolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance:0Byolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/AssignByolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/read:02Nyolov3_tiny/yolov3_tiny_head/Conv/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_1/weights:02yolov3_tiny/yolov3_tiny_head/Conv_1/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_1/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_1/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_1/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_2/weights:02yolov3_tiny/yolov3_tiny_head/Conv_2/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_2/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_2/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_2/biases:01yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_2/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_2/biases/Initializer/zeros:08
?
-yolov3_tiny/yolov3_tiny_head/Conv_3/weights:02yolov3_tiny/yolov3_tiny_head/Conv_3/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_3/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_3/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_3/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_4/weights:02yolov3_tiny/yolov3_tiny_head/Conv_4/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_4/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_4/weights/Initializer/random_uniform:08
?
5yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma:0:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Assign:yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/gamma/Initializer/ones:08
?
4yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta:09yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Assign9yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/read:02Fyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/beta/Initializer/zeros:08
?
;yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean:0@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Assign@yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/read:02Myolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_mean/Initializer/zeros:0@H
?
?yolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance:0Dyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/AssignDyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/read:02Pyolov3_tiny/yolov3_tiny_head/Conv_4/BatchNorm/moving_variance/Initializer/ones:0@H
?
-yolov3_tiny/yolov3_tiny_head/Conv_5/weights:02yolov3_tiny/yolov3_tiny_head/Conv_5/weights/Assign2yolov3_tiny/yolov3_tiny_head/Conv_5/weights/read:02Hyolov3_tiny/yolov3_tiny_head/Conv_5/weights/Initializer/random_uniform:08
?
,yolov3_tiny/yolov3_tiny_head/Conv_5/biases:01yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Assign1yolov3_tiny/yolov3_tiny_head/Conv_5/biases/read:02>yolov3_tiny/yolov3_tiny_head/Conv_5/biases/Initializer/zeros:08*?
serving_default?
3
input_image$
input_data:0??X
pred_feature_maps0B
,yolov3_tiny/yolov3_tiny_head/feature_map_1:0X
pred_feature_maps1B
,yolov3_tiny/yolov3_tiny_head/feature_map_2:0tensorflow/serving/predict