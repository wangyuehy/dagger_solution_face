??
??
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
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
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
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.15.32v1.15.2-30-g4386a66ͺ
p
image_batchPlaceholder*
shape:??*(
_output_shapes
:??*
dtype0
?
9squeezenet/conv1/weights/Initializer/random_uniform/shapeConst*%
valueB"         `   *+
_class!
loc:@squeezenet/conv1/weights*
dtype0*
_output_shapes
:
?
7squeezenet/conv1/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *^?*+
_class!
loc:@squeezenet/conv1/weights
?
7squeezenet/conv1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *^=*
dtype0*+
_class!
loc:@squeezenet/conv1/weights
?
Asqueezenet/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform9squeezenet/conv1/weights/Initializer/random_uniform/shape*
seed2 *+
_class!
loc:@squeezenet/conv1/weights*
dtype0*

seed *
T0*&
_output_shapes
:`
?
7squeezenet/conv1/weights/Initializer/random_uniform/subSub7squeezenet/conv1/weights/Initializer/random_uniform/max7squeezenet/conv1/weights/Initializer/random_uniform/min*+
_class!
loc:@squeezenet/conv1/weights*
_output_shapes
: *
T0
?
7squeezenet/conv1/weights/Initializer/random_uniform/mulMulAsqueezenet/conv1/weights/Initializer/random_uniform/RandomUniform7squeezenet/conv1/weights/Initializer/random_uniform/sub*+
_class!
loc:@squeezenet/conv1/weights*&
_output_shapes
:`*
T0
?
3squeezenet/conv1/weights/Initializer/random_uniformAdd7squeezenet/conv1/weights/Initializer/random_uniform/mul7squeezenet/conv1/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@squeezenet/conv1/weights*&
_output_shapes
:`
?
squeezenet/conv1/weights
VariableV2*+
_class!
loc:@squeezenet/conv1/weights*
	container *&
_output_shapes
:`*
shape:`*
shared_name *
dtype0
?
squeezenet/conv1/weights/AssignAssignsqueezenet/conv1/weights3squeezenet/conv1/weights/Initializer/random_uniform*
T0*+
_class!
loc:@squeezenet/conv1/weights*
use_locking(*
validate_shape(*&
_output_shapes
:`
?
squeezenet/conv1/weights/readIdentitysqueezenet/conv1/weights*+
_class!
loc:@squeezenet/conv1/weights*&
_output_shapes
:`*
T0
o
squeezenet/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
squeezenet/conv1/Conv2DConv2Dimage_batchsqueezenet/conv1/weights/read*
paddingSAME*
T0*
	dilations
*
data_formatNHWC*
explicit_paddings
 *
strides
*
use_cudnn_on_gpu(*&
_output_shapes
:PP`
?
1squeezenet/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta*
valueB`*    *
_output_shapes
:`
?
squeezenet/conv1/BatchNorm/beta
VariableV2*
_output_shapes
:`*
dtype0*
shape:`*
	container *
shared_name *2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta
?
&squeezenet/conv1/BatchNorm/beta/AssignAssignsqueezenet/conv1/BatchNorm/beta1squeezenet/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes
:`*2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta
?
$squeezenet/conv1/BatchNorm/beta/readIdentitysqueezenet/conv1/BatchNorm/beta*
T0*
_output_shapes
:`*2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta
m
 squeezenet/conv1/BatchNorm/ConstConst*
_output_shapes
:`*
dtype0*
valueB`*  ??
?
8squeezenet/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
_output_shapes
:`*
valueB`*    
?
&squeezenet/conv1/BatchNorm/moving_mean
VariableV2*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
_output_shapes
:`*
shape:`*
shared_name *
	container *
dtype0
?
-squeezenet/conv1/BatchNorm/moving_mean/AssignAssign&squeezenet/conv1/BatchNorm/moving_mean8squeezenet/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
_output_shapes
:`*
validate_shape(*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
T0
?
+squeezenet/conv1/BatchNorm/moving_mean/readIdentity&squeezenet/conv1/BatchNorm/moving_mean*
T0*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
_output_shapes
:`
?
;squeezenet/conv1/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:`*
valueB`*  ??*
dtype0*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance
?
*squeezenet/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
shape:`*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance*
shared_name *
	container *
_output_shapes
:`
?
1squeezenet/conv1/BatchNorm/moving_variance/AssignAssign*squeezenet/conv1/BatchNorm/moving_variance;squeezenet/conv1/BatchNorm/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*
_output_shapes
:`*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance
?
/squeezenet/conv1/BatchNorm/moving_variance/readIdentity*squeezenet/conv1/BatchNorm/moving_variance*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance*
_output_shapes
:`*
T0
?
+squeezenet/conv1/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/conv1/Conv2D squeezenet/conv1/BatchNorm/Const$squeezenet/conv1/BatchNorm/beta/read+squeezenet/conv1/BatchNorm/moving_mean/read/squeezenet/conv1/BatchNorm/moving_variance/read*
T0*
epsilon%o?:*
is_training( *
data_formatNHWC*
U0*B
_output_shapes0
.:PP`:`:`:`:`:
{
squeezenet/conv1/ReluRelu+squeezenet/conv1/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:PP`
?
squeezenet/maxpool1/MaxPoolMaxPoolsqueezenet/conv1/Relu*
paddingVALID*
data_formatNHWC*&
_output_shapes
:''`*
strides
*
ksize
*
T0
?
Asqueezenet/fire2/squeeze/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      `      *
dtype0*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights
?
?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/minConst*
valueB
 *?m?*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
_output_shapes
: *
dtype0
?
?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
valueB
 *?m>*
dtype0
?
Isqueezenet/fire2/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire2/squeeze/weights/Initializer/random_uniform/shape*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*

seed *&
_output_shapes
:`*
dtype0*
seed2 *
T0
?
?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights
?
?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire2/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
T0*&
_output_shapes
:`
?
;squeezenet/fire2/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire2/squeeze/weights/Initializer/random_uniform/min*
T0*&
_output_shapes
:`*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights
?
 squeezenet/fire2/squeeze/weights
VariableV2*
shared_name *
shape:`*
	container *3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*&
_output_shapes
:`*
dtype0
?
'squeezenet/fire2/squeeze/weights/AssignAssign squeezenet/fire2/squeeze/weights;squeezenet/fire2/squeeze/weights/Initializer/random_uniform*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:`
?
%squeezenet/fire2/squeeze/weights/readIdentity squeezenet/fire2/squeeze/weights*
T0*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*&
_output_shapes
:`
w
&squeezenet/fire2/squeeze/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
squeezenet/fire2/squeeze/Conv2DConv2Dsqueezenet/maxpool1/MaxPool%squeezenet/fire2/squeeze/weights/read*
strides
*
paddingSAME*&
_output_shapes
:''*
	dilations
*
T0*
explicit_paddings
 *
use_cudnn_on_gpu(*
data_formatNHWC
?
9squeezenet/fire2/squeeze/BatchNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta*
_output_shapes
:*
dtype0*
valueB*    
?
'squeezenet/fire2/squeeze/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
:*
	container *
shared_name *
shape:*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta
?
.squeezenet/fire2/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire2/squeeze/BatchNorm/beta9squeezenet/fire2/squeeze/BatchNorm/beta/Initializer/zeros*
T0*
_output_shapes
:*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta*
validate_shape(*
use_locking(
?
,squeezenet/fire2/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire2/squeeze/BatchNorm/beta*
T0*
_output_shapes
:*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta
u
(squeezenet/fire2/squeeze/BatchNorm/ConstConst*
dtype0*
valueB*  ??*
_output_shapes
:
?
@squeezenet/fire2/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB*    *A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean*
_output_shapes
:
?
.squeezenet/fire2/squeeze/BatchNorm/moving_mean
VariableV2*
	container *
shared_name *
shape:*
dtype0*
_output_shapes
:*A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean
?
5squeezenet/fire2/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire2/squeeze/BatchNorm/moving_mean@squeezenet/fire2/squeeze/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean
?
3squeezenet/fire2/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire2/squeeze/BatchNorm/moving_mean*
_output_shapes
:*
T0*A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean
?
Csqueezenet/fire2/squeeze/BatchNorm/moving_variance/Initializer/onesConst*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
2squeezenet/fire2/squeeze/BatchNorm/moving_variance
VariableV2*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
	container *
shared_name *
_output_shapes
:*
dtype0*
shape:
?
9squeezenet/fire2/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire2/squeeze/BatchNorm/moving_varianceCsqueezenet/fire2/squeeze/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
_output_shapes
:*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
validate_shape(*
T0
?
7squeezenet/fire2/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire2/squeeze/BatchNorm/moving_variance*
_output_shapes
:*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
T0
?
3squeezenet/fire2/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire2/squeeze/Conv2D(squeezenet/fire2/squeeze/BatchNorm/Const,squeezenet/fire2/squeeze/BatchNorm/beta/read3squeezenet/fire2/squeeze/BatchNorm/moving_mean/read7squeezenet/fire2/squeeze/BatchNorm/moving_variance/read*
T0*
U0*
data_formatNHWC*B
_output_shapes0
.:'':::::*
is_training( *
epsilon%o?:
?
squeezenet/fire2/squeeze/ReluRelu3squeezenet/fire2/squeeze/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:''
?
Dsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/shapeConst*
dtype0*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
_output_shapes
:*%
valueB"         @   
?
Bsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
_output_shapes
: *
dtype0*
valueB
 *?7??
?
Bsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
valueB
 *?7?>*
dtype0
?
Lsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/shape*
T0*
seed2 *
dtype0*

seed *6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*&
_output_shapes
:@
?
Bsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
T0*
_output_shapes
: 
?
Bsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/sub*&
_output_shapes
:@*
T0*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights
?
>squeezenet/fire2/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire2/expand/1x1/weights/Initializer/random_uniform/min*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
T0
?
#squeezenet/fire2/expand/1x1/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
shape:@*
shared_name *
dtype0*
	container *&
_output_shapes
:@
?
*squeezenet/fire2/expand/1x1/weights/AssignAssign#squeezenet/fire2/expand/1x1/weights>squeezenet/fire2/expand/1x1/weights/Initializer/random_uniform*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
T0*
validate_shape(*
use_locking(
?
(squeezenet/fire2/expand/1x1/weights/readIdentity#squeezenet/fire2/expand/1x1/weights*
T0*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*&
_output_shapes
:@
z
)squeezenet/fire2/expand/1x1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire2/expand/1x1/Conv2DConv2Dsqueezenet/fire2/squeeze/Relu(squeezenet/fire2/expand/1x1/weights/read*
use_cudnn_on_gpu(*
T0*&
_output_shapes
:''@*
explicit_paddings
 *
paddingSAME*
strides
*
data_formatNHWC*
	dilations

?
<squeezenet/fire2/expand/1x1/BatchNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta*
dtype0*
_output_shapes
:@*
valueB@*    
?
*squeezenet/fire2/expand/1x1/BatchNorm/beta
VariableV2*
shared_name *
	container *
_output_shapes
:@*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta*
dtype0*
shape:@
?
1squeezenet/fire2/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire2/expand/1x1/BatchNorm/beta<squeezenet/fire2/expand/1x1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta*
_output_shapes
:@
?
/squeezenet/fire2/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire2/expand/1x1/BatchNorm/beta*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta*
_output_shapes
:@*
T0
x
+squeezenet/fire2/expand/1x1/BatchNorm/ConstConst*
_output_shapes
:@*
valueB@*  ??*
dtype0
?
Csqueezenet/fire2/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB@*    *D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
_output_shapes
:@
?
1squeezenet/fire2/expand/1x1/BatchNorm/moving_mean
VariableV2*
	container *
dtype0*
shape:@*
shared_name *D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
_output_shapes
:@
?
8squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire2/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire2/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
T0*D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
use_locking(
?
6squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes
:@*D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean
?
Fsqueezenet/fire2/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB@*  ??*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*
_output_shapes
:@
?
5squeezenet/fire2/expand/1x1/BatchNorm/moving_variance
VariableV2*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*
shared_name *
	container *
_output_shapes
:@*
shape:@*
dtype0
?
<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire2/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire2/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes
:@
?
:squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*
_output_shapes
:@*
T0
?
6squeezenet/fire2/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire2/expand/1x1/Conv2D+squeezenet/fire2/expand/1x1/BatchNorm/Const/squeezenet/fire2/expand/1x1/BatchNorm/beta/read6squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/read*
T0*
U0*
is_training( *B
_output_shapes0
.:''@:@:@:@:@:*
epsilon%o?:*
data_formatNHWC
?
 squeezenet/fire2/expand/1x1/ReluRelu6squeezenet/fire2/expand/1x1/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:''@
?
Dsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*%
valueB"         @   
?
Bsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
valueB
 *????*
dtype0*
_output_shapes
: 
?
Bsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/maxConst*
dtype0*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
_output_shapes
: *
valueB
 *???=
?
Lsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/shape*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
T0*
seed2 *
dtype0*&
_output_shapes
:@*

seed 
?
Bsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
_output_shapes
: 
?
Bsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
T0*&
_output_shapes
:@
?
>squeezenet/fire2/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire2/expand/3x3/weights/Initializer/random_uniform/min*
T0*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights
?
#squeezenet/fire2/expand/3x3/weights
VariableV2*&
_output_shapes
:@*
dtype0*
	container *
shape:@*
shared_name *6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights
?
*squeezenet/fire2/expand/3x3/weights/AssignAssign#squeezenet/fire2/expand/3x3/weights>squeezenet/fire2/expand/3x3/weights/Initializer/random_uniform*
T0*
validate_shape(*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*&
_output_shapes
:@*
use_locking(
?
(squeezenet/fire2/expand/3x3/weights/readIdentity#squeezenet/fire2/expand/3x3/weights*
T0*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*&
_output_shapes
:@
z
)squeezenet/fire2/expand/3x3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire2/expand/3x3/Conv2DConv2Dsqueezenet/fire2/squeeze/Relu(squeezenet/fire2/expand/3x3/weights/read*
paddingSAME*&
_output_shapes
:''@*
explicit_paddings
 *
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
T0*
strides

?
<squeezenet/fire2/expand/3x3/BatchNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta*
dtype0*
_output_shapes
:@*
valueB@*    
?
*squeezenet/fire2/expand/3x3/BatchNorm/beta
VariableV2*
	container *
shared_name *
dtype0*
shape:@*
_output_shapes
:@*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta
?
1squeezenet/fire2/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire2/expand/3x3/BatchNorm/beta<squeezenet/fire2/expand/3x3/BatchNorm/beta/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta
?
/squeezenet/fire2/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire2/expand/3x3/BatchNorm/beta*
_output_shapes
:@*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta*
T0
x
+squeezenet/fire2/expand/3x3/BatchNorm/ConstConst*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
Csqueezenet/fire2/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean
?
1squeezenet/fire2/expand/3x3/BatchNorm/moving_mean
VariableV2*
_output_shapes
:@*
shape:@*
shared_name *
	container *
dtype0*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean
?
8squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire2/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire2/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*
use_locking(*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean*
_output_shapes
:@
?
6squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire2/expand/3x3/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean*
T0*
_output_shapes
:@
?
Fsqueezenet/fire2/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:@*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
valueB@*  ??*
dtype0
?
5squeezenet/fire2/expand/3x3/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:@*
	container *
dtype0*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@
?
<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire2/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire2/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
T0*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
use_locking(*
validate_shape(
?
:squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
T0*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance
?
6squeezenet/fire2/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire2/expand/3x3/Conv2D+squeezenet/fire2/expand/3x3/BatchNorm/Const/squeezenet/fire2/expand/3x3/BatchNorm/beta/read6squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
T0*
U0*
epsilon%o?:*B
_output_shapes0
.:''@:@:@:@:@:
?
 squeezenet/fire2/expand/3x3/ReluRelu6squeezenet/fire2/expand/3x3/BatchNorm/FusedBatchNormV3*&
_output_shapes
:''@*
T0
^
squeezenet/fire2/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
?
squeezenet/fire2/concatConcatV2 squeezenet/fire2/expand/1x1/Relu squeezenet/fire2/expand/3x3/Relusqueezenet/fire2/concat/axis*
N*
T0*

Tidx0*'
_output_shapes
:''?
?
Asqueezenet/fire3/squeeze/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
_output_shapes
:*%
valueB"      ?      
?
?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
valueB
 *?Q?*
_output_shapes
: 
?
?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?Q>*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
dtype0
?
Isqueezenet/fire3/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire3/squeeze/weights/Initializer/random_uniform/shape*
seed2 *

seed *'
_output_shapes
:?*
dtype0*
T0*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights
?
?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
T0
?
?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire3/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
T0*'
_output_shapes
:?
?
;squeezenet/fire3/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire3/squeeze/weights/Initializer/random_uniform/min*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*'
_output_shapes
:?*
T0
?
 squeezenet/fire3/squeeze/weights
VariableV2*'
_output_shapes
:?*
shape:?*
shared_name *3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
dtype0*
	container 
?
'squeezenet/fire3/squeeze/weights/AssignAssign squeezenet/fire3/squeeze/weights;squeezenet/fire3/squeeze/weights/Initializer/random_uniform*
validate_shape(*
T0*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*'
_output_shapes
:?*
use_locking(
?
%squeezenet/fire3/squeeze/weights/readIdentity squeezenet/fire3/squeeze/weights*
T0*'
_output_shapes
:?*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights
w
&squeezenet/fire3/squeeze/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
squeezenet/fire3/squeeze/Conv2DConv2Dsqueezenet/fire2/concat%squeezenet/fire3/squeeze/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
explicit_paddings
 *
strides
*
	dilations
*&
_output_shapes
:''*
data_formatNHWC*
T0
?
9squeezenet/fire3/squeeze/BatchNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta*
_output_shapes
:*
valueB*    *
dtype0
?
'squeezenet/fire3/squeeze/BatchNorm/beta
VariableV2*
dtype0*
	container *
_output_shapes
:*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta*
shared_name *
shape:
?
.squeezenet/fire3/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire3/squeeze/BatchNorm/beta9squeezenet/fire3/squeeze/BatchNorm/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta*
validate_shape(
?
,squeezenet/fire3/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire3/squeeze/BatchNorm/beta*
T0*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta*
_output_shapes
:
u
(squeezenet/fire3/squeeze/BatchNorm/ConstConst*
_output_shapes
:*
dtype0*
valueB*  ??
?
@squeezenet/fire3/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean*
_output_shapes
:*
valueB*    *
dtype0
?
.squeezenet/fire3/squeeze/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean*
	container *
dtype0*
_output_shapes
:
?
5squeezenet/fire3/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire3/squeeze/BatchNorm/moving_mean@squeezenet/fire3/squeeze/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean
?
3squeezenet/fire3/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire3/squeeze/BatchNorm/moving_mean*
_output_shapes
:*A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean*
T0
?
Csqueezenet/fire3/squeeze/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:*
valueB*  ??*
dtype0*E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance
?
2squeezenet/fire3/squeeze/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance*
	container *
dtype0*
_output_shapes
:
?
9squeezenet/fire3/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire3/squeeze/BatchNorm/moving_varianceCsqueezenet/fire3/squeeze/BatchNorm/moving_variance/Initializer/ones*
use_locking(*E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance*
_output_shapes
:*
T0*
validate_shape(
?
7squeezenet/fire3/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire3/squeeze/BatchNorm/moving_variance*E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance*
T0*
_output_shapes
:
?
3squeezenet/fire3/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire3/squeeze/Conv2D(squeezenet/fire3/squeeze/BatchNorm/Const,squeezenet/fire3/squeeze/BatchNorm/beta/read3squeezenet/fire3/squeeze/BatchNorm/moving_mean/read7squeezenet/fire3/squeeze/BatchNorm/moving_variance/read*
U0*
is_training( *
T0*
epsilon%o?:*B
_output_shapes0
.:'':::::*
data_formatNHWC
?
squeezenet/fire3/squeeze/ReluRelu3squeezenet/fire3/squeeze/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:''
?
Dsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights
?
Bsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
dtype0*
valueB
 *?7??*
_output_shapes
: 
?
Bsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/maxConst*
valueB
 *?7?>*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
dtype0*
_output_shapes
: 
?
Lsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/shape*

seed *
T0*
seed2 *6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*&
_output_shapes
:@*
dtype0
?
Bsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
_output_shapes
: 
?
Bsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
T0*&
_output_shapes
:@
?
>squeezenet/fire3/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire3/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*&
_output_shapes
:@
?
#squeezenet/fire3/expand/1x1/weights
VariableV2*
shape:@*
	container *6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
dtype0*&
_output_shapes
:@*
shared_name 
?
*squeezenet/fire3/expand/1x1/weights/AssignAssign#squeezenet/fire3/expand/1x1/weights>squeezenet/fire3/expand/1x1/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*&
_output_shapes
:@*
T0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights
?
(squeezenet/fire3/expand/1x1/weights/readIdentity#squeezenet/fire3/expand/1x1/weights*&
_output_shapes
:@*
T0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights
z
)squeezenet/fire3/expand/1x1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire3/expand/1x1/Conv2DConv2Dsqueezenet/fire3/squeeze/Relu(squeezenet/fire3/expand/1x1/weights/read*&
_output_shapes
:''@*
data_formatNHWC*
paddingSAME*
explicit_paddings
 *
use_cudnn_on_gpu(*
strides
*
T0*
	dilations

?
<squeezenet/fire3/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta
?
*squeezenet/fire3/expand/1x1/BatchNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta*
_output_shapes
:@*
	container *
shape:@*
dtype0
?
1squeezenet/fire3/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire3/expand/1x1/BatchNorm/beta<squeezenet/fire3/expand/1x1/BatchNorm/beta/Initializer/zeros*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta
?
/squeezenet/fire3/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire3/expand/1x1/BatchNorm/beta*
T0*=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta*
_output_shapes
:@
x
+squeezenet/fire3/expand/1x1/BatchNorm/ConstConst*
dtype0*
valueB@*  ??*
_output_shapes
:@
?
Csqueezenet/fire3/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean
?
1squeezenet/fire3/expand/1x1/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*
dtype0*
	container *
shape:@*
shared_name *
_output_shapes
:@
?
8squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire3/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire3/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*
use_locking(*
T0
?
6squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*
_output_shapes
:@*
T0
?
Fsqueezenet/fire3/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
valueB@*  ??*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance*
_output_shapes
:@*
dtype0
?
5squeezenet/fire3/expand/1x1/BatchNorm/moving_variance
VariableV2*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance*
dtype0*
	container *
shared_name *
shape:@*
_output_shapes
:@
?
<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire3/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire3/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance
?
:squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire3/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes
:@*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance
?
6squeezenet/fire3/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire3/expand/1x1/Conv2D+squeezenet/fire3/expand/1x1/BatchNorm/Const/squeezenet/fire3/expand/1x1/BatchNorm/beta/read6squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/read*
epsilon%o?:*
data_formatNHWC*B
_output_shapes0
.:''@:@:@:@:@:*
T0*
U0*
is_training( 
?
 squeezenet/fire3/expand/1x1/ReluRelu6squeezenet/fire3/expand/1x1/BatchNorm/FusedBatchNormV3*&
_output_shapes
:''@*
T0
?
Dsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights
?
Bsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
_output_shapes
: *
valueB
 *????*
dtype0
?
Bsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
dtype0*
valueB
 *???=
?
Lsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*&
_output_shapes
:@*
T0*

seed 
?
Bsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
_output_shapes
: *
T0
?
Bsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights
?
>squeezenet/fire3/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire3/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*&
_output_shapes
:@*
T0
?
#squeezenet/fire3/expand/3x3/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
dtype0*&
_output_shapes
:@*
shape:@*
shared_name *
	container 
?
*squeezenet/fire3/expand/3x3/weights/AssignAssign#squeezenet/fire3/expand/3x3/weights>squeezenet/fire3/expand/3x3/weights/Initializer/random_uniform*
T0*
validate_shape(*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
use_locking(
?
(squeezenet/fire3/expand/3x3/weights/readIdentity#squeezenet/fire3/expand/3x3/weights*
T0*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights
z
)squeezenet/fire3/expand/3x3/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
"squeezenet/fire3/expand/3x3/Conv2DConv2Dsqueezenet/fire3/squeeze/Relu(squeezenet/fire3/expand/3x3/weights/read*&
_output_shapes
:''@*
strides
*
	dilations
*
paddingSAME*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
explicit_paddings
 
?
<squeezenet/fire3/expand/3x3/BatchNorm/beta/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta
?
*squeezenet/fire3/expand/3x3/BatchNorm/beta
VariableV2*
_output_shapes
:@*=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta*
shared_name *
dtype0*
shape:@*
	container 
?
1squeezenet/fire3/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire3/expand/3x3/BatchNorm/beta<squeezenet/fire3/expand/3x3/BatchNorm/beta/Initializer/zeros*
_output_shapes
:@*
validate_shape(*
use_locking(*=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta*
T0
?
/squeezenet/fire3/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire3/expand/3x3/BatchNorm/beta*
T0*
_output_shapes
:@*=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta
x
+squeezenet/fire3/expand/3x3/BatchNorm/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*  ??
?
Csqueezenet/fire3/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:@*
dtype0*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
valueB@*    
?
1squeezenet/fire3/expand/3x3/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
	container *
shape:@*
_output_shapes
:@*
dtype0*
shared_name 
?
8squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire3/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire3/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
?
6squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
_output_shapes
:@*
T0
?
Fsqueezenet/fire3/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
valueB@*  ??*
dtype0
?
5squeezenet/fire3/expand/3x3/BatchNorm/moving_variance
VariableV2*
shared_name *
	container *
_output_shapes
:@*
dtype0*
shape:@*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance
?
<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire3/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire3/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance*
use_locking(*
T0
?
:squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire3/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
T0*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance
?
6squeezenet/fire3/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire3/expand/3x3/Conv2D+squeezenet/fire3/expand/3x3/BatchNorm/Const/squeezenet/fire3/expand/3x3/BatchNorm/beta/read6squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/read*
U0*
data_formatNHWC*
T0*B
_output_shapes0
.:''@:@:@:@:@:*
epsilon%o?:*
is_training( 
?
 squeezenet/fire3/expand/3x3/ReluRelu6squeezenet/fire3/expand/3x3/BatchNorm/FusedBatchNormV3*&
_output_shapes
:''@*
T0
^
squeezenet/fire3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
squeezenet/fire3/concatConcatV2 squeezenet/fire3/expand/1x1/Relu squeezenet/fire3/expand/3x3/Relusqueezenet/fire3/concat/axis*
N*

Tidx0*
T0*'
_output_shapes
:''?
?
Asqueezenet/fire4/squeeze/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      ?       *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*
dtype0
?
?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?KF?*
dtype0*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights
?
?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/maxConst*
valueB
 *?KF>*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*
_output_shapes
: *
dtype0
?
Isqueezenet/fire4/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire4/squeeze/weights/Initializer/random_uniform/shape*
seed2 *
T0*'
_output_shapes
:? *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*

seed *
dtype0
?
?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights
?
?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire4/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:? *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights
?
;squeezenet/fire4/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire4/squeeze/weights/Initializer/random_uniform/min*
T0*'
_output_shapes
:? *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights
?
 squeezenet/fire4/squeeze/weights
VariableV2*
	container *'
_output_shapes
:? *
dtype0*
shape:? *3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*
shared_name 
?
'squeezenet/fire4/squeeze/weights/AssignAssign squeezenet/fire4/squeeze/weights;squeezenet/fire4/squeeze/weights/Initializer/random_uniform*'
_output_shapes
:? *
use_locking(*
T0*
validate_shape(*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights
?
%squeezenet/fire4/squeeze/weights/readIdentity squeezenet/fire4/squeeze/weights*
T0*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*'
_output_shapes
:? 
w
&squeezenet/fire4/squeeze/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
squeezenet/fire4/squeeze/Conv2DConv2Dsqueezenet/fire3/concat%squeezenet/fire4/squeeze/weights/read*&
_output_shapes
:'' *
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
T0*
strides
*
explicit_paddings
 *
paddingSAME
?
9squeezenet/fire4/squeeze/BatchNorm/beta/Initializer/zerosConst*
valueB *    *
_output_shapes
: *:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta*
dtype0
?
'squeezenet/fire4/squeeze/BatchNorm/beta
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container *:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta
?
.squeezenet/fire4/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire4/squeeze/BatchNorm/beta9squeezenet/fire4/squeeze/BatchNorm/beta/Initializer/zeros*
T0*
use_locking(*:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
?
,squeezenet/fire4/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire4/squeeze/BatchNorm/beta*
_output_shapes
: *:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta*
T0
u
(squeezenet/fire4/squeeze/BatchNorm/ConstConst*
valueB *  ??*
dtype0*
_output_shapes
: 
?
@squeezenet/fire4/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean*
_output_shapes
: *
dtype0*
valueB *    
?
.squeezenet/fire4/squeeze/BatchNorm/moving_mean
VariableV2*
dtype0*A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean*
shared_name *
	container *
_output_shapes
: *
shape: 
?
5squeezenet/fire4/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire4/squeeze/BatchNorm/moving_mean@squeezenet/fire4/squeeze/BatchNorm/moving_mean/Initializer/zeros*
T0*
_output_shapes
: *A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean*
use_locking(*
validate_shape(
?
3squeezenet/fire4/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire4/squeeze/BatchNorm/moving_mean*A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean*
T0*
_output_shapes
: 
?
Csqueezenet/fire4/squeeze/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
: *E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
dtype0*
valueB *  ??
?
2squeezenet/fire4/squeeze/BatchNorm/moving_variance
VariableV2*
shared_name *
	container *E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
shape: *
dtype0*
_output_shapes
: 
?
9squeezenet/fire4/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire4/squeeze/BatchNorm/moving_varianceCsqueezenet/fire4/squeeze/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
T0*E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
_output_shapes
: 
?
7squeezenet/fire4/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire4/squeeze/BatchNorm/moving_variance*E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
_output_shapes
: *
T0
?
3squeezenet/fire4/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire4/squeeze/Conv2D(squeezenet/fire4/squeeze/BatchNorm/Const,squeezenet/fire4/squeeze/BatchNorm/beta/read3squeezenet/fire4/squeeze/BatchNorm/moving_mean/read7squeezenet/fire4/squeeze/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
U0*
epsilon%o?:*B
_output_shapes0
.:'' : : : : :
?
squeezenet/fire4/squeeze/ReluRelu3squeezenet/fire4/squeeze/BatchNorm/FusedBatchNormV3*&
_output_shapes
:'' *
T0
?
Dsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*%
valueB"          ?   
?
Bsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/minConst*
valueB
 *?KF?*
dtype0*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights
?
Bsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
_output_shapes
: *
dtype0*
valueB
 *?KF>
?
Lsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/shape*

seed *'
_output_shapes
: ?*
T0*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
dtype0*
seed2 
?
Bsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
_output_shapes
: 
?
Bsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
T0*'
_output_shapes
: ?
?
>squeezenet/fire4/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire4/expand/1x1/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
T0*'
_output_shapes
: ?
?
#squeezenet/fire4/expand/1x1/weights
VariableV2*
shape: ?*'
_output_shapes
: ?*
	container *6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
shared_name *
dtype0
?
*squeezenet/fire4/expand/1x1/weights/AssignAssign#squeezenet/fire4/expand/1x1/weights>squeezenet/fire4/expand/1x1/weights/Initializer/random_uniform*
T0*'
_output_shapes
: ?*
validate_shape(*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
use_locking(
?
(squeezenet/fire4/expand/1x1/weights/readIdentity#squeezenet/fire4/expand/1x1/weights*
T0*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*'
_output_shapes
: ?
z
)squeezenet/fire4/expand/1x1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
"squeezenet/fire4/expand/1x1/Conv2DConv2Dsqueezenet/fire4/squeeze/Relu(squeezenet/fire4/expand/1x1/weights/read*
paddingSAME*
T0*'
_output_shapes
:''?*
explicit_paddings
 *
	dilations
*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
<squeezenet/fire4/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta*
dtype0
?
*squeezenet/fire4/expand/1x1/BatchNorm/beta
VariableV2*
dtype0*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta*
shared_name *
	container *
_output_shapes	
:?*
shape:?
?
1squeezenet/fire4/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire4/expand/1x1/BatchNorm/beta<squeezenet/fire4/expand/1x1/BatchNorm/beta/Initializer/zeros*
_output_shapes	
:?*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta*
validate_shape(
?
/squeezenet/fire4/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire4/expand/1x1/BatchNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta
z
+squeezenet/fire4/expand/1x1/BatchNorm/ConstConst*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Csqueezenet/fire4/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*
valueB?*    
?
1squeezenet/fire4/expand/1x1/BatchNorm/moving_mean
VariableV2*
_output_shapes	
:?*
	container *D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*
shared_name *
shape:?*
dtype0
?
8squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire4/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire4/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
6squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?
?
Fsqueezenet/fire4/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:?*
valueB?*  ??
?
5squeezenet/fire4/expand/1x1/BatchNorm/moving_variance
VariableV2*
shared_name *
	container *
shape:?*
_output_shapes	
:?*
dtype0*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance
?
<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire4/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire4/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?
?
:squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire4/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance
?
6squeezenet/fire4/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire4/expand/1x1/Conv2D+squeezenet/fire4/expand/1x1/BatchNorm/Const/squeezenet/fire4/expand/1x1/BatchNorm/beta/read6squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
U0*
epsilon%o?:*G
_output_shapes5
3:''?:?:?:?:?:*
is_training( 
?
 squeezenet/fire4/expand/1x1/ReluRelu6squeezenet/fire4/expand/1x1/BatchNorm/FusedBatchNormV3*'
_output_shapes
:''?*
T0
?
Dsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/shapeConst*%
valueB"          ?   *
dtype0*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
_output_shapes
:
?
Bsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/minConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
valueB
 *?2??*
dtype0
?
Bsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?2?=*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
_output_shapes
: 
?
Lsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/shape*

seed *
dtype0*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*'
_output_shapes
: ?*
T0*
seed2 
?
Bsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
_output_shapes
: 
?
Bsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*'
_output_shapes
: ?*
T0
?
>squeezenet/fire4/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire4/expand/3x3/weights/Initializer/random_uniform/min*'
_output_shapes
: ?*
T0*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights
?
#squeezenet/fire4/expand/3x3/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
	container *
shape: ?*
dtype0*
shared_name *'
_output_shapes
: ?
?
*squeezenet/fire4/expand/3x3/weights/AssignAssign#squeezenet/fire4/expand/3x3/weights>squeezenet/fire4/expand/3x3/weights/Initializer/random_uniform*
T0*
validate_shape(*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*'
_output_shapes
: ?*
use_locking(
?
(squeezenet/fire4/expand/3x3/weights/readIdentity#squeezenet/fire4/expand/3x3/weights*'
_output_shapes
: ?*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
T0
z
)squeezenet/fire4/expand/3x3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
"squeezenet/fire4/expand/3x3/Conv2DConv2Dsqueezenet/fire4/squeeze/Relu(squeezenet/fire4/expand/3x3/weights/read*
T0*
data_formatNHWC*
use_cudnn_on_gpu(*
	dilations
*
strides
*
paddingSAME*
explicit_paddings
 *'
_output_shapes
:''?
?
<squeezenet/fire4/expand/3x3/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta
?
*squeezenet/fire4/expand/3x3/BatchNorm/beta
VariableV2*
dtype0*
	container *
shape:?*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta*
shared_name 
?
1squeezenet/fire4/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire4/expand/3x3/BatchNorm/beta<squeezenet/fire4/expand/3x3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta
?
/squeezenet/fire4/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire4/expand/3x3/BatchNorm/beta*
T0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta
z
+squeezenet/fire4/expand/3x3/BatchNorm/ConstConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
Csqueezenet/fire4/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
valueB?*    
?
1squeezenet/fire4/expand/3x3/BatchNorm/moving_mean
VariableV2*
shape:?*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean*
shared_name *
dtype0*
	container *
_output_shapes	
:?
?
8squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire4/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire4/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean
?
6squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire4/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean
?
Fsqueezenet/fire4/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5squeezenet/fire4/expand/3x3/BatchNorm/moving_variance
VariableV2*
dtype0*
	container *
shared_name *
shape:?*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire4/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire4/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance
?
:squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire4/expand/3x3/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance
?
6squeezenet/fire4/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire4/expand/3x3/Conv2D+squeezenet/fire4/expand/3x3/BatchNorm/Const/squeezenet/fire4/expand/3x3/BatchNorm/beta/read6squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/read*
epsilon%o?:*
is_training( *
data_formatNHWC*
T0*G
_output_shapes5
3:''?:?:?:?:?:*
U0
?
 squeezenet/fire4/expand/3x3/ReluRelu6squeezenet/fire4/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:''?*
T0
^
squeezenet/fire4/concat/axisConst*
value	B :*
_output_shapes
: *
dtype0
?
squeezenet/fire4/concatConcatV2 squeezenet/fire4/expand/1x1/Relu squeezenet/fire4/expand/3x3/Relusqueezenet/fire4/concat/axis*
T0*

Tidx0*
N*'
_output_shapes
:''?
?
squeezenet/maxpool4/MaxPoolMaxPoolsqueezenet/fire4/concat*
paddingVALID*
T0*
strides
*'
_output_shapes
:?*
ksize
*
data_formatNHWC
?
Asqueezenet/fire5/squeeze/weights/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights
?
?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *:??*
dtype0*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights
?
?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/maxConst*
valueB
 *:?>*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
dtype0
?
Isqueezenet/fire5/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire5/squeeze/weights/Initializer/random_uniform/shape*'
_output_shapes
:? *

seed *3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
T0*
seed2 *
dtype0
?
?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/min*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
T0*
_output_shapes
: 
?
?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire5/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
T0*'
_output_shapes
:? 
?
;squeezenet/fire5/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire5/squeeze/weights/Initializer/random_uniform/min*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*'
_output_shapes
:? *
T0
?
 squeezenet/fire5/squeeze/weights
VariableV2*
shared_name *
dtype0*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
shape:? *'
_output_shapes
:? *
	container 
?
'squeezenet/fire5/squeeze/weights/AssignAssign squeezenet/fire5/squeeze/weights;squeezenet/fire5/squeeze/weights/Initializer/random_uniform*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
validate_shape(*
T0*'
_output_shapes
:? *
use_locking(
?
%squeezenet/fire5/squeeze/weights/readIdentity squeezenet/fire5/squeeze/weights*
T0*'
_output_shapes
:? *3
_class)
'%loc:@squeezenet/fire5/squeeze/weights
w
&squeezenet/fire5/squeeze/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
squeezenet/fire5/squeeze/Conv2DConv2Dsqueezenet/maxpool4/MaxPool%squeezenet/fire5/squeeze/weights/read*
strides
*
explicit_paddings
 *
paddingSAME*
T0*
	dilations
*
data_formatNHWC*&
_output_shapes
: *
use_cudnn_on_gpu(
?
9squeezenet/fire5/squeeze/BatchNorm/beta/Initializer/zerosConst*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
dtype0*
valueB *    *
_output_shapes
: 
?
'squeezenet/fire5/squeeze/BatchNorm/beta
VariableV2*
shared_name *
dtype0*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
	container *
shape: *
_output_shapes
: 
?
.squeezenet/fire5/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire5/squeeze/BatchNorm/beta9squeezenet/fire5/squeeze/BatchNorm/beta/Initializer/zeros*
T0*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
validate_shape(*
_output_shapes
: *
use_locking(
?
,squeezenet/fire5/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire5/squeeze/BatchNorm/beta*
T0*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
_output_shapes
: 
u
(squeezenet/fire5/squeeze/BatchNorm/ConstConst*
_output_shapes
: *
dtype0*
valueB *  ??
?
@squeezenet/fire5/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean*
_output_shapes
: *
dtype0*
valueB *    
?
.squeezenet/fire5/squeeze/BatchNorm/moving_mean
VariableV2*
dtype0*
shape: *
_output_shapes
: *A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean*
	container *
shared_name 
?
5squeezenet/fire5/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire5/squeeze/BatchNorm/moving_mean@squeezenet/fire5/squeeze/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
T0*A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean*
_output_shapes
: *
use_locking(
?
3squeezenet/fire5/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire5/squeeze/BatchNorm/moving_mean*
_output_shapes
: *
T0*A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean
?
Csqueezenet/fire5/squeeze/BatchNorm/moving_variance/Initializer/onesConst*E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance*
dtype0*
valueB *  ??*
_output_shapes
: 
?
2squeezenet/fire5/squeeze/BatchNorm/moving_variance
VariableV2*
	container *
shape: *
shared_name *
dtype0*
_output_shapes
: *E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance
?
9squeezenet/fire5/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire5/squeeze/BatchNorm/moving_varianceCsqueezenet/fire5/squeeze/BatchNorm/moving_variance/Initializer/ones*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance
?
7squeezenet/fire5/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire5/squeeze/BatchNorm/moving_variance*E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance*
_output_shapes
: *
T0
?
3squeezenet/fire5/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire5/squeeze/Conv2D(squeezenet/fire5/squeeze/BatchNorm/Const,squeezenet/fire5/squeeze/BatchNorm/beta/read3squeezenet/fire5/squeeze/BatchNorm/moving_mean/read7squeezenet/fire5/squeeze/BatchNorm/moving_variance/read*
U0*
epsilon%o?:*
data_formatNHWC*
is_training( *
T0*B
_output_shapes0
.: : : : : :
?
squeezenet/fire5/squeeze/ReluRelu3squeezenet/fire5/squeeze/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
: 
?
Dsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/shapeConst*
dtype0*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*%
valueB"          ?   *
_output_shapes
:
?
Bsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?KF?*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights
?
Bsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
valueB
 *?KF>*
dtype0
?
Lsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/shape*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
T0*
seed2 *
dtype0*'
_output_shapes
: ?*

seed 
?
Bsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
_output_shapes
: 
?
Bsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*'
_output_shapes
: ?*
T0
?
>squeezenet/fire5/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire5/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*'
_output_shapes
: ?
?
#squeezenet/fire5/expand/1x1/weights
VariableV2*
shared_name *
shape: ?*
dtype0*
	container *6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*'
_output_shapes
: ?
?
*squeezenet/fire5/expand/1x1/weights/AssignAssign#squeezenet/fire5/expand/1x1/weights>squeezenet/fire5/expand/1x1/weights/Initializer/random_uniform*
use_locking(*
T0*'
_output_shapes
: ?*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
validate_shape(
?
(squeezenet/fire5/expand/1x1/weights/readIdentity#squeezenet/fire5/expand/1x1/weights*
T0*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*'
_output_shapes
: ?
z
)squeezenet/fire5/expand/1x1/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
"squeezenet/fire5/expand/1x1/Conv2DConv2Dsqueezenet/fire5/squeeze/Relu(squeezenet/fire5/expand/1x1/weights/read*
T0*
paddingSAME*
	dilations
*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*'
_output_shapes
:?*
explicit_paddings
 
?
<squeezenet/fire5/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
dtype0*=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta*
valueB?*    *
_output_shapes	
:?
?
*squeezenet/fire5/expand/1x1/BatchNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta*
dtype0*
_output_shapes	
:?*
shape:?*
	container 
?
1squeezenet/fire5/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire5/expand/1x1/BatchNorm/beta<squeezenet/fire5/expand/1x1/BatchNorm/beta/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta*
validate_shape(
?
/squeezenet/fire5/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire5/expand/1x1/BatchNorm/beta*
T0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta
z
+squeezenet/fire5/expand/1x1/BatchNorm/ConstConst*
dtype0*
valueB?*  ??*
_output_shapes	
:?
?
Csqueezenet/fire5/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:?*
valueB?*    
?
1squeezenet/fire5/expand/1x1/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
dtype0*
shape:?*
shared_name *
	container *
_output_shapes	
:?
?
8squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire5/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire5/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
T0*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
6squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?
?
Fsqueezenet/fire5/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:?
?
5squeezenet/fire5/expand/1x1/BatchNorm/moving_variance
VariableV2*
_output_shapes	
:?*
shared_name *
	container *
shape:?*
dtype0*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance
?
<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire5/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire5/expand/1x1/BatchNorm/moving_variance/Initializer/ones*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
:squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire5/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance
?
6squeezenet/fire5/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire5/expand/1x1/Conv2D+squeezenet/fire5/expand/1x1/BatchNorm/Const/squeezenet/fire5/expand/1x1/BatchNorm/beta/read6squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/read*
data_formatNHWC*
U0*G
_output_shapes5
3:?:?:?:?:?:*
T0*
is_training( *
epsilon%o?:
?
 squeezenet/fire5/expand/1x1/ReluRelu6squeezenet/fire5/expand/1x1/BatchNorm/FusedBatchNormV3*
T0*'
_output_shapes
:?
?
Dsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"          ?   *
_output_shapes
:*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights
?
Bsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/minConst*
dtype0*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
_output_shapes
: *
valueB
 *?2??
?
Bsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?2?=*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
_output_shapes
: 
?
Lsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/shape*
T0*
seed2 *'
_output_shapes
: ?*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
dtype0*

seed 
?
Bsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
_output_shapes
: *
T0
?
Bsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*'
_output_shapes
: ?*
T0
?
>squeezenet/fire5/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire5/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*'
_output_shapes
: ?*
T0
?
#squeezenet/fire5/expand/3x3/weights
VariableV2*
shape: ?*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*'
_output_shapes
: ?*
shared_name *
dtype0*
	container 
?
*squeezenet/fire5/expand/3x3/weights/AssignAssign#squeezenet/fire5/expand/3x3/weights>squeezenet/fire5/expand/3x3/weights/Initializer/random_uniform*'
_output_shapes
: ?*
use_locking(*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
T0*
validate_shape(
?
(squeezenet/fire5/expand/3x3/weights/readIdentity#squeezenet/fire5/expand/3x3/weights*
T0*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*'
_output_shapes
: ?
z
)squeezenet/fire5/expand/3x3/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
"squeezenet/fire5/expand/3x3/Conv2DConv2Dsqueezenet/fire5/squeeze/Relu(squeezenet/fire5/expand/3x3/weights/read*
explicit_paddings
 *
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*'
_output_shapes
:?*
	dilations
*
paddingSAME
?
<squeezenet/fire5/expand/3x3/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta
?
*squeezenet/fire5/expand/3x3/BatchNorm/beta
VariableV2*
shape:?*=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
dtype0*
	container *
shared_name 
?
1squeezenet/fire5/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire5/expand/3x3/BatchNorm/beta<squeezenet/fire5/expand/3x3/BatchNorm/beta/Initializer/zeros*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta*
validate_shape(*
T0*
use_locking(
?
/squeezenet/fire5/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire5/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta
z
+squeezenet/fire5/expand/3x3/BatchNorm/ConstConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
Csqueezenet/fire5/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
valueB?*    *
dtype0
?
1squeezenet/fire5/expand/3x3/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
dtype0*
shared_name *
	container *
_output_shapes	
:?*
shape:?
?
8squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire5/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire5/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*
use_locking(*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?
?
6squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean
?
Fsqueezenet/fire5/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance*
dtype0
?
5squeezenet/fire5/expand/3x3/BatchNorm/moving_variance
VariableV2*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance*
dtype0*
	container *
shared_name *
shape:?*
_output_shapes	
:?
?
<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire5/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire5/expand/3x3/BatchNorm/moving_variance/Initializer/ones*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
:squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire5/expand/3x3/BatchNorm/moving_variance*
T0*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
6squeezenet/fire5/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire5/expand/3x3/Conv2D+squeezenet/fire5/expand/3x3/BatchNorm/Const/squeezenet/fire5/expand/3x3/BatchNorm/beta/read6squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/read*
data_formatNHWC*
T0*
epsilon%o?:*G
_output_shapes5
3:?:?:?:?:?:*
is_training( *
U0
?
 squeezenet/fire5/expand/3x3/ReluRelu6squeezenet/fire5/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:?*
T0
^
squeezenet/fire5/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
?
squeezenet/fire5/concatConcatV2 squeezenet/fire5/expand/1x1/Relu squeezenet/fire5/expand/3x3/Relusqueezenet/fire5/concat/axis*

Tidx0*'
_output_shapes
:?*
N*
T0
?
Asqueezenet/fire6/squeeze/weights/Initializer/random_uniform/shapeConst*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
_output_shapes
:*
dtype0*%
valueB"         0   
?
?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *??*
dtype0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights
?
?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/maxConst*
valueB
 *?>*
_output_shapes
: *
dtype0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights
?
Isqueezenet/fire6/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire6/squeeze/weights/Initializer/random_uniform/shape*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
dtype0*'
_output_shapes
:?0*

seed *
seed2 *
T0
?
?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/min*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
T0
?
?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire6/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*'
_output_shapes
:?0
?
;squeezenet/fire6/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire6/squeeze/weights/Initializer/random_uniform/min*'
_output_shapes
:?0*
T0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights
?
 squeezenet/fire6/squeeze/weights
VariableV2*
	container *
dtype0*'
_output_shapes
:?0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
shared_name *
shape:?0
?
'squeezenet/fire6/squeeze/weights/AssignAssign squeezenet/fire6/squeeze/weights;squeezenet/fire6/squeeze/weights/Initializer/random_uniform*
T0*
validate_shape(*'
_output_shapes
:?0*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
use_locking(
?
%squeezenet/fire6/squeeze/weights/readIdentity squeezenet/fire6/squeeze/weights*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
T0*'
_output_shapes
:?0
w
&squeezenet/fire6/squeeze/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
squeezenet/fire6/squeeze/Conv2DConv2Dsqueezenet/fire5/concat%squeezenet/fire6/squeeze/weights/read*
T0*&
_output_shapes
:0*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
strides
*
	dilations
*
data_formatNHWC
?
9squeezenet/fire6/squeeze/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:0*
valueB0*    *:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta
?
'squeezenet/fire6/squeeze/BatchNorm/beta
VariableV2*
shared_name *
	container *
shape:0*:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta*
dtype0*
_output_shapes
:0
?
.squeezenet/fire6/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire6/squeeze/BatchNorm/beta9squeezenet/fire6/squeeze/BatchNorm/beta/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:0*:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta*
T0
?
,squeezenet/fire6/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire6/squeeze/BatchNorm/beta*
_output_shapes
:0*:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta*
T0
u
(squeezenet/fire6/squeeze/BatchNorm/ConstConst*
_output_shapes
:0*
valueB0*  ??*
dtype0
?
@squeezenet/fire6/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*
valueB0*    *
_output_shapes
:0*
dtype0*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean
?
.squeezenet/fire6/squeeze/BatchNorm/moving_mean
VariableV2*
_output_shapes
:0*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean*
shape:0*
dtype0*
	container *
shared_name 
?
5squeezenet/fire6/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire6/squeeze/BatchNorm/moving_mean@squeezenet/fire6/squeeze/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:0*
use_locking(*
validate_shape(*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean*
T0
?
3squeezenet/fire6/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire6/squeeze/BatchNorm/moving_mean*
_output_shapes
:0*
T0*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean
?
Csqueezenet/fire6/squeeze/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:0*
valueB0*  ??*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance*
dtype0
?
2squeezenet/fire6/squeeze/BatchNorm/moving_variance
VariableV2*
dtype0*
	container *
_output_shapes
:0*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance*
shared_name *
shape:0
?
9squeezenet/fire6/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire6/squeeze/BatchNorm/moving_varianceCsqueezenet/fire6/squeeze/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
:0*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance*
use_locking(*
validate_shape(*
T0
?
7squeezenet/fire6/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire6/squeeze/BatchNorm/moving_variance*
_output_shapes
:0*
T0*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance
?
3squeezenet/fire6/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire6/squeeze/Conv2D(squeezenet/fire6/squeeze/BatchNorm/Const,squeezenet/fire6/squeeze/BatchNorm/beta/read3squeezenet/fire6/squeeze/BatchNorm/moving_mean/read7squeezenet/fire6/squeeze/BatchNorm/moving_variance/read*
U0*
is_training( *
data_formatNHWC*
epsilon%o?:*
T0*B
_output_shapes0
.:0:0:0:0:0:
?
squeezenet/fire6/squeeze/ReluRelu3squeezenet/fire6/squeeze/BatchNorm/FusedBatchNormV3*&
_output_shapes
:0*
T0
?
Dsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      0   ?   *
dtype0*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights
?
Bsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
dtype0*
valueB
 *??!?*
_output_shapes
: 
?
Bsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
valueB
 *??!>*
dtype0
?
Lsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*

seed *
T0
?
Bsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/min*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
T0
?
Bsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/sub*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
T0
?
>squeezenet/fire6/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire6/expand/1x1/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*'
_output_shapes
:0?*
T0
?
#squeezenet/fire6/expand/1x1/weights
VariableV2*
shared_name *
shape:0?*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
dtype0*
	container *'
_output_shapes
:0?
?
*squeezenet/fire6/expand/1x1/weights/AssignAssign#squeezenet/fire6/expand/1x1/weights>squeezenet/fire6/expand/1x1/weights/Initializer/random_uniform*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
T0*
use_locking(*
validate_shape(
?
(squeezenet/fire6/expand/1x1/weights/readIdentity#squeezenet/fire6/expand/1x1/weights*'
_output_shapes
:0?*
T0*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights
z
)squeezenet/fire6/expand/1x1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire6/expand/1x1/Conv2DConv2Dsqueezenet/fire6/squeeze/Relu(squeezenet/fire6/expand/1x1/weights/read*
use_cudnn_on_gpu(*
T0*
	dilations
*'
_output_shapes
:?*
paddingSAME*
data_formatNHWC*
strides
*
explicit_paddings
 
?
<squeezenet/fire6/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
valueB?*    *=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta*
_output_shapes	
:?
?
*squeezenet/fire6/expand/1x1/BatchNorm/beta
VariableV2*
dtype0*
	container *
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta*
shape:?
?
1squeezenet/fire6/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire6/expand/1x1/BatchNorm/beta<squeezenet/fire6/expand/1x1/BatchNorm/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta*
T0*
validate_shape(
?
/squeezenet/fire6/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire6/expand/1x1/BatchNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta
z
+squeezenet/fire6/expand/1x1/BatchNorm/ConstConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??
?
Csqueezenet/fire6/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
dtype0
?
1squeezenet/fire6/expand/1x1/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
	container *D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
shape:?*
_output_shapes	
:?
?
8squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire6/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire6/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
validate_shape(*D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?
?
6squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean
?
Fsqueezenet/fire6/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
dtype0*
valueB?*  ??*
_output_shapes	
:?
?
5squeezenet/fire6/expand/1x1/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
shape:?*
	container *
_output_shapes	
:?
?
<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire6/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire6/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
T0
?
:squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0
?
6squeezenet/fire6/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire6/expand/1x1/Conv2D+squeezenet/fire6/expand/1x1/BatchNorm/Const/squeezenet/fire6/expand/1x1/BatchNorm/beta/read6squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/read*
U0*
data_formatNHWC*G
_output_shapes5
3:?:?:?:?:?:*
is_training( *
epsilon%o?:*
T0
?
 squeezenet/fire6/expand/1x1/ReluRelu6squeezenet/fire6/expand/1x1/BatchNorm/FusedBatchNormV3*
T0*'
_output_shapes
:?
?
Dsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/shapeConst*
dtype0*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*%
valueB"      0   ?   *
_output_shapes
:
?
Bsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/minConst*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*
valueB
 *??W?*
dtype0*
_output_shapes
: 
?
Bsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *??W=*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights
?
Lsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/shape*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*
T0*'
_output_shapes
:0?*
dtype0*
seed2 *

seed 
?
Bsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/min*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*
T0
?
Bsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*'
_output_shapes
:0?*
T0
?
>squeezenet/fire6/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire6/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*'
_output_shapes
:0?*
T0
?
#squeezenet/fire6/expand/3x3/weights
VariableV2*'
_output_shapes
:0?*
shape:0?*
dtype0*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*
shared_name *
	container 
?
*squeezenet/fire6/expand/3x3/weights/AssignAssign#squeezenet/fire6/expand/3x3/weights>squeezenet/fire6/expand/3x3/weights/Initializer/random_uniform*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights
?
(squeezenet/fire6/expand/3x3/weights/readIdentity#squeezenet/fire6/expand/3x3/weights*
T0*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights
z
)squeezenet/fire6/expand/3x3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire6/expand/3x3/Conv2DConv2Dsqueezenet/fire6/squeeze/Relu(squeezenet/fire6/expand/3x3/weights/read*
data_formatNHWC*
paddingSAME*
	dilations
*
strides
*
T0*
explicit_paddings
 *'
_output_shapes
:?*
use_cudnn_on_gpu(
?
<squeezenet/fire6/expand/3x3/BatchNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
valueB?*    *
dtype0
?
*squeezenet/fire6/expand/3x3/BatchNorm/beta
VariableV2*
	container *
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
shape:?
?
1squeezenet/fire6/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire6/expand/3x3/BatchNorm/beta<squeezenet/fire6/expand/3x3/BatchNorm/beta/Initializer/zeros*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/squeezenet/fire6/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire6/expand/3x3/BatchNorm/beta*
T0*=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
_output_shapes	
:?
z
+squeezenet/fire6/expand/3x3/BatchNorm/ConstConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
Csqueezenet/fire6/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean*
valueB?*    
?
1squeezenet/fire6/expand/3x3/BatchNorm/moving_mean
VariableV2*
shared_name *
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean*
dtype0*
shape:?*
	container 
?
8squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire6/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire6/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean
?
6squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire6/expand/3x3/BatchNorm/moving_mean*
T0*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?
?
Fsqueezenet/fire6/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance
?
5squeezenet/fire6/expand/3x3/BatchNorm/moving_variance
VariableV2*
shared_name *
dtype0*
	container *H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?*
shape:?
?
<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire6/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire6/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?*
T0
?
:squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
T0
?
6squeezenet/fire6/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire6/expand/3x3/Conv2D+squeezenet/fire6/expand/3x3/BatchNorm/Const/squeezenet/fire6/expand/3x3/BatchNorm/beta/read6squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/read*
U0*
epsilon%o?:*G
_output_shapes5
3:?:?:?:?:?:*
T0*
data_formatNHWC*
is_training( 
?
 squeezenet/fire6/expand/3x3/ReluRelu6squeezenet/fire6/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:?*
T0
^
squeezenet/fire6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
?
squeezenet/fire6/concatConcatV2 squeezenet/fire6/expand/1x1/Relu squeezenet/fire6/expand/3x3/Relusqueezenet/fire6/concat/axis*

Tidx0*'
_output_shapes
:?*
T0*
N
?
Asqueezenet/fire7/squeeze/weights/Initializer/random_uniform/shapeConst*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*%
valueB"      ?  0   *
dtype0*
_output_shapes
:
?
?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/minConst*
valueB
 *?[??*
_output_shapes
: *
dtype0*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights
?
?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/maxConst*
valueB
 *?[?=*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
dtype0*
_output_shapes
: 
?
Isqueezenet/fire7/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire7/squeeze/weights/Initializer/random_uniform/shape*

seed *'
_output_shapes
:?0*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
T0*
seed2 *
dtype0
?
?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/min*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
T0*
_output_shapes
: 
?
?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire7/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/sub*'
_output_shapes
:?0*
T0*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights
?
;squeezenet/fire7/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire7/squeeze/weights/Initializer/random_uniform/min*
T0*'
_output_shapes
:?0*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights
?
 squeezenet/fire7/squeeze/weights
VariableV2*
	container *3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
shape:?0*'
_output_shapes
:?0*
dtype0*
shared_name 
?
'squeezenet/fire7/squeeze/weights/AssignAssign squeezenet/fire7/squeeze/weights;squeezenet/fire7/squeeze/weights/Initializer/random_uniform*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
use_locking(*'
_output_shapes
:?0*
validate_shape(*
T0
?
%squeezenet/fire7/squeeze/weights/readIdentity squeezenet/fire7/squeeze/weights*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
T0*'
_output_shapes
:?0
w
&squeezenet/fire7/squeeze/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
squeezenet/fire7/squeeze/Conv2DConv2Dsqueezenet/fire6/concat%squeezenet/fire7/squeeze/weights/read*
data_formatNHWC*
	dilations
*
strides
*
explicit_paddings
 *
paddingSAME*
T0*&
_output_shapes
:0*
use_cudnn_on_gpu(
?
9squeezenet/fire7/squeeze/BatchNorm/beta/Initializer/zerosConst*
valueB0*    *
_output_shapes
:0*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta*
dtype0
?
'squeezenet/fire7/squeeze/BatchNorm/beta
VariableV2*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta*
shape:0*
_output_shapes
:0*
shared_name *
	container *
dtype0
?
.squeezenet/fire7/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire7/squeeze/BatchNorm/beta9squeezenet/fire7/squeeze/BatchNorm/beta/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:0*
T0*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta
?
,squeezenet/fire7/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire7/squeeze/BatchNorm/beta*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta*
_output_shapes
:0*
T0
u
(squeezenet/fire7/squeeze/BatchNorm/ConstConst*
dtype0*
valueB0*  ??*
_output_shapes
:0
?
@squeezenet/fire7/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:0*
dtype0*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
valueB0*    
?
.squeezenet/fire7/squeeze/BatchNorm/moving_mean
VariableV2*
_output_shapes
:0*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
dtype0*
	container *
shared_name *
shape:0
?
5squeezenet/fire7/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire7/squeeze/BatchNorm/moving_mean@squeezenet/fire7/squeeze/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:0*
validate_shape(*
use_locking(*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
T0
?
3squeezenet/fire7/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire7/squeeze/BatchNorm/moving_mean*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
T0*
_output_shapes
:0
?
Csqueezenet/fire7/squeeze/BatchNorm/moving_variance/Initializer/onesConst*
valueB0*  ??*
dtype0*
_output_shapes
:0*E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance
?
2squeezenet/fire7/squeeze/BatchNorm/moving_variance
VariableV2*
_output_shapes
:0*
	container *E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance*
shared_name *
dtype0*
shape:0
?
9squeezenet/fire7/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire7/squeeze/BatchNorm/moving_varianceCsqueezenet/fire7/squeeze/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
_output_shapes
:0*
validate_shape(*E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance
?
7squeezenet/fire7/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire7/squeeze/BatchNorm/moving_variance*
T0*E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance*
_output_shapes
:0
?
3squeezenet/fire7/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire7/squeeze/Conv2D(squeezenet/fire7/squeeze/BatchNorm/Const,squeezenet/fire7/squeeze/BatchNorm/beta/read3squeezenet/fire7/squeeze/BatchNorm/moving_mean/read7squeezenet/fire7/squeeze/BatchNorm/moving_variance/read*
T0*
U0*
epsilon%o?:*
is_training( *B
_output_shapes0
.:0:0:0:0:0:*
data_formatNHWC
?
squeezenet/fire7/squeeze/ReluRelu3squeezenet/fire7/squeeze/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:0
?
Dsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      0   ?   *6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
dtype0
?
Bsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *??!?*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights
?
Bsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/maxConst*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
dtype0*
valueB
 *??!>*
_output_shapes
: 
?
Lsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/shape*
dtype0*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
T0*
seed2 *'
_output_shapes
:0?*

seed 
?
Bsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
T0*
_output_shapes
: 
?
Bsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*'
_output_shapes
:0?
?
>squeezenet/fire7/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire7/expand/1x1/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*'
_output_shapes
:0?
?
#squeezenet/fire7/expand/1x1/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
shared_name *
	container *
shape:0?*
dtype0*'
_output_shapes
:0?
?
*squeezenet/fire7/expand/1x1/weights/AssignAssign#squeezenet/fire7/expand/1x1/weights>squeezenet/fire7/expand/1x1/weights/Initializer/random_uniform*'
_output_shapes
:0?*
use_locking(*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
T0*
validate_shape(
?
(squeezenet/fire7/expand/1x1/weights/readIdentity#squeezenet/fire7/expand/1x1/weights*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*'
_output_shapes
:0?*
T0
z
)squeezenet/fire7/expand/1x1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
"squeezenet/fire7/expand/1x1/Conv2DConv2Dsqueezenet/fire7/squeeze/Relu(squeezenet/fire7/expand/1x1/weights/read*
T0*'
_output_shapes
:?*
explicit_paddings
 *
	dilations
*
paddingSAME*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC
?
<squeezenet/fire7/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta
?
*squeezenet/fire7/expand/1x1/BatchNorm/beta
VariableV2*
_output_shapes	
:?*
	container *
shared_name *
shape:?*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta*
dtype0
?
1squeezenet/fire7/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire7/expand/1x1/BatchNorm/beta<squeezenet/fire7/expand/1x1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta*
_output_shapes	
:?
?
/squeezenet/fire7/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire7/expand/1x1/BatchNorm/beta*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta*
T0*
_output_shapes	
:?
z
+squeezenet/fire7/expand/1x1/BatchNorm/ConstConst*
dtype0*
valueB?*  ??*
_output_shapes	
:?
?
Csqueezenet/fire7/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
1squeezenet/fire7/expand/1x1/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:?*
shape:?*
shared_name *D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*
dtype0
?
8squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire7/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire7/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
validate_shape(*D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0
?
6squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?
?
Fsqueezenet/fire7/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance
?
5squeezenet/fire7/expand/1x1/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:?*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance*
dtype0*
	container 
?
<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire7/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire7/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance*
T0*
validate_shape(
?
:squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire7/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance
?
6squeezenet/fire7/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire7/expand/1x1/Conv2D+squeezenet/fire7/expand/1x1/BatchNorm/Const/squeezenet/fire7/expand/1x1/BatchNorm/beta/read6squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/read*
epsilon%o?:*
data_formatNHWC*
is_training( *
T0*
U0*G
_output_shapes5
3:?:?:?:?:?:
?
 squeezenet/fire7/expand/1x1/ReluRelu6squeezenet/fire7/expand/1x1/BatchNorm/FusedBatchNormV3*'
_output_shapes
:?*
T0
?
Dsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      0   ?   *6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights
?
Bsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/minConst*
valueB
 *??W?*
dtype0*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
_output_shapes
: 
?
Bsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/maxConst*
valueB
 *??W=*
dtype0*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
_output_shapes
: 
?
Lsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights
?
Bsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
_output_shapes
: 
?
Bsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/sub*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*'
_output_shapes
:0?*
T0
?
>squeezenet/fire7/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire7/expand/3x3/weights/Initializer/random_uniform/min*'
_output_shapes
:0?*
T0*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights
?
#squeezenet/fire7/expand/3x3/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
dtype0*'
_output_shapes
:0?*
shape:0?*
	container *
shared_name 
?
*squeezenet/fire7/expand/3x3/weights/AssignAssign#squeezenet/fire7/expand/3x3/weights>squeezenet/fire7/expand/3x3/weights/Initializer/random_uniform*
T0*
use_locking(*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
validate_shape(
?
(squeezenet/fire7/expand/3x3/weights/readIdentity#squeezenet/fire7/expand/3x3/weights*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
T0*'
_output_shapes
:0?
z
)squeezenet/fire7/expand/3x3/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
"squeezenet/fire7/expand/3x3/Conv2DConv2Dsqueezenet/fire7/squeeze/Relu(squeezenet/fire7/expand/3x3/weights/read*
strides
*'
_output_shapes
:?*
explicit_paddings
 *
T0*
use_cudnn_on_gpu(*
	dilations
*
paddingSAME*
data_formatNHWC
?
<squeezenet/fire7/expand/3x3/BatchNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
valueB?*    *
dtype0
?
*squeezenet/fire7/expand/3x3/BatchNorm/beta
VariableV2*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta*
dtype0*
shape:?*
_output_shapes	
:?*
shared_name *
	container 
?
1squeezenet/fire7/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire7/expand/3x3/BatchNorm/beta<squeezenet/fire7/expand/3x3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
T0*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta*
use_locking(*
_output_shapes	
:?
?
/squeezenet/fire7/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire7/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta*
T0
z
+squeezenet/fire7/expand/3x3/BatchNorm/ConstConst*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Csqueezenet/fire7/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
valueB?*    *
dtype0
?
1squeezenet/fire7/expand/3x3/BatchNorm/moving_mean
VariableV2*
shared_name *
	container *
shape:?*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
dtype0
?
8squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire7/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire7/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes	
:?*
T0*
use_locking(*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
validate_shape(
?
6squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean
?
Fsqueezenet/fire7/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
valueB?*  ??*
dtype0
?
5squeezenet/fire7/expand/3x3/BatchNorm/moving_variance
VariableV2*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
dtype0*
	container *
shape:?
?
<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire7/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire7/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
validate_shape(
?
:squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
T0*H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
6squeezenet/fire7/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire7/expand/3x3/Conv2D+squeezenet/fire7/expand/3x3/BatchNorm/Const/squeezenet/fire7/expand/3x3/BatchNorm/beta/read6squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/read*
epsilon%o?:*
T0*
data_formatNHWC*
U0*G
_output_shapes5
3:?:?:?:?:?:*
is_training( 
?
 squeezenet/fire7/expand/3x3/ReluRelu6squeezenet/fire7/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:?*
T0
^
squeezenet/fire7/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
?
squeezenet/fire7/concatConcatV2 squeezenet/fire7/expand/1x1/Relu squeezenet/fire7/expand/3x3/Relusqueezenet/fire7/concat/axis*
N*
T0*

Tidx0*'
_output_shapes
:?
?
Asqueezenet/fire8/squeeze/weights/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
_output_shapes
:*%
valueB"      ?  @   
?
?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/minConst*
valueB
 *???*
dtype0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
_output_shapes
: 
?
?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *??=*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights
?
Isqueezenet/fire8/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire8/squeeze/weights/Initializer/random_uniform/shape*
T0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
dtype0*'
_output_shapes
:?@*

seed *
seed2 
?
?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
_output_shapes
: 
?
?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire8/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*'
_output_shapes
:?@*
T0
?
;squeezenet/fire8/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire8/squeeze/weights/Initializer/random_uniform/min*'
_output_shapes
:?@*
T0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights
?
 squeezenet/fire8/squeeze/weights
VariableV2*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
	container *
shape:?@*
dtype0*
shared_name *'
_output_shapes
:?@
?
'squeezenet/fire8/squeeze/weights/AssignAssign squeezenet/fire8/squeeze/weights;squeezenet/fire8/squeeze/weights/Initializer/random_uniform*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
validate_shape(*
use_locking(*
T0*'
_output_shapes
:?@
?
%squeezenet/fire8/squeeze/weights/readIdentity squeezenet/fire8/squeeze/weights*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*'
_output_shapes
:?@*
T0
w
&squeezenet/fire8/squeeze/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
squeezenet/fire8/squeeze/Conv2DConv2Dsqueezenet/fire7/concat%squeezenet/fire8/squeeze/weights/read*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
T0*
paddingSAME*
explicit_paddings
 *
strides
*&
_output_shapes
:@
?
9squeezenet/fire8/squeeze/BatchNorm/beta/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta
?
'squeezenet/fire8/squeeze/BatchNorm/beta
VariableV2*
shape:@*
shared_name *
	container *
_output_shapes
:@*:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta*
dtype0
?
.squeezenet/fire8/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire8/squeeze/BatchNorm/beta9squeezenet/fire8/squeeze/BatchNorm/beta/Initializer/zeros*:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
?
,squeezenet/fire8/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire8/squeeze/BatchNorm/beta*
_output_shapes
:@*
T0*:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta
u
(squeezenet/fire8/squeeze/BatchNorm/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*  ??
?
@squeezenet/fire8/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB@*    *A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean*
_output_shapes
:@
?
.squeezenet/fire8/squeeze/BatchNorm/moving_mean
VariableV2*A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean*
dtype0*
shared_name *
	container *
shape:@*
_output_shapes
:@
?
5squeezenet/fire8/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire8/squeeze/BatchNorm/moving_mean@squeezenet/fire8/squeeze/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:@*
T0*
use_locking(*A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean*
validate_shape(
?
3squeezenet/fire8/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire8/squeeze/BatchNorm/moving_mean*
_output_shapes
:@*A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean*
T0
?
Csqueezenet/fire8/squeeze/BatchNorm/moving_variance/Initializer/onesConst*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
2squeezenet/fire8/squeeze/BatchNorm/moving_variance
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:@*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance*
shape:@
?
9squeezenet/fire8/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire8/squeeze/BatchNorm/moving_varianceCsqueezenet/fire8/squeeze/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
:@*
T0*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance*
validate_shape(*
use_locking(
?
7squeezenet/fire8/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire8/squeeze/BatchNorm/moving_variance*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance*
T0*
_output_shapes
:@
?
3squeezenet/fire8/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire8/squeeze/Conv2D(squeezenet/fire8/squeeze/BatchNorm/Const,squeezenet/fire8/squeeze/BatchNorm/beta/read3squeezenet/fire8/squeeze/BatchNorm/moving_mean/read7squeezenet/fire8/squeeze/BatchNorm/moving_variance/read*
epsilon%o?:*B
_output_shapes0
.:@:@:@:@:@:*
U0*
T0*
is_training( *
data_formatNHWC
?
squeezenet/fire8/squeeze/ReluRelu3squeezenet/fire8/squeeze/BatchNorm/FusedBatchNormV3*&
_output_shapes
:@*
T0
?
Dsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/shapeConst*%
valueB"      @      *
dtype0*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
_output_shapes
:
?
Bsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/minConst*
valueB
 *?7?*
dtype0*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
_output_shapes
: 
?
Bsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
dtype0*
valueB
 *?7>
?
Lsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/shape*
seed2 *'
_output_shapes
:@?*

seed *
dtype0*
T0*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights
?
Bsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/min*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
T0
?
Bsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*'
_output_shapes
:@?
?
>squeezenet/fire8/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire8/expand/1x1/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
T0*'
_output_shapes
:@?
?
#squeezenet/fire8/expand/1x1/weights
VariableV2*
	container *6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
shape:@?*'
_output_shapes
:@?*
shared_name *
dtype0
?
*squeezenet/fire8/expand/1x1/weights/AssignAssign#squeezenet/fire8/expand/1x1/weights>squeezenet/fire8/expand/1x1/weights/Initializer/random_uniform*
T0*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
use_locking(*
validate_shape(
?
(squeezenet/fire8/expand/1x1/weights/readIdentity#squeezenet/fire8/expand/1x1/weights*'
_output_shapes
:@?*
T0*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights
z
)squeezenet/fire8/expand/1x1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
"squeezenet/fire8/expand/1x1/Conv2DConv2Dsqueezenet/fire8/squeeze/Relu(squeezenet/fire8/expand/1x1/weights/read*
use_cudnn_on_gpu(*
strides
*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*'
_output_shapes
:?*
explicit_paddings
 
?
<squeezenet/fire8/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta
?
*squeezenet/fire8/expand/1x1/BatchNorm/beta
VariableV2*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta*
dtype0*
shape:?*
	container *
shared_name *
_output_shapes	
:?
?
1squeezenet/fire8/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire8/expand/1x1/BatchNorm/beta<squeezenet/fire8/expand/1x1/BatchNorm/beta/Initializer/zeros*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta*
_output_shapes	
:?*
validate_shape(
?
/squeezenet/fire8/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire8/expand/1x1/BatchNorm/beta*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta*
T0*
_output_shapes	
:?
z
+squeezenet/fire8/expand/1x1/BatchNorm/ConstConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
Csqueezenet/fire8/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*
dtype0*
valueB?*    *
_output_shapes	
:?
?
1squeezenet/fire8/expand/1x1/BatchNorm/moving_mean
VariableV2*
shared_name *
	container *
_output_shapes	
:?*
dtype0*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*
shape:?
?
8squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire8/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire8/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean
?
6squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*
T0*
_output_shapes	
:?
?
Fsqueezenet/fire8/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance
?
5squeezenet/fire8/expand/1x1/BatchNorm/moving_variance
VariableV2*
shared_name *
dtype0*
	container *
shape:?*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?
?
<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire8/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire8/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
T0*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
:squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes	
:?
?
6squeezenet/fire8/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire8/expand/1x1/Conv2D+squeezenet/fire8/expand/1x1/BatchNorm/Const/squeezenet/fire8/expand/1x1/BatchNorm/beta/read6squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/read*
epsilon%o?:*
is_training( *
T0*
U0*
data_formatNHWC*G
_output_shapes5
3:?:?:?:?:?:
?
 squeezenet/fire8/expand/1x1/ReluRelu6squeezenet/fire8/expand/1x1/BatchNorm/FusedBatchNormV3*
T0*'
_output_shapes
:?
?
Dsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*%
valueB"      @      *
_output_shapes
:*
dtype0
?
Bsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*
valueB
 *??:?
?
Bsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/maxConst*
valueB
 *??:=*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*
dtype0*
_output_shapes
: 
?
Lsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/shape*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*
T0*
dtype0*

seed *
seed2 
?
Bsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights
?
Bsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*'
_output_shapes
:@?
?
>squeezenet/fire8/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire8/expand/3x3/weights/Initializer/random_uniform/min*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*'
_output_shapes
:@?*
T0
?
#squeezenet/fire8/expand/3x3/weights
VariableV2*
shared_name *6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*'
_output_shapes
:@?*
	container *
shape:@?*
dtype0
?
*squeezenet/fire8/expand/3x3/weights/AssignAssign#squeezenet/fire8/expand/3x3/weights>squeezenet/fire8/expand/3x3/weights/Initializer/random_uniform*'
_output_shapes
:@?*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights
?
(squeezenet/fire8/expand/3x3/weights/readIdentity#squeezenet/fire8/expand/3x3/weights*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*
T0*'
_output_shapes
:@?
z
)squeezenet/fire8/expand/3x3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
"squeezenet/fire8/expand/3x3/Conv2DConv2Dsqueezenet/fire8/squeeze/Relu(squeezenet/fire8/expand/3x3/weights/read*
explicit_paddings
 *
strides
*'
_output_shapes
:?*
paddingSAME*
use_cudnn_on_gpu(*
T0*
	dilations
*
data_formatNHWC
?
<squeezenet/fire8/expand/3x3/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta*
dtype0
?
*squeezenet/fire8/expand/3x3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta*
shape:?*
shared_name *
	container 
?
1squeezenet/fire8/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire8/expand/3x3/BatchNorm/beta<squeezenet/fire8/expand/3x3/BatchNorm/beta/Initializer/zeros*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
/squeezenet/fire8/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire8/expand/3x3/BatchNorm/beta*
T0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta
z
+squeezenet/fire8/expand/3x3/BatchNorm/ConstConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??
?
Csqueezenet/fire8/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean
?
1squeezenet/fire8/expand/3x3/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
shared_name *
dtype0*
	container *
shape:?
?
8squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire8/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire8/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
T0*
_output_shapes	
:?*
validate_shape(*D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean*
use_locking(
?
6squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire8/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean
?
Fsqueezenet/fire8/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance
?
5squeezenet/fire8/expand/3x3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance*
shape:?*
shared_name *
	container 
?
<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire8/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire8/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*
use_locking(
?
:squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire8/expand/3x3/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance
?
6squeezenet/fire8/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire8/expand/3x3/Conv2D+squeezenet/fire8/expand/3x3/BatchNorm/Const/squeezenet/fire8/expand/3x3/BatchNorm/beta/read6squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*G
_output_shapes5
3:?:?:?:?:?:*
data_formatNHWC*
is_training( 
?
 squeezenet/fire8/expand/3x3/ReluRelu6squeezenet/fire8/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:?*
T0
^
squeezenet/fire8/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
squeezenet/fire8/concatConcatV2 squeezenet/fire8/expand/1x1/Relu squeezenet/fire8/expand/3x3/Relusqueezenet/fire8/concat/axis*'
_output_shapes
:?*
N*
T0*

Tidx0
?
squeezenet/maxpool8/MaxPoolMaxPoolsqueezenet/fire8/concat*
strides
*
ksize
*'
_output_shapes
:		?*
T0*
paddingVALID*
data_formatNHWC
?
Asqueezenet/fire9/squeeze/weights/Initializer/random_uniform/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights
?
?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
valueB
 *?ѽ
?
?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/maxConst*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
dtype0*
valueB
 *??=*
_output_shapes
: 
?
Isqueezenet/fire9/squeeze/weights/Initializer/random_uniform/RandomUniformRandomUniformAsqueezenet/fire9/squeeze/weights/Initializer/random_uniform/shape*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
dtype0*
seed2 *

seed *'
_output_shapes
:?@*
T0
?
?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/subSub?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/max?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/min*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
_output_shapes
: *
T0
?
?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/mulMulIsqueezenet/fire9/squeeze/weights/Initializer/random_uniform/RandomUniform?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/sub*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
T0*'
_output_shapes
:?@
?
;squeezenet/fire9/squeeze/weights/Initializer/random_uniformAdd?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/mul?squeezenet/fire9/squeeze/weights/Initializer/random_uniform/min*'
_output_shapes
:?@*
T0*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights
?
 squeezenet/fire9/squeeze/weights
VariableV2*
shape:?@*
shared_name *3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*'
_output_shapes
:?@*
dtype0*
	container 
?
'squeezenet/fire9/squeeze/weights/AssignAssign squeezenet/fire9/squeeze/weights;squeezenet/fire9/squeeze/weights/Initializer/random_uniform*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:?@*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights
?
%squeezenet/fire9/squeeze/weights/readIdentity squeezenet/fire9/squeeze/weights*
T0*'
_output_shapes
:?@*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights
w
&squeezenet/fire9/squeeze/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
squeezenet/fire9/squeeze/Conv2DConv2Dsqueezenet/maxpool8/MaxPool%squeezenet/fire9/squeeze/weights/read*&
_output_shapes
:		@*
use_cudnn_on_gpu(*
strides
*
explicit_paddings
 *
paddingSAME*
data_formatNHWC*
T0*
	dilations

?
9squeezenet/fire9/squeeze/BatchNorm/beta/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta
?
'squeezenet/fire9/squeeze/BatchNorm/beta
VariableV2*
	container *
shape:@*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta*
dtype0
?
.squeezenet/fire9/squeeze/BatchNorm/beta/AssignAssign'squeezenet/fire9/squeeze/BatchNorm/beta9squeezenet/fire9/squeeze/BatchNorm/beta/Initializer/zeros*
_output_shapes
:@*:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta*
validate_shape(*
use_locking(*
T0
?
,squeezenet/fire9/squeeze/BatchNorm/beta/readIdentity'squeezenet/fire9/squeeze/BatchNorm/beta*:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta*
_output_shapes
:@*
T0
u
(squeezenet/fire9/squeeze/BatchNorm/ConstConst*
_output_shapes
:@*
dtype0*
valueB@*  ??
?
@squeezenet/fire9/squeeze/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean
?
.squeezenet/fire9/squeeze/BatchNorm/moving_mean
VariableV2*
_output_shapes
:@*
shape:@*
shared_name *
	container *
dtype0*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean
?
5squeezenet/fire9/squeeze/BatchNorm/moving_mean/AssignAssign.squeezenet/fire9/squeeze/BatchNorm/moving_mean@squeezenet/fire9/squeeze/BatchNorm/moving_mean/Initializer/zeros*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
?
3squeezenet/fire9/squeeze/BatchNorm/moving_mean/readIdentity.squeezenet/fire9/squeeze/BatchNorm/moving_mean*
T0*
_output_shapes
:@*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean
?
Csqueezenet/fire9/squeeze/BatchNorm/moving_variance/Initializer/onesConst*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
2squeezenet/fire9/squeeze/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes
:@*
shared_name *
dtype0*
shape:@*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance
?
9squeezenet/fire9/squeeze/BatchNorm/moving_variance/AssignAssign2squeezenet/fire9/squeeze/BatchNorm/moving_varianceCsqueezenet/fire9/squeeze/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance
?
7squeezenet/fire9/squeeze/BatchNorm/moving_variance/readIdentity2squeezenet/fire9/squeeze/BatchNorm/moving_variance*
T0*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance*
_output_shapes
:@
?
3squeezenet/fire9/squeeze/BatchNorm/FusedBatchNormV3FusedBatchNormV3squeezenet/fire9/squeeze/Conv2D(squeezenet/fire9/squeeze/BatchNorm/Const,squeezenet/fire9/squeeze/BatchNorm/beta/read3squeezenet/fire9/squeeze/BatchNorm/moving_mean/read7squeezenet/fire9/squeeze/BatchNorm/moving_variance/read*B
_output_shapes0
.:		@:@:@:@:@:*
is_training( *
data_formatNHWC*
T0*
U0*
epsilon%o?:
?
squeezenet/fire9/squeeze/ReluRelu3squeezenet/fire9/squeeze/BatchNorm/FusedBatchNormV3*
T0*&
_output_shapes
:		@
?
Dsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*
_output_shapes
:*%
valueB"      @      *
dtype0
?
Bsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/minConst*
valueB
 *?7?*
_output_shapes
: *
dtype0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
?
Bsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/maxConst*
dtype0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*
valueB
 *?7>*
_output_shapes
: 
?
Lsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/shape*

seed *
T0*
dtype0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*
seed2 *'
_output_shapes
:@?
?
Bsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/subSubBsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/maxBsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/min*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*
T0
?
Bsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/mulMulLsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/sub*'
_output_shapes
:@?*
T0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
?
>squeezenet/fire9/expand/1x1/weights/Initializer/random_uniformAddBsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/mulBsqueezenet/fire9/expand/1x1/weights/Initializer/random_uniform/min*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*
T0
?
#squeezenet/fire9/expand/1x1/weights
VariableV2*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights*'
_output_shapes
:@?*
	container *
shared_name *
shape:@?*
dtype0
?
*squeezenet/fire9/expand/1x1/weights/AssignAssign#squeezenet/fire9/expand/1x1/weights>squeezenet/fire9/expand/1x1/weights/Initializer/random_uniform*'
_output_shapes
:@?*
use_locking(*
T0*
validate_shape(*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
?
(squeezenet/fire9/expand/1x1/weights/readIdentity#squeezenet/fire9/expand/1x1/weights*'
_output_shapes
:@?*
T0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
z
)squeezenet/fire9/expand/1x1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
"squeezenet/fire9/expand/1x1/Conv2DConv2Dsqueezenet/fire9/squeeze/Relu(squeezenet/fire9/expand/1x1/weights/read*
use_cudnn_on_gpu(*'
_output_shapes
:		?*
explicit_paddings
 *
T0*
paddingSAME*
strides
*
data_formatNHWC*
	dilations

?
<squeezenet/fire9/expand/1x1/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta
?
*squeezenet/fire9/expand/1x1/BatchNorm/beta
VariableV2*
shape:?*
	container *
dtype0*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta*
_output_shapes	
:?*
shared_name 
?
1squeezenet/fire9/expand/1x1/BatchNorm/beta/AssignAssign*squeezenet/fire9/expand/1x1/BatchNorm/beta<squeezenet/fire9/expand/1x1/BatchNorm/beta/Initializer/zeros*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
/squeezenet/fire9/expand/1x1/BatchNorm/beta/readIdentity*squeezenet/fire9/expand/1x1/BatchNorm/beta*
T0*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta*
_output_shapes	
:?
z
+squeezenet/fire9/expand/1x1/BatchNorm/ConstConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
Csqueezenet/fire9/expand/1x1/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:?*
valueB?*    
?
1squeezenet/fire9/expand/1x1/BatchNorm/moving_mean
VariableV2*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
	container *
_output_shapes	
:?*
shape:?*
shared_name *
dtype0
?
8squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/AssignAssign1squeezenet/fire9/expand/1x1/BatchNorm/moving_meanCsqueezenet/fire9/expand/1x1/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes	
:?*
T0*
use_locking(*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
validate_shape(
?
6squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/readIdentity1squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
T0
?
Fsqueezenet/fire9/expand/1x1/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
dtype0*H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*
valueB?*  ??
?
5squeezenet/fire9/expand/1x1/BatchNorm/moving_variance
VariableV2*
_output_shapes	
:?*
	container *H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*
shape:?*
shared_name *
dtype0
?
<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/AssignAssign5squeezenet/fire9/expand/1x1/BatchNorm/moving_varianceFsqueezenet/fire9/expand/1x1/BatchNorm/moving_variance/Initializer/ones*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance
?
:squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/readIdentity5squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0
?
6squeezenet/fire9/expand/1x1/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire9/expand/1x1/Conv2D+squeezenet/fire9/expand/1x1/BatchNorm/Const/squeezenet/fire9/expand/1x1/BatchNorm/beta/read6squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/read:squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/read*G
_output_shapes5
3:		?:?:?:?:?:*
T0*
U0*
data_formatNHWC*
epsilon%o?:*
is_training( 
?
 squeezenet/fire9/expand/1x1/ReluRelu6squeezenet/fire9/expand/1x1/BatchNorm/FusedBatchNormV3*'
_output_shapes
:		?*
T0
?
Dsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/shapeConst*%
valueB"      @      *6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
_output_shapes
:*
dtype0
?
Bsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/minConst*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
dtype0*
valueB
 *??:?
?
Bsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/maxConst*
valueB
 *??:=*
dtype0*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
_output_shapes
: 
?
Lsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/RandomUniformRandomUniformDsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/shape*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
dtype0*
T0*

seed *'
_output_shapes
:@?*
seed2 
?
Bsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/subSubBsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/maxBsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/min*
_output_shapes
: *6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
T0
?
Bsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/mulMulLsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/RandomUniformBsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/sub*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
T0
?
>squeezenet/fire9/expand/3x3/weights/Initializer/random_uniformAddBsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/mulBsqueezenet/fire9/expand/3x3/weights/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*'
_output_shapes
:@?
?
#squeezenet/fire9/expand/3x3/weights
VariableV2*'
_output_shapes
:@?*
dtype0*
shared_name *
shape:@?*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
	container 
?
*squeezenet/fire9/expand/3x3/weights/AssignAssign#squeezenet/fire9/expand/3x3/weights>squeezenet/fire9/expand/3x3/weights/Initializer/random_uniform*
use_locking(*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
validate_shape(*
T0
?
(squeezenet/fire9/expand/3x3/weights/readIdentity#squeezenet/fire9/expand/3x3/weights*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
T0
z
)squeezenet/fire9/expand/3x3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
"squeezenet/fire9/expand/3x3/Conv2DConv2Dsqueezenet/fire9/squeeze/Relu(squeezenet/fire9/expand/3x3/weights/read*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
T0*'
_output_shapes
:		?*
paddingSAME*
data_formatNHWC*
	dilations

?
<squeezenet/fire9/expand/3x3/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta
?
*squeezenet/fire9/expand/3x3/BatchNorm/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
	container *=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta*
shared_name 
?
1squeezenet/fire9/expand/3x3/BatchNorm/beta/AssignAssign*squeezenet/fire9/expand/3x3/BatchNorm/beta<squeezenet/fire9/expand/3x3/BatchNorm/beta/Initializer/zeros*
T0*=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
/squeezenet/fire9/expand/3x3/BatchNorm/beta/readIdentity*squeezenet/fire9/expand/3x3/BatchNorm/beta*
T0*=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta*
_output_shapes	
:?
z
+squeezenet/fire9/expand/3x3/BatchNorm/ConstConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
Csqueezenet/fire9/expand/3x3/BatchNorm/moving_mean/Initializer/zerosConst*D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
1squeezenet/fire9/expand/3x3/BatchNorm/moving_mean
VariableV2*
dtype0*
shape:?*
_output_shapes	
:?*
shared_name *
	container *D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean
?
8squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/AssignAssign1squeezenet/fire9/expand/3x3/BatchNorm/moving_meanCsqueezenet/fire9/expand/3x3/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
validate_shape(*
use_locking(*
T0
?
6squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/readIdentity1squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
T0
?
Fsqueezenet/fire9/expand/3x3/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
dtype0*
valueB?*  ??*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance
?
5squeezenet/fire9/expand/3x3/BatchNorm/moving_variance
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:?*
shape:?*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance
?
<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/AssignAssign5squeezenet/fire9/expand/3x3/BatchNorm/moving_varianceFsqueezenet/fire9/expand/3x3/BatchNorm/moving_variance/Initializer/ones*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance
?
:squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/readIdentity5squeezenet/fire9/expand/3x3/BatchNorm/moving_variance*
T0*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
6squeezenet/fire9/expand/3x3/BatchNorm/FusedBatchNormV3FusedBatchNormV3"squeezenet/fire9/expand/3x3/Conv2D+squeezenet/fire9/expand/3x3/BatchNorm/Const/squeezenet/fire9/expand/3x3/BatchNorm/beta/read6squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/read:squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/read*
T0*
U0*
epsilon%o?:*
data_formatNHWC*
is_training( *G
_output_shapes5
3:		?:?:?:?:?:
?
 squeezenet/fire9/expand/3x3/ReluRelu6squeezenet/fire9/expand/3x3/BatchNorm/FusedBatchNormV3*'
_output_shapes
:		?*
T0
^
squeezenet/fire9/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
?
squeezenet/fire9/concatConcatV2 squeezenet/fire9/expand/1x1/Relu squeezenet/fire9/expand/3x3/Relusqueezenet/fire9/concat/axis*'
_output_shapes
:		?*

Tidx0*
T0*
N
r
squeezenet/Dropout/IdentityIdentitysqueezenet/fire9/concat*'
_output_shapes
:		?*
T0
?
:squeezenet/conv10/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*,
_class"
 loc:@squeezenet/conv10/weights*
dtype0*%
valueB"         ?  
?
8squeezenet/conv10/weights/Initializer/random_uniform/minConst*
valueB
 *
??*
_output_shapes
: *,
_class"
 loc:@squeezenet/conv10/weights*
dtype0
?
8squeezenet/conv10/weights/Initializer/random_uniform/maxConst*,
_class"
 loc:@squeezenet/conv10/weights*
dtype0*
_output_shapes
: *
valueB
 *
?=
?
Bsqueezenet/conv10/weights/Initializer/random_uniform/RandomUniformRandomUniform:squeezenet/conv10/weights/Initializer/random_uniform/shape*

seed *(
_output_shapes
:??*
T0*,
_class"
 loc:@squeezenet/conv10/weights*
seed2 *
dtype0
?
8squeezenet/conv10/weights/Initializer/random_uniform/subSub8squeezenet/conv10/weights/Initializer/random_uniform/max8squeezenet/conv10/weights/Initializer/random_uniform/min*
T0*,
_class"
 loc:@squeezenet/conv10/weights*
_output_shapes
: 
?
8squeezenet/conv10/weights/Initializer/random_uniform/mulMulBsqueezenet/conv10/weights/Initializer/random_uniform/RandomUniform8squeezenet/conv10/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*
T0*,
_class"
 loc:@squeezenet/conv10/weights
?
4squeezenet/conv10/weights/Initializer/random_uniformAdd8squeezenet/conv10/weights/Initializer/random_uniform/mul8squeezenet/conv10/weights/Initializer/random_uniform/min*,
_class"
 loc:@squeezenet/conv10/weights*
T0*(
_output_shapes
:??
?
squeezenet/conv10/weights
VariableV2*
	container *(
_output_shapes
:??*
shared_name *,
_class"
 loc:@squeezenet/conv10/weights*
dtype0*
shape:??
?
 squeezenet/conv10/weights/AssignAssignsqueezenet/conv10/weights4squeezenet/conv10/weights/Initializer/random_uniform*,
_class"
 loc:@squeezenet/conv10/weights*(
_output_shapes
:??*
validate_shape(*
use_locking(*
T0
?
squeezenet/conv10/weights/readIdentitysqueezenet/conv10/weights*
T0*(
_output_shapes
:??*,
_class"
 loc:@squeezenet/conv10/weights
?
:squeezenet/conv10/biases/Initializer/zeros/shape_as_tensorConst*+
_class!
loc:@squeezenet/conv10/biases*
valueB:?*
dtype0*
_output_shapes
:
?
0squeezenet/conv10/biases/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes
: 
?
*squeezenet/conv10/biases/Initializer/zerosFill:squeezenet/conv10/biases/Initializer/zeros/shape_as_tensor0squeezenet/conv10/biases/Initializer/zeros/Const*
T0*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes	
:?*

index_type0
?
squeezenet/conv10/biases
VariableV2*
	container *
dtype0*
shape:?*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes	
:?*
shared_name 
?
squeezenet/conv10/biases/AssignAssignsqueezenet/conv10/biases*squeezenet/conv10/biases/Initializer/zeros*
T0*
validate_shape(*
_output_shapes	
:?*+
_class!
loc:@squeezenet/conv10/biases*
use_locking(
?
squeezenet/conv10/biases/readIdentitysqueezenet/conv10/biases*
T0*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes	
:?
p
squeezenet/conv10/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
squeezenet/conv10/Conv2DConv2Dsqueezenet/Dropout/Identitysqueezenet/conv10/weights/read*
	dilations
*
use_cudnn_on_gpu(*'
_output_shapes
:		?*
data_formatNHWC*
strides
*
T0*
explicit_paddings
 *
paddingSAME
?
squeezenet/conv10/BiasAddBiasAddsqueezenet/conv10/Conv2Dsqueezenet/conv10/biases/read*
T0*'
_output_shapes
:		?*
data_formatNHWC
?
squeezenet/avgpool10/AvgPoolAvgPoolsqueezenet/conv10/BiasAdd*
ksize
		*
paddingVALID*'
_output_shapes
:?*
data_formatNHWC*
strides
*
T0
|
squeezenet/logitsSqueezesqueezenet/avgpool10/AvgPool*
T0*
_output_shapes
:	?*
squeeze_dims

?
>squeezenet/Bottleneck/weights/Initializer/random_uniform/shapeConst*
valueB"?     *
dtype0*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
_output_shapes
:
?
<squeezenet/Bottleneck/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *
??*0
_class&
$"loc:@squeezenet/Bottleneck/weights
?
<squeezenet/Bottleneck/weights/Initializer/random_uniform/maxConst*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
dtype0*
_output_shapes
: *
valueB
 *
?=
?
Fsqueezenet/Bottleneck/weights/Initializer/random_uniform/RandomUniformRandomUniform>squeezenet/Bottleneck/weights/Initializer/random_uniform/shape*
dtype0*
seed2 * 
_output_shapes
:
??*0
_class&
$"loc:@squeezenet/Bottleneck/weights*

seed *
T0
?
<squeezenet/Bottleneck/weights/Initializer/random_uniform/subSub<squeezenet/Bottleneck/weights/Initializer/random_uniform/max<squeezenet/Bottleneck/weights/Initializer/random_uniform/min*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
_output_shapes
: *
T0
?
<squeezenet/Bottleneck/weights/Initializer/random_uniform/mulMulFsqueezenet/Bottleneck/weights/Initializer/random_uniform/RandomUniform<squeezenet/Bottleneck/weights/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
??*0
_class&
$"loc:@squeezenet/Bottleneck/weights
?
8squeezenet/Bottleneck/weights/Initializer/random_uniformAdd<squeezenet/Bottleneck/weights/Initializer/random_uniform/mul<squeezenet/Bottleneck/weights/Initializer/random_uniform/min* 
_output_shapes
:
??*
T0*0
_class&
$"loc:@squeezenet/Bottleneck/weights
?
squeezenet/Bottleneck/weights
VariableV2* 
_output_shapes
:
??*
shared_name *
	container *
shape:
??*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
dtype0
?
$squeezenet/Bottleneck/weights/AssignAssignsqueezenet/Bottleneck/weights8squeezenet/Bottleneck/weights/Initializer/random_uniform*
validate_shape(*
T0* 
_output_shapes
:
??*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
use_locking(
?
"squeezenet/Bottleneck/weights/readIdentitysqueezenet/Bottleneck/weights*0
_class&
$"loc:@squeezenet/Bottleneck/weights* 
_output_shapes
:
??*
T0
?
squeezenet/Bottleneck/MatMulMatMulsqueezenet/logits"squeezenet/Bottleneck/weights/read*
_output_shapes
:	?*
T0*
transpose_a( *
transpose_b( 
?
-squeezenet/Bottleneck/BatchNorm/Reshape/shapeConst*
dtype0*%
valueB"????         *
_output_shapes
:
?
'squeezenet/Bottleneck/BatchNorm/ReshapeReshapesqueezenet/Bottleneck/MatMul-squeezenet/Bottleneck/BatchNorm/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:?
?
6squeezenet/Bottleneck/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta
?
$squeezenet/Bottleneck/BatchNorm/beta
VariableV2*
	container *7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta*
dtype0*
shared_name *
_output_shapes	
:?*
shape:?
?
+squeezenet/Bottleneck/BatchNorm/beta/AssignAssign$squeezenet/Bottleneck/BatchNorm/beta6squeezenet/Bottleneck/BatchNorm/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
validate_shape(*7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta*
T0
?
)squeezenet/Bottleneck/BatchNorm/beta/readIdentity$squeezenet/Bottleneck/BatchNorm/beta*
_output_shapes	
:?*7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta*
T0
t
%squeezenet/Bottleneck/BatchNorm/ConstConst*
_output_shapes	
:?*
dtype0*
valueB?*  ??
?
=squeezenet/Bottleneck/BatchNorm/moving_mean/Initializer/zerosConst*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean*
valueB?*    *
_output_shapes	
:?*
dtype0
?
+squeezenet/Bottleneck/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *
shared_name *
dtype0*
_output_shapes	
:?*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean
?
2squeezenet/Bottleneck/BatchNorm/moving_mean/AssignAssign+squeezenet/Bottleneck/BatchNorm/moving_mean=squeezenet/Bottleneck/BatchNorm/moving_mean/Initializer/zeros*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
0squeezenet/Bottleneck/BatchNorm/moving_mean/readIdentity+squeezenet/Bottleneck/BatchNorm/moving_mean*
_output_shapes	
:?*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean*
T0
?
@squeezenet/Bottleneck/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance*
dtype0
?
/squeezenet/Bottleneck/BatchNorm/moving_variance
VariableV2*
dtype0*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance*
shared_name *
_output_shapes	
:?*
shape:?*
	container 
?
6squeezenet/Bottleneck/BatchNorm/moving_variance/AssignAssign/squeezenet/Bottleneck/BatchNorm/moving_variance@squeezenet/Bottleneck/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance
?
4squeezenet/Bottleneck/BatchNorm/moving_variance/readIdentity/squeezenet/Bottleneck/BatchNorm/moving_variance*
_output_shapes	
:?*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance*
T0
?
0squeezenet/Bottleneck/BatchNorm/FusedBatchNormV3FusedBatchNormV3'squeezenet/Bottleneck/BatchNorm/Reshape%squeezenet/Bottleneck/BatchNorm/Const)squeezenet/Bottleneck/BatchNorm/beta/read0squeezenet/Bottleneck/BatchNorm/moving_mean/read4squeezenet/Bottleneck/BatchNorm/moving_variance/read*
is_training( *
T0*
data_formatNHWC*
U0*
epsilon%o?:*G
_output_shapes5
3:?:?:?:?:?:
v
%squeezenet/Bottleneck/BatchNorm/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
?
)squeezenet/Bottleneck/BatchNorm/Reshape_1Reshape0squeezenet/Bottleneck/BatchNorm/FusedBatchNormV3%squeezenet/Bottleneck/BatchNorm/Shape*
Tshape0*
T0*
_output_shapes
:	?
p
embeddings/SquareSquare)squeezenet/Bottleneck/BatchNorm/Reshape_1*
_output_shapes
:	?*
T0
b
 embeddings/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
?
embeddings/SumSumembeddings/Square embeddings/Sum/reduction_indices*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
Y
embeddings/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.
l
embeddings/MaximumMaximumembeddings/Sumembeddings/Maximum/y*
T0*
_output_shapes

:
V
embeddings/RsqrtRsqrtembeddings/Maximum*
_output_shapes

:*
T0
x

embeddingsMul)squeezenet/Bottleneck/BatchNorm/Reshape_1embeddings/Rsqrt*
T0*
_output_shapes
:	?
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
?%
save/SaveV2/tensor_namesConst*
_output_shapes
:j*?%
value?%B?%jB$squeezenet/Bottleneck/BatchNorm/betaB+squeezenet/Bottleneck/BatchNorm/moving_meanB/squeezenet/Bottleneck/BatchNorm/moving_varianceBsqueezenet/Bottleneck/weightsBsqueezenet/conv1/BatchNorm/betaB&squeezenet/conv1/BatchNorm/moving_meanB*squeezenet/conv1/BatchNorm/moving_varianceBsqueezenet/conv1/weightsBsqueezenet/conv10/biasesBsqueezenet/conv10/weightsB*squeezenet/fire2/expand/1x1/BatchNorm/betaB1squeezenet/fire2/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire2/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire2/expand/1x1/weightsB*squeezenet/fire2/expand/3x3/BatchNorm/betaB1squeezenet/fire2/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire2/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire2/expand/3x3/weightsB'squeezenet/fire2/squeeze/BatchNorm/betaB.squeezenet/fire2/squeeze/BatchNorm/moving_meanB2squeezenet/fire2/squeeze/BatchNorm/moving_varianceB squeezenet/fire2/squeeze/weightsB*squeezenet/fire3/expand/1x1/BatchNorm/betaB1squeezenet/fire3/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire3/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire3/expand/1x1/weightsB*squeezenet/fire3/expand/3x3/BatchNorm/betaB1squeezenet/fire3/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire3/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire3/expand/3x3/weightsB'squeezenet/fire3/squeeze/BatchNorm/betaB.squeezenet/fire3/squeeze/BatchNorm/moving_meanB2squeezenet/fire3/squeeze/BatchNorm/moving_varianceB squeezenet/fire3/squeeze/weightsB*squeezenet/fire4/expand/1x1/BatchNorm/betaB1squeezenet/fire4/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire4/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire4/expand/1x1/weightsB*squeezenet/fire4/expand/3x3/BatchNorm/betaB1squeezenet/fire4/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire4/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire4/expand/3x3/weightsB'squeezenet/fire4/squeeze/BatchNorm/betaB.squeezenet/fire4/squeeze/BatchNorm/moving_meanB2squeezenet/fire4/squeeze/BatchNorm/moving_varianceB squeezenet/fire4/squeeze/weightsB*squeezenet/fire5/expand/1x1/BatchNorm/betaB1squeezenet/fire5/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire5/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire5/expand/1x1/weightsB*squeezenet/fire5/expand/3x3/BatchNorm/betaB1squeezenet/fire5/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire5/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire5/expand/3x3/weightsB'squeezenet/fire5/squeeze/BatchNorm/betaB.squeezenet/fire5/squeeze/BatchNorm/moving_meanB2squeezenet/fire5/squeeze/BatchNorm/moving_varianceB squeezenet/fire5/squeeze/weightsB*squeezenet/fire6/expand/1x1/BatchNorm/betaB1squeezenet/fire6/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire6/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire6/expand/1x1/weightsB*squeezenet/fire6/expand/3x3/BatchNorm/betaB1squeezenet/fire6/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire6/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire6/expand/3x3/weightsB'squeezenet/fire6/squeeze/BatchNorm/betaB.squeezenet/fire6/squeeze/BatchNorm/moving_meanB2squeezenet/fire6/squeeze/BatchNorm/moving_varianceB squeezenet/fire6/squeeze/weightsB*squeezenet/fire7/expand/1x1/BatchNorm/betaB1squeezenet/fire7/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire7/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire7/expand/1x1/weightsB*squeezenet/fire7/expand/3x3/BatchNorm/betaB1squeezenet/fire7/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire7/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire7/expand/3x3/weightsB'squeezenet/fire7/squeeze/BatchNorm/betaB.squeezenet/fire7/squeeze/BatchNorm/moving_meanB2squeezenet/fire7/squeeze/BatchNorm/moving_varianceB squeezenet/fire7/squeeze/weightsB*squeezenet/fire8/expand/1x1/BatchNorm/betaB1squeezenet/fire8/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire8/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire8/expand/1x1/weightsB*squeezenet/fire8/expand/3x3/BatchNorm/betaB1squeezenet/fire8/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire8/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire8/expand/3x3/weightsB'squeezenet/fire8/squeeze/BatchNorm/betaB.squeezenet/fire8/squeeze/BatchNorm/moving_meanB2squeezenet/fire8/squeeze/BatchNorm/moving_varianceB squeezenet/fire8/squeeze/weightsB*squeezenet/fire9/expand/1x1/BatchNorm/betaB1squeezenet/fire9/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire9/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire9/expand/1x1/weightsB*squeezenet/fire9/expand/3x3/BatchNorm/betaB1squeezenet/fire9/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire9/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire9/expand/3x3/weightsB'squeezenet/fire9/squeeze/BatchNorm/betaB.squeezenet/fire9/squeeze/BatchNorm/moving_meanB2squeezenet/fire9/squeeze/BatchNorm/moving_varianceB squeezenet/fire9/squeeze/weights*
dtype0
?
save/SaveV2/shape_and_slicesConst*
_output_shapes
:j*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?&
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$squeezenet/Bottleneck/BatchNorm/beta+squeezenet/Bottleneck/BatchNorm/moving_mean/squeezenet/Bottleneck/BatchNorm/moving_variancesqueezenet/Bottleneck/weightssqueezenet/conv1/BatchNorm/beta&squeezenet/conv1/BatchNorm/moving_mean*squeezenet/conv1/BatchNorm/moving_variancesqueezenet/conv1/weightssqueezenet/conv10/biasessqueezenet/conv10/weights*squeezenet/fire2/expand/1x1/BatchNorm/beta1squeezenet/fire2/expand/1x1/BatchNorm/moving_mean5squeezenet/fire2/expand/1x1/BatchNorm/moving_variance#squeezenet/fire2/expand/1x1/weights*squeezenet/fire2/expand/3x3/BatchNorm/beta1squeezenet/fire2/expand/3x3/BatchNorm/moving_mean5squeezenet/fire2/expand/3x3/BatchNorm/moving_variance#squeezenet/fire2/expand/3x3/weights'squeezenet/fire2/squeeze/BatchNorm/beta.squeezenet/fire2/squeeze/BatchNorm/moving_mean2squeezenet/fire2/squeeze/BatchNorm/moving_variance squeezenet/fire2/squeeze/weights*squeezenet/fire3/expand/1x1/BatchNorm/beta1squeezenet/fire3/expand/1x1/BatchNorm/moving_mean5squeezenet/fire3/expand/1x1/BatchNorm/moving_variance#squeezenet/fire3/expand/1x1/weights*squeezenet/fire3/expand/3x3/BatchNorm/beta1squeezenet/fire3/expand/3x3/BatchNorm/moving_mean5squeezenet/fire3/expand/3x3/BatchNorm/moving_variance#squeezenet/fire3/expand/3x3/weights'squeezenet/fire3/squeeze/BatchNorm/beta.squeezenet/fire3/squeeze/BatchNorm/moving_mean2squeezenet/fire3/squeeze/BatchNorm/moving_variance squeezenet/fire3/squeeze/weights*squeezenet/fire4/expand/1x1/BatchNorm/beta1squeezenet/fire4/expand/1x1/BatchNorm/moving_mean5squeezenet/fire4/expand/1x1/BatchNorm/moving_variance#squeezenet/fire4/expand/1x1/weights*squeezenet/fire4/expand/3x3/BatchNorm/beta1squeezenet/fire4/expand/3x3/BatchNorm/moving_mean5squeezenet/fire4/expand/3x3/BatchNorm/moving_variance#squeezenet/fire4/expand/3x3/weights'squeezenet/fire4/squeeze/BatchNorm/beta.squeezenet/fire4/squeeze/BatchNorm/moving_mean2squeezenet/fire4/squeeze/BatchNorm/moving_variance squeezenet/fire4/squeeze/weights*squeezenet/fire5/expand/1x1/BatchNorm/beta1squeezenet/fire5/expand/1x1/BatchNorm/moving_mean5squeezenet/fire5/expand/1x1/BatchNorm/moving_variance#squeezenet/fire5/expand/1x1/weights*squeezenet/fire5/expand/3x3/BatchNorm/beta1squeezenet/fire5/expand/3x3/BatchNorm/moving_mean5squeezenet/fire5/expand/3x3/BatchNorm/moving_variance#squeezenet/fire5/expand/3x3/weights'squeezenet/fire5/squeeze/BatchNorm/beta.squeezenet/fire5/squeeze/BatchNorm/moving_mean2squeezenet/fire5/squeeze/BatchNorm/moving_variance squeezenet/fire5/squeeze/weights*squeezenet/fire6/expand/1x1/BatchNorm/beta1squeezenet/fire6/expand/1x1/BatchNorm/moving_mean5squeezenet/fire6/expand/1x1/BatchNorm/moving_variance#squeezenet/fire6/expand/1x1/weights*squeezenet/fire6/expand/3x3/BatchNorm/beta1squeezenet/fire6/expand/3x3/BatchNorm/moving_mean5squeezenet/fire6/expand/3x3/BatchNorm/moving_variance#squeezenet/fire6/expand/3x3/weights'squeezenet/fire6/squeeze/BatchNorm/beta.squeezenet/fire6/squeeze/BatchNorm/moving_mean2squeezenet/fire6/squeeze/BatchNorm/moving_variance squeezenet/fire6/squeeze/weights*squeezenet/fire7/expand/1x1/BatchNorm/beta1squeezenet/fire7/expand/1x1/BatchNorm/moving_mean5squeezenet/fire7/expand/1x1/BatchNorm/moving_variance#squeezenet/fire7/expand/1x1/weights*squeezenet/fire7/expand/3x3/BatchNorm/beta1squeezenet/fire7/expand/3x3/BatchNorm/moving_mean5squeezenet/fire7/expand/3x3/BatchNorm/moving_variance#squeezenet/fire7/expand/3x3/weights'squeezenet/fire7/squeeze/BatchNorm/beta.squeezenet/fire7/squeeze/BatchNorm/moving_mean2squeezenet/fire7/squeeze/BatchNorm/moving_variance squeezenet/fire7/squeeze/weights*squeezenet/fire8/expand/1x1/BatchNorm/beta1squeezenet/fire8/expand/1x1/BatchNorm/moving_mean5squeezenet/fire8/expand/1x1/BatchNorm/moving_variance#squeezenet/fire8/expand/1x1/weights*squeezenet/fire8/expand/3x3/BatchNorm/beta1squeezenet/fire8/expand/3x3/BatchNorm/moving_mean5squeezenet/fire8/expand/3x3/BatchNorm/moving_variance#squeezenet/fire8/expand/3x3/weights'squeezenet/fire8/squeeze/BatchNorm/beta.squeezenet/fire8/squeeze/BatchNorm/moving_mean2squeezenet/fire8/squeeze/BatchNorm/moving_variance squeezenet/fire8/squeeze/weights*squeezenet/fire9/expand/1x1/BatchNorm/beta1squeezenet/fire9/expand/1x1/BatchNorm/moving_mean5squeezenet/fire9/expand/1x1/BatchNorm/moving_variance#squeezenet/fire9/expand/1x1/weights*squeezenet/fire9/expand/3x3/BatchNorm/beta1squeezenet/fire9/expand/3x3/BatchNorm/moving_mean5squeezenet/fire9/expand/3x3/BatchNorm/moving_variance#squeezenet/fire9/expand/3x3/weights'squeezenet/fire9/squeeze/BatchNorm/beta.squeezenet/fire9/squeeze/BatchNorm/moving_mean2squeezenet/fire9/squeeze/BatchNorm/moving_variance squeezenet/fire9/squeeze/weights*x
dtypesn
l2j
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
?%
save/RestoreV2/tensor_namesConst*
_output_shapes
:j*
dtype0*?%
value?%B?%jB$squeezenet/Bottleneck/BatchNorm/betaB+squeezenet/Bottleneck/BatchNorm/moving_meanB/squeezenet/Bottleneck/BatchNorm/moving_varianceBsqueezenet/Bottleneck/weightsBsqueezenet/conv1/BatchNorm/betaB&squeezenet/conv1/BatchNorm/moving_meanB*squeezenet/conv1/BatchNorm/moving_varianceBsqueezenet/conv1/weightsBsqueezenet/conv10/biasesBsqueezenet/conv10/weightsB*squeezenet/fire2/expand/1x1/BatchNorm/betaB1squeezenet/fire2/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire2/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire2/expand/1x1/weightsB*squeezenet/fire2/expand/3x3/BatchNorm/betaB1squeezenet/fire2/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire2/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire2/expand/3x3/weightsB'squeezenet/fire2/squeeze/BatchNorm/betaB.squeezenet/fire2/squeeze/BatchNorm/moving_meanB2squeezenet/fire2/squeeze/BatchNorm/moving_varianceB squeezenet/fire2/squeeze/weightsB*squeezenet/fire3/expand/1x1/BatchNorm/betaB1squeezenet/fire3/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire3/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire3/expand/1x1/weightsB*squeezenet/fire3/expand/3x3/BatchNorm/betaB1squeezenet/fire3/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire3/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire3/expand/3x3/weightsB'squeezenet/fire3/squeeze/BatchNorm/betaB.squeezenet/fire3/squeeze/BatchNorm/moving_meanB2squeezenet/fire3/squeeze/BatchNorm/moving_varianceB squeezenet/fire3/squeeze/weightsB*squeezenet/fire4/expand/1x1/BatchNorm/betaB1squeezenet/fire4/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire4/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire4/expand/1x1/weightsB*squeezenet/fire4/expand/3x3/BatchNorm/betaB1squeezenet/fire4/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire4/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire4/expand/3x3/weightsB'squeezenet/fire4/squeeze/BatchNorm/betaB.squeezenet/fire4/squeeze/BatchNorm/moving_meanB2squeezenet/fire4/squeeze/BatchNorm/moving_varianceB squeezenet/fire4/squeeze/weightsB*squeezenet/fire5/expand/1x1/BatchNorm/betaB1squeezenet/fire5/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire5/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire5/expand/1x1/weightsB*squeezenet/fire5/expand/3x3/BatchNorm/betaB1squeezenet/fire5/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire5/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire5/expand/3x3/weightsB'squeezenet/fire5/squeeze/BatchNorm/betaB.squeezenet/fire5/squeeze/BatchNorm/moving_meanB2squeezenet/fire5/squeeze/BatchNorm/moving_varianceB squeezenet/fire5/squeeze/weightsB*squeezenet/fire6/expand/1x1/BatchNorm/betaB1squeezenet/fire6/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire6/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire6/expand/1x1/weightsB*squeezenet/fire6/expand/3x3/BatchNorm/betaB1squeezenet/fire6/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire6/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire6/expand/3x3/weightsB'squeezenet/fire6/squeeze/BatchNorm/betaB.squeezenet/fire6/squeeze/BatchNorm/moving_meanB2squeezenet/fire6/squeeze/BatchNorm/moving_varianceB squeezenet/fire6/squeeze/weightsB*squeezenet/fire7/expand/1x1/BatchNorm/betaB1squeezenet/fire7/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire7/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire7/expand/1x1/weightsB*squeezenet/fire7/expand/3x3/BatchNorm/betaB1squeezenet/fire7/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire7/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire7/expand/3x3/weightsB'squeezenet/fire7/squeeze/BatchNorm/betaB.squeezenet/fire7/squeeze/BatchNorm/moving_meanB2squeezenet/fire7/squeeze/BatchNorm/moving_varianceB squeezenet/fire7/squeeze/weightsB*squeezenet/fire8/expand/1x1/BatchNorm/betaB1squeezenet/fire8/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire8/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire8/expand/1x1/weightsB*squeezenet/fire8/expand/3x3/BatchNorm/betaB1squeezenet/fire8/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire8/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire8/expand/3x3/weightsB'squeezenet/fire8/squeeze/BatchNorm/betaB.squeezenet/fire8/squeeze/BatchNorm/moving_meanB2squeezenet/fire8/squeeze/BatchNorm/moving_varianceB squeezenet/fire8/squeeze/weightsB*squeezenet/fire9/expand/1x1/BatchNorm/betaB1squeezenet/fire9/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire9/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire9/expand/1x1/weightsB*squeezenet/fire9/expand/3x3/BatchNorm/betaB1squeezenet/fire9/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire9/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire9/expand/3x3/weightsB'squeezenet/fire9/squeeze/BatchNorm/betaB.squeezenet/fire9/squeeze/BatchNorm/moving_meanB2squeezenet/fire9/squeeze/BatchNorm/moving_varianceB squeezenet/fire9/squeeze/weights
?
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:j*
dtype0*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*x
dtypesn
l2j*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssign$squeezenet/Bottleneck/BatchNorm/betasave/RestoreV2*7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(
?
save/Assign_1Assign+squeezenet/Bottleneck/BatchNorm/moving_meansave/RestoreV2:1*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean*
validate_shape(
?
save/Assign_2Assign/squeezenet/Bottleneck/BatchNorm/moving_variancesave/RestoreV2:2*
_output_shapes	
:?*
validate_shape(*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance*
use_locking(*
T0
?
save/Assign_3Assignsqueezenet/Bottleneck/weightssave/RestoreV2:3*
T0* 
_output_shapes
:
??*
use_locking(*0
_class&
$"loc:@squeezenet/Bottleneck/weights*
validate_shape(
?
save/Assign_4Assignsqueezenet/conv1/BatchNorm/betasave/RestoreV2:4*
T0*
use_locking(*
validate_shape(*2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta*
_output_shapes
:`
?
save/Assign_5Assign&squeezenet/conv1/BatchNorm/moving_meansave/RestoreV2:5*
validate_shape(*
_output_shapes
:`*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
T0*
use_locking(
?
save/Assign_6Assign*squeezenet/conv1/BatchNorm/moving_variancesave/RestoreV2:6*
validate_shape(*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance*
_output_shapes
:`*
use_locking(*
T0
?
save/Assign_7Assignsqueezenet/conv1/weightssave/RestoreV2:7*
use_locking(*
validate_shape(*+
_class!
loc:@squeezenet/conv1/weights*&
_output_shapes
:`*
T0
?
save/Assign_8Assignsqueezenet/conv10/biasessave/RestoreV2:8*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
save/Assign_9Assignsqueezenet/conv10/weightssave/RestoreV2:9*
use_locking(*
T0*(
_output_shapes
:??*,
_class"
 loc:@squeezenet/conv10/weights*
validate_shape(
?
save/Assign_10Assign*squeezenet/fire2/expand/1x1/BatchNorm/betasave/RestoreV2:10*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta
?
save/Assign_11Assign1squeezenet/fire2/expand/1x1/BatchNorm/moving_meansave/RestoreV2:11*
use_locking(*
_output_shapes
:@*D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
T0*
validate_shape(
?
save/Assign_12Assign5squeezenet/fire2/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:12*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_13Assign#squeezenet/fire2/expand/1x1/weightssave/RestoreV2:13*
use_locking(*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
T0*
validate_shape(*&
_output_shapes
:@
?
save/Assign_14Assign*squeezenet/fire2/expand/3x3/BatchNorm/betasave/RestoreV2:14*
_output_shapes
:@*
validate_shape(*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta*
T0*
use_locking(
?
save/Assign_15Assign1squeezenet/fire2/expand/3x3/BatchNorm/moving_meansave/RestoreV2:15*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean
?
save/Assign_16Assign5squeezenet/fire2/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:16*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
T0
?
save/Assign_17Assign#squeezenet/fire2/expand/3x3/weightssave/RestoreV2:17*
T0*&
_output_shapes
:@*
use_locking(*
validate_shape(*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights
?
save/Assign_18Assign'squeezenet/fire2/squeeze/BatchNorm/betasave/RestoreV2:18*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
?
save/Assign_19Assign.squeezenet/fire2/squeeze/BatchNorm/moving_meansave/RestoreV2:19*
_output_shapes
:*A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean*
use_locking(*
validate_shape(*
T0
?
save/Assign_20Assign2squeezenet/fire2/squeeze/BatchNorm/moving_variancesave/RestoreV2:20*
_output_shapes
:*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
save/Assign_21Assign squeezenet/fire2/squeeze/weightssave/RestoreV2:21*&
_output_shapes
:`*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
T0*
validate_shape(*
use_locking(
?
save/Assign_22Assign*squeezenet/fire3/expand/1x1/BatchNorm/betasave/RestoreV2:22*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(*=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta
?
save/Assign_23Assign1squeezenet/fire3/expand/1x1/BatchNorm/moving_meansave/RestoreV2:23*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_24Assign5squeezenet/fire3/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:24*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance
?
save/Assign_25Assign#squeezenet/fire3/expand/1x1/weightssave/RestoreV2:25*
use_locking(*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*
validate_shape(*&
_output_shapes
:@*
T0
?
save/Assign_26Assign*squeezenet/fire3/expand/3x3/BatchNorm/betasave/RestoreV2:26*
T0*=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta*
use_locking(*
_output_shapes
:@*
validate_shape(
?
save/Assign_27Assign1squeezenet/fire3/expand/3x3/BatchNorm/moving_meansave/RestoreV2:27*
use_locking(*
_output_shapes
:@*
validate_shape(*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
T0
?
save/Assign_28Assign5squeezenet/fire3/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:28*
validate_shape(*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@*
use_locking(*
T0
?
save/Assign_29Assign#squeezenet/fire3/expand/3x3/weightssave/RestoreV2:29*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*
T0*&
_output_shapes
:@*
use_locking(*
validate_shape(
?
save/Assign_30Assign'squeezenet/fire3/squeeze/BatchNorm/betasave/RestoreV2:30*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta
?
save/Assign_31Assign.squeezenet/fire3/squeeze/BatchNorm/moving_meansave/RestoreV2:31*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean
?
save/Assign_32Assign2squeezenet/fire3/squeeze/BatchNorm/moving_variancesave/RestoreV2:32*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance
?
save/Assign_33Assign squeezenet/fire3/squeeze/weightssave/RestoreV2:33*
T0*'
_output_shapes
:?*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights*
use_locking(*
validate_shape(
?
save/Assign_34Assign*squeezenet/fire4/expand/1x1/BatchNorm/betasave/RestoreV2:34*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_35Assign1squeezenet/fire4/expand/1x1/BatchNorm/moving_meansave/RestoreV2:35*
validate_shape(*
use_locking(*D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0
?
save/Assign_36Assign5squeezenet/fire4/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:36*
T0*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_37Assign#squeezenet/fire4/expand/1x1/weightssave/RestoreV2:37*
T0*
use_locking(*'
_output_shapes
: ?*
validate_shape(*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights
?
save/Assign_38Assign*squeezenet/fire4/expand/3x3/BatchNorm/betasave/RestoreV2:38*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save/Assign_39Assign1squeezenet/fire4/expand/3x3/BatchNorm/moving_meansave/RestoreV2:39*
use_locking(*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?*
validate_shape(*
T0
?
save/Assign_40Assign5squeezenet/fire4/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:40*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance
?
save/Assign_41Assign#squeezenet/fire4/expand/3x3/weightssave/RestoreV2:41*
use_locking(*'
_output_shapes
: ?*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights*
T0*
validate_shape(
?
save/Assign_42Assign'squeezenet/fire4/squeeze/BatchNorm/betasave/RestoreV2:42*
validate_shape(*:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta*
_output_shapes
: *
use_locking(*
T0
?
save/Assign_43Assign.squeezenet/fire4/squeeze/BatchNorm/moving_meansave/RestoreV2:43*A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
?
save/Assign_44Assign2squeezenet/fire4/squeeze/BatchNorm/moving_variancesave/RestoreV2:44*
T0*E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
use_locking(*
validate_shape(*
_output_shapes
: 
?
save/Assign_45Assign squeezenet/fire4/squeeze/weightssave/RestoreV2:45*
use_locking(*
T0*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*
validate_shape(*'
_output_shapes
:? 
?
save/Assign_46Assign*squeezenet/fire5/expand/1x1/BatchNorm/betasave/RestoreV2:46*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta*
_output_shapes	
:?
?
save/Assign_47Assign1squeezenet/fire5/expand/1x1/BatchNorm/moving_meansave/RestoreV2:47*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_48Assign5squeezenet/fire5/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:48*
use_locking(*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_49Assign#squeezenet/fire5/expand/1x1/weightssave/RestoreV2:49*
use_locking(*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
T0*
validate_shape(*'
_output_shapes
: ?
?
save/Assign_50Assign*squeezenet/fire5/expand/3x3/BatchNorm/betasave/RestoreV2:50*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta*
use_locking(*
T0*
validate_shape(
?
save/Assign_51Assign1squeezenet/fire5/expand/3x3/BatchNorm/moving_meansave/RestoreV2:51*
T0*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_52Assign5squeezenet/fire5/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:52*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance
?
save/Assign_53Assign#squeezenet/fire5/expand/3x3/weightssave/RestoreV2:53*
T0*
validate_shape(*'
_output_shapes
: ?*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
use_locking(
?
save/Assign_54Assign'squeezenet/fire5/squeeze/BatchNorm/betasave/RestoreV2:54*
validate_shape(*
T0*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
_output_shapes
: *
use_locking(
?
save/Assign_55Assign.squeezenet/fire5/squeeze/BatchNorm/moving_meansave/RestoreV2:55*
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean*
_output_shapes
: 
?
save/Assign_56Assign2squeezenet/fire5/squeeze/BatchNorm/moving_variancesave/RestoreV2:56*E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
?
save/Assign_57Assign squeezenet/fire5/squeeze/weightssave/RestoreV2:57*
validate_shape(*
T0*
use_locking(*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*'
_output_shapes
:? 
?
save/Assign_58Assign*squeezenet/fire6/expand/1x1/BatchNorm/betasave/RestoreV2:58*
_output_shapes	
:?*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta*
validate_shape(
?
save/Assign_59Assign1squeezenet/fire6/expand/1x1/BatchNorm/moving_meansave/RestoreV2:59*
validate_shape(*D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?*
T0
?
save/Assign_60Assign5squeezenet/fire6/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:60*
T0*H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_61Assign#squeezenet/fire6/expand/1x1/weightssave/RestoreV2:61*
T0*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
use_locking(*
validate_shape(*'
_output_shapes
:0?
?
save/Assign_62Assign*squeezenet/fire6/expand/3x3/BatchNorm/betasave/RestoreV2:62*
T0*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
validate_shape(*
use_locking(
?
save/Assign_63Assign1squeezenet/fire6/expand/3x3/BatchNorm/moving_meansave/RestoreV2:63*
validate_shape(*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean*
use_locking(*
T0
?
save/Assign_64Assign5squeezenet/fire6/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:64*H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_65Assign#squeezenet/fire6/expand/3x3/weightssave/RestoreV2:65*
use_locking(*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights*'
_output_shapes
:0?*
T0*
validate_shape(
?
save/Assign_66Assign'squeezenet/fire6/squeeze/BatchNorm/betasave/RestoreV2:66*
validate_shape(*:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta*
use_locking(*
_output_shapes
:0*
T0
?
save/Assign_67Assign.squeezenet/fire6/squeeze/BatchNorm/moving_meansave/RestoreV2:67*
validate_shape(*
use_locking(*
_output_shapes
:0*
T0*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean
?
save/Assign_68Assign2squeezenet/fire6/squeeze/BatchNorm/moving_variancesave/RestoreV2:68*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance*
_output_shapes
:0*
T0*
validate_shape(*
use_locking(
?
save/Assign_69Assign squeezenet/fire6/squeeze/weightssave/RestoreV2:69*
use_locking(*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*'
_output_shapes
:?0*
T0*
validate_shape(
?
save/Assign_70Assign*squeezenet/fire7/expand/1x1/BatchNorm/betasave/RestoreV2:70*
T0*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_71Assign1squeezenet/fire7/expand/1x1/BatchNorm/moving_meansave/RestoreV2:71*
validate_shape(*
use_locking(*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean*
T0
?
save/Assign_72Assign5squeezenet/fire7/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:72*
validate_shape(*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*
use_locking(
?
save/Assign_73Assign#squeezenet/fire7/expand/1x1/weightssave/RestoreV2:73*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
T0*'
_output_shapes
:0?*
use_locking(*
validate_shape(
?
save/Assign_74Assign*squeezenet/fire7/expand/3x3/BatchNorm/betasave/RestoreV2:74*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta
?
save/Assign_75Assign1squeezenet/fire7/expand/3x3/BatchNorm/moving_meansave/RestoreV2:75*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_76Assign5squeezenet/fire7/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:76*
T0*H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_77Assign#squeezenet/fire7/expand/3x3/weightssave/RestoreV2:77*
validate_shape(*'
_output_shapes
:0?*
use_locking(*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
T0
?
save/Assign_78Assign'squeezenet/fire7/squeeze/BatchNorm/betasave/RestoreV2:78*
_output_shapes
:0*
T0*
use_locking(*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta*
validate_shape(
?
save/Assign_79Assign.squeezenet/fire7/squeeze/BatchNorm/moving_meansave/RestoreV2:79*
validate_shape(*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
use_locking(*
_output_shapes
:0*
T0
?
save/Assign_80Assign2squeezenet/fire7/squeeze/BatchNorm/moving_variancesave/RestoreV2:80*
T0*
_output_shapes
:0*
validate_shape(*E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance*
use_locking(
?
save/Assign_81Assign squeezenet/fire7/squeeze/weightssave/RestoreV2:81*
use_locking(*'
_output_shapes
:?0*
validate_shape(*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*
T0
?
save/Assign_82Assign*squeezenet/fire8/expand/1x1/BatchNorm/betasave/RestoreV2:82*
use_locking(*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta*
T0
?
save/Assign_83Assign1squeezenet/fire8/expand/1x1/BatchNorm/moving_meansave/RestoreV2:83*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_84Assign5squeezenet/fire8/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:84*
use_locking(*
T0*
validate_shape(*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?
?
save/Assign_85Assign#squeezenet/fire8/expand/1x1/weightssave/RestoreV2:85*
T0*'
_output_shapes
:@?*
validate_shape(*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights*
use_locking(
?
save/Assign_86Assign*squeezenet/fire8/expand/3x3/BatchNorm/betasave/RestoreV2:86*
_output_shapes	
:?*
T0*
validate_shape(*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta*
use_locking(
?
save/Assign_87Assign1squeezenet/fire8/expand/3x3/BatchNorm/moving_meansave/RestoreV2:87*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean
?
save/Assign_88Assign5squeezenet/fire8/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:88*
T0*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
save/Assign_89Assign#squeezenet/fire8/expand/3x3/weightssave/RestoreV2:89*
validate_shape(*
use_locking(*
T0*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights
?
save/Assign_90Assign'squeezenet/fire8/squeeze/BatchNorm/betasave/RestoreV2:90*
_output_shapes
:@*
validate_shape(*:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta*
T0*
use_locking(
?
save/Assign_91Assign.squeezenet/fire8/squeeze/BatchNorm/moving_meansave/RestoreV2:91*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean
?
save/Assign_92Assign2squeezenet/fire8/squeeze/BatchNorm/moving_variancesave/RestoreV2:92*
use_locking(*
_output_shapes
:@*
validate_shape(*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance*
T0
?
save/Assign_93Assign squeezenet/fire8/squeeze/weightssave/RestoreV2:93*'
_output_shapes
:?@*
T0*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
validate_shape(*
use_locking(
?
save/Assign_94Assign*squeezenet/fire9/expand/1x1/BatchNorm/betasave/RestoreV2:94*
_output_shapes	
:?*
T0*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta*
validate_shape(*
use_locking(
?
save/Assign_95Assign1squeezenet/fire9/expand/1x1/BatchNorm/moving_meansave/RestoreV2:95*
validate_shape(*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_96Assign5squeezenet/fire9/expand/1x1/BatchNorm/moving_variancesave/RestoreV2:96*
validate_shape(*
use_locking(*H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*
T0*
_output_shapes	
:?
?
save/Assign_97Assign#squeezenet/fire9/expand/1x1/weightssave/RestoreV2:97*
T0*
use_locking(*'
_output_shapes
:@?*
validate_shape(*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
?
save/Assign_98Assign*squeezenet/fire9/expand/3x3/BatchNorm/betasave/RestoreV2:98*
use_locking(*=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_99Assign1squeezenet/fire9/expand/3x3/BatchNorm/moving_meansave/RestoreV2:99*D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save/Assign_100Assign5squeezenet/fire9/expand/3x3/BatchNorm/moving_variancesave/RestoreV2:100*
validate_shape(*
T0*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance*
use_locking(
?
save/Assign_101Assign#squeezenet/fire9/expand/3x3/weightssave/RestoreV2:101*
validate_shape(*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
use_locking(*'
_output_shapes
:@?*
T0
?
save/Assign_102Assign'squeezenet/fire9/squeeze/BatchNorm/betasave/RestoreV2:102*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta
?
save/Assign_103Assign.squeezenet/fire9/squeeze/BatchNorm/moving_meansave/RestoreV2:103*
_output_shapes
:@*
T0*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean*
validate_shape(*
use_locking(
?
save/Assign_104Assign2squeezenet/fire9/squeeze/BatchNorm/moving_variancesave/RestoreV2:104*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
?
save/Assign_105Assign squeezenet/fire9/squeeze/weightssave/RestoreV2:105*'
_output_shapes
:?@*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights*
T0*
use_locking(*
validate_shape(
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
[
save_1/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_21853470c42849d184d4732e5c5064da/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?%
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:j*?%
value?%B?%jB$squeezenet/Bottleneck/BatchNorm/betaB+squeezenet/Bottleneck/BatchNorm/moving_meanB/squeezenet/Bottleneck/BatchNorm/moving_varianceBsqueezenet/Bottleneck/weightsBsqueezenet/conv1/BatchNorm/betaB&squeezenet/conv1/BatchNorm/moving_meanB*squeezenet/conv1/BatchNorm/moving_varianceBsqueezenet/conv1/weightsBsqueezenet/conv10/biasesBsqueezenet/conv10/weightsB*squeezenet/fire2/expand/1x1/BatchNorm/betaB1squeezenet/fire2/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire2/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire2/expand/1x1/weightsB*squeezenet/fire2/expand/3x3/BatchNorm/betaB1squeezenet/fire2/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire2/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire2/expand/3x3/weightsB'squeezenet/fire2/squeeze/BatchNorm/betaB.squeezenet/fire2/squeeze/BatchNorm/moving_meanB2squeezenet/fire2/squeeze/BatchNorm/moving_varianceB squeezenet/fire2/squeeze/weightsB*squeezenet/fire3/expand/1x1/BatchNorm/betaB1squeezenet/fire3/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire3/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire3/expand/1x1/weightsB*squeezenet/fire3/expand/3x3/BatchNorm/betaB1squeezenet/fire3/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire3/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire3/expand/3x3/weightsB'squeezenet/fire3/squeeze/BatchNorm/betaB.squeezenet/fire3/squeeze/BatchNorm/moving_meanB2squeezenet/fire3/squeeze/BatchNorm/moving_varianceB squeezenet/fire3/squeeze/weightsB*squeezenet/fire4/expand/1x1/BatchNorm/betaB1squeezenet/fire4/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire4/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire4/expand/1x1/weightsB*squeezenet/fire4/expand/3x3/BatchNorm/betaB1squeezenet/fire4/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire4/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire4/expand/3x3/weightsB'squeezenet/fire4/squeeze/BatchNorm/betaB.squeezenet/fire4/squeeze/BatchNorm/moving_meanB2squeezenet/fire4/squeeze/BatchNorm/moving_varianceB squeezenet/fire4/squeeze/weightsB*squeezenet/fire5/expand/1x1/BatchNorm/betaB1squeezenet/fire5/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire5/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire5/expand/1x1/weightsB*squeezenet/fire5/expand/3x3/BatchNorm/betaB1squeezenet/fire5/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire5/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire5/expand/3x3/weightsB'squeezenet/fire5/squeeze/BatchNorm/betaB.squeezenet/fire5/squeeze/BatchNorm/moving_meanB2squeezenet/fire5/squeeze/BatchNorm/moving_varianceB squeezenet/fire5/squeeze/weightsB*squeezenet/fire6/expand/1x1/BatchNorm/betaB1squeezenet/fire6/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire6/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire6/expand/1x1/weightsB*squeezenet/fire6/expand/3x3/BatchNorm/betaB1squeezenet/fire6/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire6/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire6/expand/3x3/weightsB'squeezenet/fire6/squeeze/BatchNorm/betaB.squeezenet/fire6/squeeze/BatchNorm/moving_meanB2squeezenet/fire6/squeeze/BatchNorm/moving_varianceB squeezenet/fire6/squeeze/weightsB*squeezenet/fire7/expand/1x1/BatchNorm/betaB1squeezenet/fire7/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire7/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire7/expand/1x1/weightsB*squeezenet/fire7/expand/3x3/BatchNorm/betaB1squeezenet/fire7/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire7/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire7/expand/3x3/weightsB'squeezenet/fire7/squeeze/BatchNorm/betaB.squeezenet/fire7/squeeze/BatchNorm/moving_meanB2squeezenet/fire7/squeeze/BatchNorm/moving_varianceB squeezenet/fire7/squeeze/weightsB*squeezenet/fire8/expand/1x1/BatchNorm/betaB1squeezenet/fire8/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire8/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire8/expand/1x1/weightsB*squeezenet/fire8/expand/3x3/BatchNorm/betaB1squeezenet/fire8/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire8/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire8/expand/3x3/weightsB'squeezenet/fire8/squeeze/BatchNorm/betaB.squeezenet/fire8/squeeze/BatchNorm/moving_meanB2squeezenet/fire8/squeeze/BatchNorm/moving_varianceB squeezenet/fire8/squeeze/weightsB*squeezenet/fire9/expand/1x1/BatchNorm/betaB1squeezenet/fire9/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire9/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire9/expand/1x1/weightsB*squeezenet/fire9/expand/3x3/BatchNorm/betaB1squeezenet/fire9/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire9/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire9/expand/3x3/weightsB'squeezenet/fire9/squeeze/BatchNorm/betaB.squeezenet/fire9/squeeze/BatchNorm/moving_meanB2squeezenet/fire9/squeeze/BatchNorm/moving_varianceB squeezenet/fire9/squeeze/weights
?
save_1/SaveV2/shape_and_slicesConst*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:j*
dtype0
?&
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices$squeezenet/Bottleneck/BatchNorm/beta+squeezenet/Bottleneck/BatchNorm/moving_mean/squeezenet/Bottleneck/BatchNorm/moving_variancesqueezenet/Bottleneck/weightssqueezenet/conv1/BatchNorm/beta&squeezenet/conv1/BatchNorm/moving_mean*squeezenet/conv1/BatchNorm/moving_variancesqueezenet/conv1/weightssqueezenet/conv10/biasessqueezenet/conv10/weights*squeezenet/fire2/expand/1x1/BatchNorm/beta1squeezenet/fire2/expand/1x1/BatchNorm/moving_mean5squeezenet/fire2/expand/1x1/BatchNorm/moving_variance#squeezenet/fire2/expand/1x1/weights*squeezenet/fire2/expand/3x3/BatchNorm/beta1squeezenet/fire2/expand/3x3/BatchNorm/moving_mean5squeezenet/fire2/expand/3x3/BatchNorm/moving_variance#squeezenet/fire2/expand/3x3/weights'squeezenet/fire2/squeeze/BatchNorm/beta.squeezenet/fire2/squeeze/BatchNorm/moving_mean2squeezenet/fire2/squeeze/BatchNorm/moving_variance squeezenet/fire2/squeeze/weights*squeezenet/fire3/expand/1x1/BatchNorm/beta1squeezenet/fire3/expand/1x1/BatchNorm/moving_mean5squeezenet/fire3/expand/1x1/BatchNorm/moving_variance#squeezenet/fire3/expand/1x1/weights*squeezenet/fire3/expand/3x3/BatchNorm/beta1squeezenet/fire3/expand/3x3/BatchNorm/moving_mean5squeezenet/fire3/expand/3x3/BatchNorm/moving_variance#squeezenet/fire3/expand/3x3/weights'squeezenet/fire3/squeeze/BatchNorm/beta.squeezenet/fire3/squeeze/BatchNorm/moving_mean2squeezenet/fire3/squeeze/BatchNorm/moving_variance squeezenet/fire3/squeeze/weights*squeezenet/fire4/expand/1x1/BatchNorm/beta1squeezenet/fire4/expand/1x1/BatchNorm/moving_mean5squeezenet/fire4/expand/1x1/BatchNorm/moving_variance#squeezenet/fire4/expand/1x1/weights*squeezenet/fire4/expand/3x3/BatchNorm/beta1squeezenet/fire4/expand/3x3/BatchNorm/moving_mean5squeezenet/fire4/expand/3x3/BatchNorm/moving_variance#squeezenet/fire4/expand/3x3/weights'squeezenet/fire4/squeeze/BatchNorm/beta.squeezenet/fire4/squeeze/BatchNorm/moving_mean2squeezenet/fire4/squeeze/BatchNorm/moving_variance squeezenet/fire4/squeeze/weights*squeezenet/fire5/expand/1x1/BatchNorm/beta1squeezenet/fire5/expand/1x1/BatchNorm/moving_mean5squeezenet/fire5/expand/1x1/BatchNorm/moving_variance#squeezenet/fire5/expand/1x1/weights*squeezenet/fire5/expand/3x3/BatchNorm/beta1squeezenet/fire5/expand/3x3/BatchNorm/moving_mean5squeezenet/fire5/expand/3x3/BatchNorm/moving_variance#squeezenet/fire5/expand/3x3/weights'squeezenet/fire5/squeeze/BatchNorm/beta.squeezenet/fire5/squeeze/BatchNorm/moving_mean2squeezenet/fire5/squeeze/BatchNorm/moving_variance squeezenet/fire5/squeeze/weights*squeezenet/fire6/expand/1x1/BatchNorm/beta1squeezenet/fire6/expand/1x1/BatchNorm/moving_mean5squeezenet/fire6/expand/1x1/BatchNorm/moving_variance#squeezenet/fire6/expand/1x1/weights*squeezenet/fire6/expand/3x3/BatchNorm/beta1squeezenet/fire6/expand/3x3/BatchNorm/moving_mean5squeezenet/fire6/expand/3x3/BatchNorm/moving_variance#squeezenet/fire6/expand/3x3/weights'squeezenet/fire6/squeeze/BatchNorm/beta.squeezenet/fire6/squeeze/BatchNorm/moving_mean2squeezenet/fire6/squeeze/BatchNorm/moving_variance squeezenet/fire6/squeeze/weights*squeezenet/fire7/expand/1x1/BatchNorm/beta1squeezenet/fire7/expand/1x1/BatchNorm/moving_mean5squeezenet/fire7/expand/1x1/BatchNorm/moving_variance#squeezenet/fire7/expand/1x1/weights*squeezenet/fire7/expand/3x3/BatchNorm/beta1squeezenet/fire7/expand/3x3/BatchNorm/moving_mean5squeezenet/fire7/expand/3x3/BatchNorm/moving_variance#squeezenet/fire7/expand/3x3/weights'squeezenet/fire7/squeeze/BatchNorm/beta.squeezenet/fire7/squeeze/BatchNorm/moving_mean2squeezenet/fire7/squeeze/BatchNorm/moving_variance squeezenet/fire7/squeeze/weights*squeezenet/fire8/expand/1x1/BatchNorm/beta1squeezenet/fire8/expand/1x1/BatchNorm/moving_mean5squeezenet/fire8/expand/1x1/BatchNorm/moving_variance#squeezenet/fire8/expand/1x1/weights*squeezenet/fire8/expand/3x3/BatchNorm/beta1squeezenet/fire8/expand/3x3/BatchNorm/moving_mean5squeezenet/fire8/expand/3x3/BatchNorm/moving_variance#squeezenet/fire8/expand/3x3/weights'squeezenet/fire8/squeeze/BatchNorm/beta.squeezenet/fire8/squeeze/BatchNorm/moving_mean2squeezenet/fire8/squeeze/BatchNorm/moving_variance squeezenet/fire8/squeeze/weights*squeezenet/fire9/expand/1x1/BatchNorm/beta1squeezenet/fire9/expand/1x1/BatchNorm/moving_mean5squeezenet/fire9/expand/1x1/BatchNorm/moving_variance#squeezenet/fire9/expand/1x1/weights*squeezenet/fire9/expand/3x3/BatchNorm/beta1squeezenet/fire9/expand/3x3/BatchNorm/moving_mean5squeezenet/fire9/expand/3x3/BatchNorm/moving_variance#squeezenet/fire9/expand/3x3/weights'squeezenet/fire9/squeeze/BatchNorm/beta.squeezenet/fire9/squeeze/BatchNorm/moving_mean2squeezenet/fire9/squeeze/BatchNorm/moving_variance squeezenet/fire9/squeeze/weights*x
dtypesn
l2j
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
N*
_output_shapes
:*
T0
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
?%
save_1/RestoreV2/tensor_namesConst*?%
value?%B?%jB$squeezenet/Bottleneck/BatchNorm/betaB+squeezenet/Bottleneck/BatchNorm/moving_meanB/squeezenet/Bottleneck/BatchNorm/moving_varianceBsqueezenet/Bottleneck/weightsBsqueezenet/conv1/BatchNorm/betaB&squeezenet/conv1/BatchNorm/moving_meanB*squeezenet/conv1/BatchNorm/moving_varianceBsqueezenet/conv1/weightsBsqueezenet/conv10/biasesBsqueezenet/conv10/weightsB*squeezenet/fire2/expand/1x1/BatchNorm/betaB1squeezenet/fire2/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire2/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire2/expand/1x1/weightsB*squeezenet/fire2/expand/3x3/BatchNorm/betaB1squeezenet/fire2/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire2/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire2/expand/3x3/weightsB'squeezenet/fire2/squeeze/BatchNorm/betaB.squeezenet/fire2/squeeze/BatchNorm/moving_meanB2squeezenet/fire2/squeeze/BatchNorm/moving_varianceB squeezenet/fire2/squeeze/weightsB*squeezenet/fire3/expand/1x1/BatchNorm/betaB1squeezenet/fire3/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire3/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire3/expand/1x1/weightsB*squeezenet/fire3/expand/3x3/BatchNorm/betaB1squeezenet/fire3/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire3/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire3/expand/3x3/weightsB'squeezenet/fire3/squeeze/BatchNorm/betaB.squeezenet/fire3/squeeze/BatchNorm/moving_meanB2squeezenet/fire3/squeeze/BatchNorm/moving_varianceB squeezenet/fire3/squeeze/weightsB*squeezenet/fire4/expand/1x1/BatchNorm/betaB1squeezenet/fire4/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire4/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire4/expand/1x1/weightsB*squeezenet/fire4/expand/3x3/BatchNorm/betaB1squeezenet/fire4/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire4/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire4/expand/3x3/weightsB'squeezenet/fire4/squeeze/BatchNorm/betaB.squeezenet/fire4/squeeze/BatchNorm/moving_meanB2squeezenet/fire4/squeeze/BatchNorm/moving_varianceB squeezenet/fire4/squeeze/weightsB*squeezenet/fire5/expand/1x1/BatchNorm/betaB1squeezenet/fire5/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire5/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire5/expand/1x1/weightsB*squeezenet/fire5/expand/3x3/BatchNorm/betaB1squeezenet/fire5/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire5/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire5/expand/3x3/weightsB'squeezenet/fire5/squeeze/BatchNorm/betaB.squeezenet/fire5/squeeze/BatchNorm/moving_meanB2squeezenet/fire5/squeeze/BatchNorm/moving_varianceB squeezenet/fire5/squeeze/weightsB*squeezenet/fire6/expand/1x1/BatchNorm/betaB1squeezenet/fire6/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire6/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire6/expand/1x1/weightsB*squeezenet/fire6/expand/3x3/BatchNorm/betaB1squeezenet/fire6/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire6/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire6/expand/3x3/weightsB'squeezenet/fire6/squeeze/BatchNorm/betaB.squeezenet/fire6/squeeze/BatchNorm/moving_meanB2squeezenet/fire6/squeeze/BatchNorm/moving_varianceB squeezenet/fire6/squeeze/weightsB*squeezenet/fire7/expand/1x1/BatchNorm/betaB1squeezenet/fire7/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire7/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire7/expand/1x1/weightsB*squeezenet/fire7/expand/3x3/BatchNorm/betaB1squeezenet/fire7/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire7/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire7/expand/3x3/weightsB'squeezenet/fire7/squeeze/BatchNorm/betaB.squeezenet/fire7/squeeze/BatchNorm/moving_meanB2squeezenet/fire7/squeeze/BatchNorm/moving_varianceB squeezenet/fire7/squeeze/weightsB*squeezenet/fire8/expand/1x1/BatchNorm/betaB1squeezenet/fire8/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire8/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire8/expand/1x1/weightsB*squeezenet/fire8/expand/3x3/BatchNorm/betaB1squeezenet/fire8/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire8/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire8/expand/3x3/weightsB'squeezenet/fire8/squeeze/BatchNorm/betaB.squeezenet/fire8/squeeze/BatchNorm/moving_meanB2squeezenet/fire8/squeeze/BatchNorm/moving_varianceB squeezenet/fire8/squeeze/weightsB*squeezenet/fire9/expand/1x1/BatchNorm/betaB1squeezenet/fire9/expand/1x1/BatchNorm/moving_meanB5squeezenet/fire9/expand/1x1/BatchNorm/moving_varianceB#squeezenet/fire9/expand/1x1/weightsB*squeezenet/fire9/expand/3x3/BatchNorm/betaB1squeezenet/fire9/expand/3x3/BatchNorm/moving_meanB5squeezenet/fire9/expand/3x3/BatchNorm/moving_varianceB#squeezenet/fire9/expand/3x3/weightsB'squeezenet/fire9/squeeze/BatchNorm/betaB.squeezenet/fire9/squeeze/BatchNorm/moving_meanB2squeezenet/fire9/squeeze/BatchNorm/moving_varianceB squeezenet/fire9/squeeze/weights*
dtype0*
_output_shapes
:j
?
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:j*?
value?B?jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*x
dtypesn
l2j*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save_1/AssignAssign$squeezenet/Bottleneck/BatchNorm/betasave_1/RestoreV2*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*7
_class-
+)loc:@squeezenet/Bottleneck/BatchNorm/beta
?
save_1/Assign_1Assign+squeezenet/Bottleneck/BatchNorm/moving_meansave_1/RestoreV2:1*
validate_shape(*>
_class4
20loc:@squeezenet/Bottleneck/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?*
T0
?
save_1/Assign_2Assign/squeezenet/Bottleneck/BatchNorm/moving_variancesave_1/RestoreV2:2*
_output_shapes	
:?*
validate_shape(*
T0*B
_class8
64loc:@squeezenet/Bottleneck/BatchNorm/moving_variance*
use_locking(
?
save_1/Assign_3Assignsqueezenet/Bottleneck/weightssave_1/RestoreV2:3*
T0*0
_class&
$"loc:@squeezenet/Bottleneck/weights* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_1/Assign_4Assignsqueezenet/conv1/BatchNorm/betasave_1/RestoreV2:4*
_output_shapes
:`*
use_locking(*2
_class(
&$loc:@squeezenet/conv1/BatchNorm/beta*
validate_shape(*
T0
?
save_1/Assign_5Assign&squeezenet/conv1/BatchNorm/moving_meansave_1/RestoreV2:5*
_output_shapes
:`*
validate_shape(*9
_class/
-+loc:@squeezenet/conv1/BatchNorm/moving_mean*
T0*
use_locking(
?
save_1/Assign_6Assign*squeezenet/conv1/BatchNorm/moving_variancesave_1/RestoreV2:6*
validate_shape(*
_output_shapes
:`*
T0*=
_class3
1/loc:@squeezenet/conv1/BatchNorm/moving_variance*
use_locking(
?
save_1/Assign_7Assignsqueezenet/conv1/weightssave_1/RestoreV2:7*+
_class!
loc:@squeezenet/conv1/weights*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:`
?
save_1/Assign_8Assignsqueezenet/conv10/biasessave_1/RestoreV2:8*+
_class!
loc:@squeezenet/conv10/biases*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_9Assignsqueezenet/conv10/weightssave_1/RestoreV2:9*
T0*(
_output_shapes
:??*,
_class"
 loc:@squeezenet/conv10/weights*
use_locking(*
validate_shape(
?
save_1/Assign_10Assign*squeezenet/fire2/expand/1x1/BatchNorm/betasave_1/RestoreV2:10*
validate_shape(*=
_class3
1/loc:@squeezenet/fire2/expand/1x1/BatchNorm/beta*
T0*
_output_shapes
:@*
use_locking(
?
save_1/Assign_11Assign1squeezenet/fire2/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:11*
validate_shape(*
T0*
use_locking(*D
_class:
86loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_mean*
_output_shapes
:@
?
save_1/Assign_12Assign5squeezenet/fire2/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:12*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0*H
_class>
<:loc:@squeezenet/fire2/expand/1x1/BatchNorm/moving_variance
?
save_1/Assign_13Assign#squeezenet/fire2/expand/1x1/weightssave_1/RestoreV2:13*
use_locking(*
validate_shape(*&
_output_shapes
:@*6
_class,
*(loc:@squeezenet/fire2/expand/1x1/weights*
T0
?
save_1/Assign_14Assign*squeezenet/fire2/expand/3x3/BatchNorm/betasave_1/RestoreV2:14*
T0*
_output_shapes
:@*
validate_shape(*=
_class3
1/loc:@squeezenet/fire2/expand/3x3/BatchNorm/beta*
use_locking(
?
save_1/Assign_15Assign1squeezenet/fire2/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:15*D
_class:
86loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_mean*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
?
save_1/Assign_16Assign5squeezenet/fire2/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:16*
T0*
validate_shape(*
use_locking(*H
_class>
<:loc:@squeezenet/fire2/expand/3x3/BatchNorm/moving_variance*
_output_shapes
:@
?
save_1/Assign_17Assign#squeezenet/fire2/expand/3x3/weightssave_1/RestoreV2:17*6
_class,
*(loc:@squeezenet/fire2/expand/3x3/weights*
use_locking(*
T0*
validate_shape(*&
_output_shapes
:@
?
save_1/Assign_18Assign'squeezenet/fire2/squeeze/BatchNorm/betasave_1/RestoreV2:18*
use_locking(*:
_class0
.,loc:@squeezenet/fire2/squeeze/BatchNorm/beta*
T0*
_output_shapes
:*
validate_shape(
?
save_1/Assign_19Assign.squeezenet/fire2/squeeze/BatchNorm/moving_meansave_1/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*A
_class7
53loc:@squeezenet/fire2/squeeze/BatchNorm/moving_mean
?
save_1/Assign_20Assign2squeezenet/fire2/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:20*
_output_shapes
:*
T0*E
_class;
97loc:@squeezenet/fire2/squeeze/BatchNorm/moving_variance*
use_locking(*
validate_shape(
?
save_1/Assign_21Assign squeezenet/fire2/squeeze/weightssave_1/RestoreV2:21*
use_locking(*3
_class)
'%loc:@squeezenet/fire2/squeeze/weights*
validate_shape(*
T0*&
_output_shapes
:`
?
save_1/Assign_22Assign*squeezenet/fire3/expand/1x1/BatchNorm/betasave_1/RestoreV2:22*
T0*
_output_shapes
:@*
validate_shape(*=
_class3
1/loc:@squeezenet/fire3/expand/1x1/BatchNorm/beta*
use_locking(
?
save_1/Assign_23Assign1squeezenet/fire3/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:23*
T0*
validate_shape(*D
_class:
86loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_mean*
_output_shapes
:@*
use_locking(
?
save_1/Assign_24Assign5squeezenet/fire3/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*H
_class>
<:loc:@squeezenet/fire3/expand/1x1/BatchNorm/moving_variance
?
save_1/Assign_25Assign#squeezenet/fire3/expand/1x1/weightssave_1/RestoreV2:25*
T0*6
_class,
*(loc:@squeezenet/fire3/expand/1x1/weights*&
_output_shapes
:@*
validate_shape(*
use_locking(
?
save_1/Assign_26Assign*squeezenet/fire3/expand/3x3/BatchNorm/betasave_1/RestoreV2:26*
T0*
_output_shapes
:@*
validate_shape(*=
_class3
1/loc:@squeezenet/fire3/expand/3x3/BatchNorm/beta*
use_locking(
?
save_1/Assign_27Assign1squeezenet/fire3/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:27*
use_locking(*D
_class:
86loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@*
T0
?
save_1/Assign_28Assign5squeezenet/fire3/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:28*
_output_shapes
:@*H
_class>
<:loc:@squeezenet/fire3/expand/3x3/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_29Assign#squeezenet/fire3/expand/3x3/weightssave_1/RestoreV2:29*6
_class,
*(loc:@squeezenet/fire3/expand/3x3/weights*&
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_30Assign'squeezenet/fire3/squeeze/BatchNorm/betasave_1/RestoreV2:30*
use_locking(*:
_class0
.,loc:@squeezenet/fire3/squeeze/BatchNorm/beta*
_output_shapes
:*
validate_shape(*
T0
?
save_1/Assign_31Assign.squeezenet/fire3/squeeze/BatchNorm/moving_meansave_1/RestoreV2:31*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*A
_class7
53loc:@squeezenet/fire3/squeeze/BatchNorm/moving_mean
?
save_1/Assign_32Assign2squeezenet/fire3/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:32*
validate_shape(*E
_class;
97loc:@squeezenet/fire3/squeeze/BatchNorm/moving_variance*
_output_shapes
:*
T0*
use_locking(
?
save_1/Assign_33Assign squeezenet/fire3/squeeze/weightssave_1/RestoreV2:33*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:?*3
_class)
'%loc:@squeezenet/fire3/squeeze/weights
?
save_1/Assign_34Assign*squeezenet/fire4/expand/1x1/BatchNorm/betasave_1/RestoreV2:34*
use_locking(*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire4/expand/1x1/BatchNorm/beta*
T0
?
save_1/Assign_35Assign1squeezenet/fire4/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:35*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_mean
?
save_1/Assign_36Assign5squeezenet/fire4/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:36*H
_class>
<:loc:@squeezenet/fire4/expand/1x1/BatchNorm/moving_variance*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_37Assign#squeezenet/fire4/expand/1x1/weightssave_1/RestoreV2:37*6
_class,
*(loc:@squeezenet/fire4/expand/1x1/weights*
use_locking(*'
_output_shapes
: ?*
validate_shape(*
T0
?
save_1/Assign_38Assign*squeezenet/fire4/expand/3x3/BatchNorm/betasave_1/RestoreV2:38*
T0*
validate_shape(*=
_class3
1/loc:@squeezenet/fire4/expand/3x3/BatchNorm/beta*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_39Assign1squeezenet/fire4/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:39*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_mean*
_output_shapes	
:?
?
save_1/Assign_40Assign5squeezenet/fire4/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:40*H
_class>
<:loc:@squeezenet/fire4/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_41Assign#squeezenet/fire4/expand/3x3/weightssave_1/RestoreV2:41*
T0*
validate_shape(*'
_output_shapes
: ?*
use_locking(*6
_class,
*(loc:@squeezenet/fire4/expand/3x3/weights
?
save_1/Assign_42Assign'squeezenet/fire4/squeeze/BatchNorm/betasave_1/RestoreV2:42*:
_class0
.,loc:@squeezenet/fire4/squeeze/BatchNorm/beta*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
?
save_1/Assign_43Assign.squeezenet/fire4/squeeze/BatchNorm/moving_meansave_1/RestoreV2:43*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *A
_class7
53loc:@squeezenet/fire4/squeeze/BatchNorm/moving_mean
?
save_1/Assign_44Assign2squeezenet/fire4/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:44*E
_class;
97loc:@squeezenet/fire4/squeeze/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
?
save_1/Assign_45Assign squeezenet/fire4/squeeze/weightssave_1/RestoreV2:45*3
_class)
'%loc:@squeezenet/fire4/squeeze/weights*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:? 
?
save_1/Assign_46Assign*squeezenet/fire5/expand/1x1/BatchNorm/betasave_1/RestoreV2:46*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire5/expand/1x1/BatchNorm/beta*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_47Assign1squeezenet/fire5/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:47*
T0*
validate_shape(*D
_class:
86loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_48Assign5squeezenet/fire5/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:48*
use_locking(*H
_class>
<:loc:@squeezenet/fire5/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_49Assign#squeezenet/fire5/expand/1x1/weightssave_1/RestoreV2:49*'
_output_shapes
: ?*
use_locking(*
validate_shape(*6
_class,
*(loc:@squeezenet/fire5/expand/1x1/weights*
T0
?
save_1/Assign_50Assign*squeezenet/fire5/expand/3x3/BatchNorm/betasave_1/RestoreV2:50*
validate_shape(*
use_locking(*=
_class3
1/loc:@squeezenet/fire5/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
T0
?
save_1/Assign_51Assign1squeezenet/fire5/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:51*D
_class:
86loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_52Assign5squeezenet/fire5/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:52*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire5/expand/3x3/BatchNorm/moving_variance
?
save_1/Assign_53Assign#squeezenet/fire5/expand/3x3/weightssave_1/RestoreV2:53*
T0*6
_class,
*(loc:@squeezenet/fire5/expand/3x3/weights*
use_locking(*
validate_shape(*'
_output_shapes
: ?
?
save_1/Assign_54Assign'squeezenet/fire5/squeeze/BatchNorm/betasave_1/RestoreV2:54*
validate_shape(*
T0*:
_class0
.,loc:@squeezenet/fire5/squeeze/BatchNorm/beta*
use_locking(*
_output_shapes
: 
?
save_1/Assign_55Assign.squeezenet/fire5/squeeze/BatchNorm/moving_meansave_1/RestoreV2:55*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*A
_class7
53loc:@squeezenet/fire5/squeeze/BatchNorm/moving_mean
?
save_1/Assign_56Assign2squeezenet/fire5/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:56*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*E
_class;
97loc:@squeezenet/fire5/squeeze/BatchNorm/moving_variance
?
save_1/Assign_57Assign squeezenet/fire5/squeeze/weightssave_1/RestoreV2:57*
use_locking(*
validate_shape(*3
_class)
'%loc:@squeezenet/fire5/squeeze/weights*
T0*'
_output_shapes
:? 
?
save_1/Assign_58Assign*squeezenet/fire6/expand/1x1/BatchNorm/betasave_1/RestoreV2:58*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@squeezenet/fire6/expand/1x1/BatchNorm/beta*
validate_shape(
?
save_1/Assign_59Assign1squeezenet/fire6/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:59*
T0*
use_locking(*
validate_shape(*D
_class:
86loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_mean*
_output_shapes	
:?
?
save_1/Assign_60Assign5squeezenet/fire6/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:60*H
_class>
<:loc:@squeezenet/fire6/expand/1x1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_61Assign#squeezenet/fire6/expand/1x1/weightssave_1/RestoreV2:61*'
_output_shapes
:0?*
T0*
use_locking(*6
_class,
*(loc:@squeezenet/fire6/expand/1x1/weights*
validate_shape(
?
save_1/Assign_62Assign*squeezenet/fire6/expand/3x3/BatchNorm/betasave_1/RestoreV2:62*
validate_shape(*=
_class3
1/loc:@squeezenet/fire6/expand/3x3/BatchNorm/beta*
use_locking(*
_output_shapes	
:?*
T0
?
save_1/Assign_63Assign1squeezenet/fire6/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:63*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(*D
_class:
86loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_mean
?
save_1/Assign_64Assign5squeezenet/fire6/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:64*
validate_shape(*
use_locking(*
T0*H
_class>
<:loc:@squeezenet/fire6/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?
?
save_1/Assign_65Assign#squeezenet/fire6/expand/3x3/weightssave_1/RestoreV2:65*'
_output_shapes
:0?*
validate_shape(*
T0*
use_locking(*6
_class,
*(loc:@squeezenet/fire6/expand/3x3/weights
?
save_1/Assign_66Assign'squeezenet/fire6/squeeze/BatchNorm/betasave_1/RestoreV2:66*
T0*
use_locking(*
validate_shape(*:
_class0
.,loc:@squeezenet/fire6/squeeze/BatchNorm/beta*
_output_shapes
:0
?
save_1/Assign_67Assign.squeezenet/fire6/squeeze/BatchNorm/moving_meansave_1/RestoreV2:67*
validate_shape(*A
_class7
53loc:@squeezenet/fire6/squeeze/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:0
?
save_1/Assign_68Assign2squeezenet/fire6/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:68*
use_locking(*
_output_shapes
:0*
validate_shape(*E
_class;
97loc:@squeezenet/fire6/squeeze/BatchNorm/moving_variance*
T0
?
save_1/Assign_69Assign squeezenet/fire6/squeeze/weightssave_1/RestoreV2:69*
use_locking(*
validate_shape(*3
_class)
'%loc:@squeezenet/fire6/squeeze/weights*
T0*'
_output_shapes
:?0
?
save_1/Assign_70Assign*squeezenet/fire7/expand/1x1/BatchNorm/betasave_1/RestoreV2:70*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire7/expand/1x1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_71Assign1squeezenet/fire7/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:71*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*D
_class:
86loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_mean
?
save_1/Assign_72Assign5squeezenet/fire7/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:72*
T0*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire7/expand/1x1/BatchNorm/moving_variance*
_output_shapes	
:?
?
save_1/Assign_73Assign#squeezenet/fire7/expand/1x1/weightssave_1/RestoreV2:73*6
_class,
*(loc:@squeezenet/fire7/expand/1x1/weights*
validate_shape(*
use_locking(*
T0*'
_output_shapes
:0?
?
save_1/Assign_74Assign*squeezenet/fire7/expand/3x3/BatchNorm/betasave_1/RestoreV2:74*=
_class3
1/loc:@squeezenet/fire7/expand/3x3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save_1/Assign_75Assign1squeezenet/fire7/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:75*
T0*D
_class:
86loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_mean*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_76Assign5squeezenet/fire7/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:76*
use_locking(*
_output_shapes	
:?*H
_class>
<:loc:@squeezenet/fire7/expand/3x3/BatchNorm/moving_variance*
validate_shape(*
T0
?
save_1/Assign_77Assign#squeezenet/fire7/expand/3x3/weightssave_1/RestoreV2:77*'
_output_shapes
:0?*6
_class,
*(loc:@squeezenet/fire7/expand/3x3/weights*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_78Assign'squeezenet/fire7/squeeze/BatchNorm/betasave_1/RestoreV2:78*
T0*
use_locking(*:
_class0
.,loc:@squeezenet/fire7/squeeze/BatchNorm/beta*
validate_shape(*
_output_shapes
:0
?
save_1/Assign_79Assign.squeezenet/fire7/squeeze/BatchNorm/moving_meansave_1/RestoreV2:79*
use_locking(*
validate_shape(*
T0*A
_class7
53loc:@squeezenet/fire7/squeeze/BatchNorm/moving_mean*
_output_shapes
:0
?
save_1/Assign_80Assign2squeezenet/fire7/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:80*
T0*E
_class;
97loc:@squeezenet/fire7/squeeze/BatchNorm/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
:0
?
save_1/Assign_81Assign squeezenet/fire7/squeeze/weightssave_1/RestoreV2:81*
validate_shape(*
use_locking(*3
_class)
'%loc:@squeezenet/fire7/squeeze/weights*'
_output_shapes
:?0*
T0
?
save_1/Assign_82Assign*squeezenet/fire8/expand/1x1/BatchNorm/betasave_1/RestoreV2:82*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*=
_class3
1/loc:@squeezenet/fire8/expand/1x1/BatchNorm/beta
?
save_1/Assign_83Assign1squeezenet/fire8/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:83*
_output_shapes	
:?*
validate_shape(*D
_class:
86loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_mean*
use_locking(*
T0
?
save_1/Assign_84Assign5squeezenet/fire8/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:84*
_output_shapes	
:?*
use_locking(*
validate_shape(*H
_class>
<:loc:@squeezenet/fire8/expand/1x1/BatchNorm/moving_variance*
T0
?
save_1/Assign_85Assign#squeezenet/fire8/expand/1x1/weightssave_1/RestoreV2:85*'
_output_shapes
:@?*
validate_shape(*
T0*
use_locking(*6
_class,
*(loc:@squeezenet/fire8/expand/1x1/weights
?
save_1/Assign_86Assign*squeezenet/fire8/expand/3x3/BatchNorm/betasave_1/RestoreV2:86*
T0*
use_locking(*=
_class3
1/loc:@squeezenet/fire8/expand/3x3/BatchNorm/beta*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_87Assign1squeezenet/fire8/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:87*
use_locking(*
T0*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_mean*
validate_shape(
?
save_1/Assign_88Assign5squeezenet/fire8/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:88*
use_locking(*H
_class>
<:loc:@squeezenet/fire8/expand/3x3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*
validate_shape(
?
save_1/Assign_89Assign#squeezenet/fire8/expand/3x3/weightssave_1/RestoreV2:89*'
_output_shapes
:@?*
T0*
use_locking(*6
_class,
*(loc:@squeezenet/fire8/expand/3x3/weights*
validate_shape(
?
save_1/Assign_90Assign'squeezenet/fire8/squeeze/BatchNorm/betasave_1/RestoreV2:90*
_output_shapes
:@*
validate_shape(*:
_class0
.,loc:@squeezenet/fire8/squeeze/BatchNorm/beta*
use_locking(*
T0
?
save_1/Assign_91Assign.squeezenet/fire8/squeeze/BatchNorm/moving_meansave_1/RestoreV2:91*
_output_shapes
:@*
T0*A
_class7
53loc:@squeezenet/fire8/squeeze/BatchNorm/moving_mean*
use_locking(*
validate_shape(
?
save_1/Assign_92Assign2squeezenet/fire8/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:92*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(*E
_class;
97loc:@squeezenet/fire8/squeeze/BatchNorm/moving_variance
?
save_1/Assign_93Assign squeezenet/fire8/squeeze/weightssave_1/RestoreV2:93*
validate_shape(*3
_class)
'%loc:@squeezenet/fire8/squeeze/weights*
T0*'
_output_shapes
:?@*
use_locking(
?
save_1/Assign_94Assign*squeezenet/fire9/expand/1x1/BatchNorm/betasave_1/RestoreV2:94*=
_class3
1/loc:@squeezenet/fire9/expand/1x1/BatchNorm/beta*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_95Assign1squeezenet/fire9/expand/1x1/BatchNorm/moving_meansave_1/RestoreV2:95*
validate_shape(*
use_locking(*
_output_shapes	
:?*D
_class:
86loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_mean*
T0
?
save_1/Assign_96Assign5squeezenet/fire9/expand/1x1/BatchNorm/moving_variancesave_1/RestoreV2:96*
use_locking(*H
_class>
<:loc:@squeezenet/fire9/expand/1x1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?*
T0
?
save_1/Assign_97Assign#squeezenet/fire9/expand/1x1/weightssave_1/RestoreV2:97*'
_output_shapes
:@?*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@squeezenet/fire9/expand/1x1/weights
?
save_1/Assign_98Assign*squeezenet/fire9/expand/3x3/BatchNorm/betasave_1/RestoreV2:98*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@squeezenet/fire9/expand/3x3/BatchNorm/beta
?
save_1/Assign_99Assign1squeezenet/fire9/expand/3x3/BatchNorm/moving_meansave_1/RestoreV2:99*D
_class:
86loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_mean*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_100Assign5squeezenet/fire9/expand/3x3/BatchNorm/moving_variancesave_1/RestoreV2:100*H
_class>
<:loc:@squeezenet/fire9/expand/3x3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_101Assign#squeezenet/fire9/expand/3x3/weightssave_1/RestoreV2:101*
T0*
use_locking(*'
_output_shapes
:@?*6
_class,
*(loc:@squeezenet/fire9/expand/3x3/weights*
validate_shape(
?
save_1/Assign_102Assign'squeezenet/fire9/squeeze/BatchNorm/betasave_1/RestoreV2:102*
use_locking(*
validate_shape(*
_output_shapes
:@*:
_class0
.,loc:@squeezenet/fire9/squeeze/BatchNorm/beta*
T0
?
save_1/Assign_103Assign.squeezenet/fire9/squeeze/BatchNorm/moving_meansave_1/RestoreV2:103*
validate_shape(*A
_class7
53loc:@squeezenet/fire9/squeeze/BatchNorm/moving_mean*
_output_shapes
:@*
T0*
use_locking(
?
save_1/Assign_104Assign2squeezenet/fire9/squeeze/BatchNorm/moving_variancesave_1/RestoreV2:104*E
_class;
97loc:@squeezenet/fire9/squeeze/BatchNorm/moving_variance*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_105Assign squeezenet/fire9/squeeze/weightssave_1/RestoreV2:105*
validate_shape(*'
_output_shapes
:?@*
T0*
use_locking(*3
_class)
'%loc:@squeezenet/fire9/squeeze/weights
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99
1
save_1/restore_allNoOp^save_1/restore_shard "?B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"ķ
trainable_variables????
?
squeezenet/conv1/weights:0squeezenet/conv1/weights/Assignsqueezenet/conv1/weights/read:025squeezenet/conv1/weights/Initializer/random_uniform:08
?
!squeezenet/conv1/BatchNorm/beta:0&squeezenet/conv1/BatchNorm/beta/Assign&squeezenet/conv1/BatchNorm/beta/read:023squeezenet/conv1/BatchNorm/beta/Initializer/zeros:08
?
(squeezenet/conv1/BatchNorm/moving_mean:0-squeezenet/conv1/BatchNorm/moving_mean/Assign-squeezenet/conv1/BatchNorm/moving_mean/read:02:squeezenet/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
,squeezenet/conv1/BatchNorm/moving_variance:01squeezenet/conv1/BatchNorm/moving_variance/Assign1squeezenet/conv1/BatchNorm/moving_variance/read:02=squeezenet/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire2/squeeze/weights:0'squeezenet/fire2/squeeze/weights/Assign'squeezenet/fire2/squeeze/weights/read:02=squeezenet/fire2/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire2/squeeze/BatchNorm/beta:0.squeezenet/fire2/squeeze/BatchNorm/beta/Assign.squeezenet/fire2/squeeze/BatchNorm/beta/read:02;squeezenet/fire2/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire2/squeeze/BatchNorm/moving_mean:05squeezenet/fire2/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire2/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire2/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire2/squeeze/BatchNorm/moving_variance:09squeezenet/fire2/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire2/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire2/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/1x1/weights:0*squeezenet/fire2/expand/1x1/weights/Assign*squeezenet/fire2/expand/1x1/weights/read:02@squeezenet/fire2/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/1x1/BatchNorm/beta:01squeezenet/fire2/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire2/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire2/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/3x3/weights:0*squeezenet/fire2/expand/3x3/weights/Assign*squeezenet/fire2/expand/3x3/weights/read:02@squeezenet/fire2/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/3x3/BatchNorm/beta:01squeezenet/fire2/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire2/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire2/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire3/squeeze/weights:0'squeezenet/fire3/squeeze/weights/Assign'squeezenet/fire3/squeeze/weights/read:02=squeezenet/fire3/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire3/squeeze/BatchNorm/beta:0.squeezenet/fire3/squeeze/BatchNorm/beta/Assign.squeezenet/fire3/squeeze/BatchNorm/beta/read:02;squeezenet/fire3/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire3/squeeze/BatchNorm/moving_mean:05squeezenet/fire3/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire3/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire3/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire3/squeeze/BatchNorm/moving_variance:09squeezenet/fire3/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire3/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire3/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/1x1/weights:0*squeezenet/fire3/expand/1x1/weights/Assign*squeezenet/fire3/expand/1x1/weights/read:02@squeezenet/fire3/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/1x1/BatchNorm/beta:01squeezenet/fire3/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire3/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire3/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/3x3/weights:0*squeezenet/fire3/expand/3x3/weights/Assign*squeezenet/fire3/expand/3x3/weights/read:02@squeezenet/fire3/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/3x3/BatchNorm/beta:01squeezenet/fire3/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire3/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire3/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire4/squeeze/weights:0'squeezenet/fire4/squeeze/weights/Assign'squeezenet/fire4/squeeze/weights/read:02=squeezenet/fire4/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire4/squeeze/BatchNorm/beta:0.squeezenet/fire4/squeeze/BatchNorm/beta/Assign.squeezenet/fire4/squeeze/BatchNorm/beta/read:02;squeezenet/fire4/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire4/squeeze/BatchNorm/moving_mean:05squeezenet/fire4/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire4/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire4/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire4/squeeze/BatchNorm/moving_variance:09squeezenet/fire4/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire4/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire4/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/1x1/weights:0*squeezenet/fire4/expand/1x1/weights/Assign*squeezenet/fire4/expand/1x1/weights/read:02@squeezenet/fire4/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/1x1/BatchNorm/beta:01squeezenet/fire4/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire4/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire4/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/3x3/weights:0*squeezenet/fire4/expand/3x3/weights/Assign*squeezenet/fire4/expand/3x3/weights/read:02@squeezenet/fire4/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/3x3/BatchNorm/beta:01squeezenet/fire4/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire4/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire4/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire5/squeeze/weights:0'squeezenet/fire5/squeeze/weights/Assign'squeezenet/fire5/squeeze/weights/read:02=squeezenet/fire5/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire5/squeeze/BatchNorm/beta:0.squeezenet/fire5/squeeze/BatchNorm/beta/Assign.squeezenet/fire5/squeeze/BatchNorm/beta/read:02;squeezenet/fire5/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire5/squeeze/BatchNorm/moving_mean:05squeezenet/fire5/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire5/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire5/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire5/squeeze/BatchNorm/moving_variance:09squeezenet/fire5/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire5/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire5/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/1x1/weights:0*squeezenet/fire5/expand/1x1/weights/Assign*squeezenet/fire5/expand/1x1/weights/read:02@squeezenet/fire5/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/1x1/BatchNorm/beta:01squeezenet/fire5/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire5/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire5/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/3x3/weights:0*squeezenet/fire5/expand/3x3/weights/Assign*squeezenet/fire5/expand/3x3/weights/read:02@squeezenet/fire5/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/3x3/BatchNorm/beta:01squeezenet/fire5/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire5/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire5/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire6/squeeze/weights:0'squeezenet/fire6/squeeze/weights/Assign'squeezenet/fire6/squeeze/weights/read:02=squeezenet/fire6/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire6/squeeze/BatchNorm/beta:0.squeezenet/fire6/squeeze/BatchNorm/beta/Assign.squeezenet/fire6/squeeze/BatchNorm/beta/read:02;squeezenet/fire6/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire6/squeeze/BatchNorm/moving_mean:05squeezenet/fire6/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire6/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire6/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire6/squeeze/BatchNorm/moving_variance:09squeezenet/fire6/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire6/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire6/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/1x1/weights:0*squeezenet/fire6/expand/1x1/weights/Assign*squeezenet/fire6/expand/1x1/weights/read:02@squeezenet/fire6/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/1x1/BatchNorm/beta:01squeezenet/fire6/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire6/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire6/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/3x3/weights:0*squeezenet/fire6/expand/3x3/weights/Assign*squeezenet/fire6/expand/3x3/weights/read:02@squeezenet/fire6/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/3x3/BatchNorm/beta:01squeezenet/fire6/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire6/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire6/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire7/squeeze/weights:0'squeezenet/fire7/squeeze/weights/Assign'squeezenet/fire7/squeeze/weights/read:02=squeezenet/fire7/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire7/squeeze/BatchNorm/beta:0.squeezenet/fire7/squeeze/BatchNorm/beta/Assign.squeezenet/fire7/squeeze/BatchNorm/beta/read:02;squeezenet/fire7/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire7/squeeze/BatchNorm/moving_mean:05squeezenet/fire7/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire7/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire7/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire7/squeeze/BatchNorm/moving_variance:09squeezenet/fire7/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire7/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire7/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/1x1/weights:0*squeezenet/fire7/expand/1x1/weights/Assign*squeezenet/fire7/expand/1x1/weights/read:02@squeezenet/fire7/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/1x1/BatchNorm/beta:01squeezenet/fire7/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire7/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire7/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/3x3/weights:0*squeezenet/fire7/expand/3x3/weights/Assign*squeezenet/fire7/expand/3x3/weights/read:02@squeezenet/fire7/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/3x3/BatchNorm/beta:01squeezenet/fire7/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire7/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire7/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire8/squeeze/weights:0'squeezenet/fire8/squeeze/weights/Assign'squeezenet/fire8/squeeze/weights/read:02=squeezenet/fire8/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire8/squeeze/BatchNorm/beta:0.squeezenet/fire8/squeeze/BatchNorm/beta/Assign.squeezenet/fire8/squeeze/BatchNorm/beta/read:02;squeezenet/fire8/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire8/squeeze/BatchNorm/moving_mean:05squeezenet/fire8/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire8/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire8/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire8/squeeze/BatchNorm/moving_variance:09squeezenet/fire8/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire8/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire8/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/1x1/weights:0*squeezenet/fire8/expand/1x1/weights/Assign*squeezenet/fire8/expand/1x1/weights/read:02@squeezenet/fire8/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/1x1/BatchNorm/beta:01squeezenet/fire8/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire8/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire8/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/3x3/weights:0*squeezenet/fire8/expand/3x3/weights/Assign*squeezenet/fire8/expand/3x3/weights/read:02@squeezenet/fire8/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/3x3/BatchNorm/beta:01squeezenet/fire8/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire8/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire8/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire9/squeeze/weights:0'squeezenet/fire9/squeeze/weights/Assign'squeezenet/fire9/squeeze/weights/read:02=squeezenet/fire9/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire9/squeeze/BatchNorm/beta:0.squeezenet/fire9/squeeze/BatchNorm/beta/Assign.squeezenet/fire9/squeeze/BatchNorm/beta/read:02;squeezenet/fire9/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire9/squeeze/BatchNorm/moving_mean:05squeezenet/fire9/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire9/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire9/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire9/squeeze/BatchNorm/moving_variance:09squeezenet/fire9/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire9/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire9/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/1x1/weights:0*squeezenet/fire9/expand/1x1/weights/Assign*squeezenet/fire9/expand/1x1/weights/read:02@squeezenet/fire9/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/1x1/BatchNorm/beta:01squeezenet/fire9/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire9/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire9/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/3x3/weights:0*squeezenet/fire9/expand/3x3/weights/Assign*squeezenet/fire9/expand/3x3/weights/read:02@squeezenet/fire9/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/3x3/BatchNorm/beta:01squeezenet/fire9/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire9/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire9/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
squeezenet/conv10/weights:0 squeezenet/conv10/weights/Assign squeezenet/conv10/weights/read:026squeezenet/conv10/weights/Initializer/random_uniform:08
?
squeezenet/conv10/biases:0squeezenet/conv10/biases/Assignsqueezenet/conv10/biases/read:02,squeezenet/conv10/biases/Initializer/zeros:08
?
squeezenet/Bottleneck/weights:0$squeezenet/Bottleneck/weights/Assign$squeezenet/Bottleneck/weights/read:02:squeezenet/Bottleneck/weights/Initializer/random_uniform:08
?
&squeezenet/Bottleneck/BatchNorm/beta:0+squeezenet/Bottleneck/BatchNorm/beta/Assign+squeezenet/Bottleneck/BatchNorm/beta/read:028squeezenet/Bottleneck/BatchNorm/beta/Initializer/zeros:08
?
-squeezenet/Bottleneck/BatchNorm/moving_mean:02squeezenet/Bottleneck/BatchNorm/moving_mean/Assign2squeezenet/Bottleneck/BatchNorm/moving_mean/read:02?squeezenet/Bottleneck/BatchNorm/moving_mean/Initializer/zeros:0
?
1squeezenet/Bottleneck/BatchNorm/moving_variance:06squeezenet/Bottleneck/BatchNorm/moving_variance/Assign6squeezenet/Bottleneck/BatchNorm/moving_variance/read:02Bsqueezenet/Bottleneck/BatchNorm/moving_variance/Initializer/ones:0"??
model_variables????
?
squeezenet/conv1/weights:0squeezenet/conv1/weights/Assignsqueezenet/conv1/weights/read:025squeezenet/conv1/weights/Initializer/random_uniform:08
?
!squeezenet/conv1/BatchNorm/beta:0&squeezenet/conv1/BatchNorm/beta/Assign&squeezenet/conv1/BatchNorm/beta/read:023squeezenet/conv1/BatchNorm/beta/Initializer/zeros:08
?
(squeezenet/conv1/BatchNorm/moving_mean:0-squeezenet/conv1/BatchNorm/moving_mean/Assign-squeezenet/conv1/BatchNorm/moving_mean/read:02:squeezenet/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
,squeezenet/conv1/BatchNorm/moving_variance:01squeezenet/conv1/BatchNorm/moving_variance/Assign1squeezenet/conv1/BatchNorm/moving_variance/read:02=squeezenet/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire2/squeeze/weights:0'squeezenet/fire2/squeeze/weights/Assign'squeezenet/fire2/squeeze/weights/read:02=squeezenet/fire2/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire2/squeeze/BatchNorm/beta:0.squeezenet/fire2/squeeze/BatchNorm/beta/Assign.squeezenet/fire2/squeeze/BatchNorm/beta/read:02;squeezenet/fire2/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire2/squeeze/BatchNorm/moving_mean:05squeezenet/fire2/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire2/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire2/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire2/squeeze/BatchNorm/moving_variance:09squeezenet/fire2/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire2/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire2/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/1x1/weights:0*squeezenet/fire2/expand/1x1/weights/Assign*squeezenet/fire2/expand/1x1/weights/read:02@squeezenet/fire2/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/1x1/BatchNorm/beta:01squeezenet/fire2/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire2/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire2/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/3x3/weights:0*squeezenet/fire2/expand/3x3/weights/Assign*squeezenet/fire2/expand/3x3/weights/read:02@squeezenet/fire2/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/3x3/BatchNorm/beta:01squeezenet/fire2/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire2/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire2/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire3/squeeze/weights:0'squeezenet/fire3/squeeze/weights/Assign'squeezenet/fire3/squeeze/weights/read:02=squeezenet/fire3/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire3/squeeze/BatchNorm/beta:0.squeezenet/fire3/squeeze/BatchNorm/beta/Assign.squeezenet/fire3/squeeze/BatchNorm/beta/read:02;squeezenet/fire3/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire3/squeeze/BatchNorm/moving_mean:05squeezenet/fire3/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire3/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire3/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire3/squeeze/BatchNorm/moving_variance:09squeezenet/fire3/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire3/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire3/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/1x1/weights:0*squeezenet/fire3/expand/1x1/weights/Assign*squeezenet/fire3/expand/1x1/weights/read:02@squeezenet/fire3/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/1x1/BatchNorm/beta:01squeezenet/fire3/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire3/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire3/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/3x3/weights:0*squeezenet/fire3/expand/3x3/weights/Assign*squeezenet/fire3/expand/3x3/weights/read:02@squeezenet/fire3/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/3x3/BatchNorm/beta:01squeezenet/fire3/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire3/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire3/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire4/squeeze/weights:0'squeezenet/fire4/squeeze/weights/Assign'squeezenet/fire4/squeeze/weights/read:02=squeezenet/fire4/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire4/squeeze/BatchNorm/beta:0.squeezenet/fire4/squeeze/BatchNorm/beta/Assign.squeezenet/fire4/squeeze/BatchNorm/beta/read:02;squeezenet/fire4/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire4/squeeze/BatchNorm/moving_mean:05squeezenet/fire4/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire4/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire4/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire4/squeeze/BatchNorm/moving_variance:09squeezenet/fire4/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire4/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire4/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/1x1/weights:0*squeezenet/fire4/expand/1x1/weights/Assign*squeezenet/fire4/expand/1x1/weights/read:02@squeezenet/fire4/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/1x1/BatchNorm/beta:01squeezenet/fire4/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire4/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire4/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/3x3/weights:0*squeezenet/fire4/expand/3x3/weights/Assign*squeezenet/fire4/expand/3x3/weights/read:02@squeezenet/fire4/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/3x3/BatchNorm/beta:01squeezenet/fire4/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire4/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire4/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire5/squeeze/weights:0'squeezenet/fire5/squeeze/weights/Assign'squeezenet/fire5/squeeze/weights/read:02=squeezenet/fire5/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire5/squeeze/BatchNorm/beta:0.squeezenet/fire5/squeeze/BatchNorm/beta/Assign.squeezenet/fire5/squeeze/BatchNorm/beta/read:02;squeezenet/fire5/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire5/squeeze/BatchNorm/moving_mean:05squeezenet/fire5/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire5/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire5/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire5/squeeze/BatchNorm/moving_variance:09squeezenet/fire5/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire5/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire5/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/1x1/weights:0*squeezenet/fire5/expand/1x1/weights/Assign*squeezenet/fire5/expand/1x1/weights/read:02@squeezenet/fire5/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/1x1/BatchNorm/beta:01squeezenet/fire5/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire5/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire5/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/3x3/weights:0*squeezenet/fire5/expand/3x3/weights/Assign*squeezenet/fire5/expand/3x3/weights/read:02@squeezenet/fire5/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/3x3/BatchNorm/beta:01squeezenet/fire5/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire5/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire5/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire6/squeeze/weights:0'squeezenet/fire6/squeeze/weights/Assign'squeezenet/fire6/squeeze/weights/read:02=squeezenet/fire6/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire6/squeeze/BatchNorm/beta:0.squeezenet/fire6/squeeze/BatchNorm/beta/Assign.squeezenet/fire6/squeeze/BatchNorm/beta/read:02;squeezenet/fire6/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire6/squeeze/BatchNorm/moving_mean:05squeezenet/fire6/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire6/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire6/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire6/squeeze/BatchNorm/moving_variance:09squeezenet/fire6/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire6/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire6/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/1x1/weights:0*squeezenet/fire6/expand/1x1/weights/Assign*squeezenet/fire6/expand/1x1/weights/read:02@squeezenet/fire6/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/1x1/BatchNorm/beta:01squeezenet/fire6/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire6/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire6/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/3x3/weights:0*squeezenet/fire6/expand/3x3/weights/Assign*squeezenet/fire6/expand/3x3/weights/read:02@squeezenet/fire6/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/3x3/BatchNorm/beta:01squeezenet/fire6/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire6/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire6/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire7/squeeze/weights:0'squeezenet/fire7/squeeze/weights/Assign'squeezenet/fire7/squeeze/weights/read:02=squeezenet/fire7/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire7/squeeze/BatchNorm/beta:0.squeezenet/fire7/squeeze/BatchNorm/beta/Assign.squeezenet/fire7/squeeze/BatchNorm/beta/read:02;squeezenet/fire7/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire7/squeeze/BatchNorm/moving_mean:05squeezenet/fire7/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire7/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire7/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire7/squeeze/BatchNorm/moving_variance:09squeezenet/fire7/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire7/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire7/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/1x1/weights:0*squeezenet/fire7/expand/1x1/weights/Assign*squeezenet/fire7/expand/1x1/weights/read:02@squeezenet/fire7/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/1x1/BatchNorm/beta:01squeezenet/fire7/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire7/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire7/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/3x3/weights:0*squeezenet/fire7/expand/3x3/weights/Assign*squeezenet/fire7/expand/3x3/weights/read:02@squeezenet/fire7/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/3x3/BatchNorm/beta:01squeezenet/fire7/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire7/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire7/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire8/squeeze/weights:0'squeezenet/fire8/squeeze/weights/Assign'squeezenet/fire8/squeeze/weights/read:02=squeezenet/fire8/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire8/squeeze/BatchNorm/beta:0.squeezenet/fire8/squeeze/BatchNorm/beta/Assign.squeezenet/fire8/squeeze/BatchNorm/beta/read:02;squeezenet/fire8/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire8/squeeze/BatchNorm/moving_mean:05squeezenet/fire8/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire8/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire8/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire8/squeeze/BatchNorm/moving_variance:09squeezenet/fire8/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire8/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire8/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/1x1/weights:0*squeezenet/fire8/expand/1x1/weights/Assign*squeezenet/fire8/expand/1x1/weights/read:02@squeezenet/fire8/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/1x1/BatchNorm/beta:01squeezenet/fire8/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire8/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire8/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/3x3/weights:0*squeezenet/fire8/expand/3x3/weights/Assign*squeezenet/fire8/expand/3x3/weights/read:02@squeezenet/fire8/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/3x3/BatchNorm/beta:01squeezenet/fire8/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire8/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire8/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire9/squeeze/weights:0'squeezenet/fire9/squeeze/weights/Assign'squeezenet/fire9/squeeze/weights/read:02=squeezenet/fire9/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire9/squeeze/BatchNorm/beta:0.squeezenet/fire9/squeeze/BatchNorm/beta/Assign.squeezenet/fire9/squeeze/BatchNorm/beta/read:02;squeezenet/fire9/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire9/squeeze/BatchNorm/moving_mean:05squeezenet/fire9/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire9/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire9/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire9/squeeze/BatchNorm/moving_variance:09squeezenet/fire9/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire9/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire9/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/1x1/weights:0*squeezenet/fire9/expand/1x1/weights/Assign*squeezenet/fire9/expand/1x1/weights/read:02@squeezenet/fire9/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/1x1/BatchNorm/beta:01squeezenet/fire9/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire9/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire9/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/3x3/weights:0*squeezenet/fire9/expand/3x3/weights/Assign*squeezenet/fire9/expand/3x3/weights/read:02@squeezenet/fire9/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/3x3/BatchNorm/beta:01squeezenet/fire9/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire9/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire9/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
squeezenet/conv10/weights:0 squeezenet/conv10/weights/Assign squeezenet/conv10/weights/read:026squeezenet/conv10/weights/Initializer/random_uniform:08
?
squeezenet/conv10/biases:0squeezenet/conv10/biases/Assignsqueezenet/conv10/biases/read:02,squeezenet/conv10/biases/Initializer/zeros:08
?
squeezenet/Bottleneck/weights:0$squeezenet/Bottleneck/weights/Assign$squeezenet/Bottleneck/weights/read:02:squeezenet/Bottleneck/weights/Initializer/random_uniform:08
?
&squeezenet/Bottleneck/BatchNorm/beta:0+squeezenet/Bottleneck/BatchNorm/beta/Assign+squeezenet/Bottleneck/BatchNorm/beta/read:028squeezenet/Bottleneck/BatchNorm/beta/Initializer/zeros:08
?
-squeezenet/Bottleneck/BatchNorm/moving_mean:02squeezenet/Bottleneck/BatchNorm/moving_mean/Assign2squeezenet/Bottleneck/BatchNorm/moving_mean/read:02?squeezenet/Bottleneck/BatchNorm/moving_mean/Initializer/zeros:0
?
1squeezenet/Bottleneck/BatchNorm/moving_variance:06squeezenet/Bottleneck/BatchNorm/moving_variance/Assign6squeezenet/Bottleneck/BatchNorm/moving_variance/read:02Bsqueezenet/Bottleneck/BatchNorm/moving_variance/Initializer/ones:0"??
	variables????
?
squeezenet/conv1/weights:0squeezenet/conv1/weights/Assignsqueezenet/conv1/weights/read:025squeezenet/conv1/weights/Initializer/random_uniform:08
?
!squeezenet/conv1/BatchNorm/beta:0&squeezenet/conv1/BatchNorm/beta/Assign&squeezenet/conv1/BatchNorm/beta/read:023squeezenet/conv1/BatchNorm/beta/Initializer/zeros:08
?
(squeezenet/conv1/BatchNorm/moving_mean:0-squeezenet/conv1/BatchNorm/moving_mean/Assign-squeezenet/conv1/BatchNorm/moving_mean/read:02:squeezenet/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
,squeezenet/conv1/BatchNorm/moving_variance:01squeezenet/conv1/BatchNorm/moving_variance/Assign1squeezenet/conv1/BatchNorm/moving_variance/read:02=squeezenet/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire2/squeeze/weights:0'squeezenet/fire2/squeeze/weights/Assign'squeezenet/fire2/squeeze/weights/read:02=squeezenet/fire2/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire2/squeeze/BatchNorm/beta:0.squeezenet/fire2/squeeze/BatchNorm/beta/Assign.squeezenet/fire2/squeeze/BatchNorm/beta/read:02;squeezenet/fire2/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire2/squeeze/BatchNorm/moving_mean:05squeezenet/fire2/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire2/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire2/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire2/squeeze/BatchNorm/moving_variance:09squeezenet/fire2/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire2/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire2/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/1x1/weights:0*squeezenet/fire2/expand/1x1/weights/Assign*squeezenet/fire2/expand/1x1/weights/read:02@squeezenet/fire2/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/1x1/BatchNorm/beta:01squeezenet/fire2/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire2/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire2/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire2/expand/3x3/weights:0*squeezenet/fire2/expand/3x3/weights/Assign*squeezenet/fire2/expand/3x3/weights/read:02@squeezenet/fire2/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire2/expand/3x3/BatchNorm/beta:01squeezenet/fire2/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire2/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire2/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire2/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire2/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire2/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire2/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire2/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire2/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire3/squeeze/weights:0'squeezenet/fire3/squeeze/weights/Assign'squeezenet/fire3/squeeze/weights/read:02=squeezenet/fire3/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire3/squeeze/BatchNorm/beta:0.squeezenet/fire3/squeeze/BatchNorm/beta/Assign.squeezenet/fire3/squeeze/BatchNorm/beta/read:02;squeezenet/fire3/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire3/squeeze/BatchNorm/moving_mean:05squeezenet/fire3/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire3/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire3/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire3/squeeze/BatchNorm/moving_variance:09squeezenet/fire3/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire3/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire3/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/1x1/weights:0*squeezenet/fire3/expand/1x1/weights/Assign*squeezenet/fire3/expand/1x1/weights/read:02@squeezenet/fire3/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/1x1/BatchNorm/beta:01squeezenet/fire3/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire3/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire3/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire3/expand/3x3/weights:0*squeezenet/fire3/expand/3x3/weights/Assign*squeezenet/fire3/expand/3x3/weights/read:02@squeezenet/fire3/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire3/expand/3x3/BatchNorm/beta:01squeezenet/fire3/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire3/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire3/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire3/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire3/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire3/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire3/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire3/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire3/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire4/squeeze/weights:0'squeezenet/fire4/squeeze/weights/Assign'squeezenet/fire4/squeeze/weights/read:02=squeezenet/fire4/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire4/squeeze/BatchNorm/beta:0.squeezenet/fire4/squeeze/BatchNorm/beta/Assign.squeezenet/fire4/squeeze/BatchNorm/beta/read:02;squeezenet/fire4/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire4/squeeze/BatchNorm/moving_mean:05squeezenet/fire4/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire4/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire4/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire4/squeeze/BatchNorm/moving_variance:09squeezenet/fire4/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire4/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire4/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/1x1/weights:0*squeezenet/fire4/expand/1x1/weights/Assign*squeezenet/fire4/expand/1x1/weights/read:02@squeezenet/fire4/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/1x1/BatchNorm/beta:01squeezenet/fire4/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire4/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire4/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire4/expand/3x3/weights:0*squeezenet/fire4/expand/3x3/weights/Assign*squeezenet/fire4/expand/3x3/weights/read:02@squeezenet/fire4/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire4/expand/3x3/BatchNorm/beta:01squeezenet/fire4/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire4/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire4/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire4/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire4/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire4/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire4/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire4/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire4/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire5/squeeze/weights:0'squeezenet/fire5/squeeze/weights/Assign'squeezenet/fire5/squeeze/weights/read:02=squeezenet/fire5/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire5/squeeze/BatchNorm/beta:0.squeezenet/fire5/squeeze/BatchNorm/beta/Assign.squeezenet/fire5/squeeze/BatchNorm/beta/read:02;squeezenet/fire5/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire5/squeeze/BatchNorm/moving_mean:05squeezenet/fire5/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire5/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire5/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire5/squeeze/BatchNorm/moving_variance:09squeezenet/fire5/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire5/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire5/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/1x1/weights:0*squeezenet/fire5/expand/1x1/weights/Assign*squeezenet/fire5/expand/1x1/weights/read:02@squeezenet/fire5/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/1x1/BatchNorm/beta:01squeezenet/fire5/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire5/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire5/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire5/expand/3x3/weights:0*squeezenet/fire5/expand/3x3/weights/Assign*squeezenet/fire5/expand/3x3/weights/read:02@squeezenet/fire5/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire5/expand/3x3/BatchNorm/beta:01squeezenet/fire5/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire5/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire5/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire5/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire5/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire5/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire5/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire5/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire5/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire6/squeeze/weights:0'squeezenet/fire6/squeeze/weights/Assign'squeezenet/fire6/squeeze/weights/read:02=squeezenet/fire6/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire6/squeeze/BatchNorm/beta:0.squeezenet/fire6/squeeze/BatchNorm/beta/Assign.squeezenet/fire6/squeeze/BatchNorm/beta/read:02;squeezenet/fire6/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire6/squeeze/BatchNorm/moving_mean:05squeezenet/fire6/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire6/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire6/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire6/squeeze/BatchNorm/moving_variance:09squeezenet/fire6/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire6/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire6/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/1x1/weights:0*squeezenet/fire6/expand/1x1/weights/Assign*squeezenet/fire6/expand/1x1/weights/read:02@squeezenet/fire6/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/1x1/BatchNorm/beta:01squeezenet/fire6/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire6/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire6/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire6/expand/3x3/weights:0*squeezenet/fire6/expand/3x3/weights/Assign*squeezenet/fire6/expand/3x3/weights/read:02@squeezenet/fire6/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire6/expand/3x3/BatchNorm/beta:01squeezenet/fire6/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire6/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire6/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire6/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire6/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire6/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire6/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire6/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire6/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire7/squeeze/weights:0'squeezenet/fire7/squeeze/weights/Assign'squeezenet/fire7/squeeze/weights/read:02=squeezenet/fire7/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire7/squeeze/BatchNorm/beta:0.squeezenet/fire7/squeeze/BatchNorm/beta/Assign.squeezenet/fire7/squeeze/BatchNorm/beta/read:02;squeezenet/fire7/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire7/squeeze/BatchNorm/moving_mean:05squeezenet/fire7/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire7/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire7/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire7/squeeze/BatchNorm/moving_variance:09squeezenet/fire7/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire7/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire7/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/1x1/weights:0*squeezenet/fire7/expand/1x1/weights/Assign*squeezenet/fire7/expand/1x1/weights/read:02@squeezenet/fire7/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/1x1/BatchNorm/beta:01squeezenet/fire7/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire7/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire7/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire7/expand/3x3/weights:0*squeezenet/fire7/expand/3x3/weights/Assign*squeezenet/fire7/expand/3x3/weights/read:02@squeezenet/fire7/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire7/expand/3x3/BatchNorm/beta:01squeezenet/fire7/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire7/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire7/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire7/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire7/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire7/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire7/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire7/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire7/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire8/squeeze/weights:0'squeezenet/fire8/squeeze/weights/Assign'squeezenet/fire8/squeeze/weights/read:02=squeezenet/fire8/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire8/squeeze/BatchNorm/beta:0.squeezenet/fire8/squeeze/BatchNorm/beta/Assign.squeezenet/fire8/squeeze/BatchNorm/beta/read:02;squeezenet/fire8/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire8/squeeze/BatchNorm/moving_mean:05squeezenet/fire8/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire8/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire8/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire8/squeeze/BatchNorm/moving_variance:09squeezenet/fire8/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire8/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire8/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/1x1/weights:0*squeezenet/fire8/expand/1x1/weights/Assign*squeezenet/fire8/expand/1x1/weights/read:02@squeezenet/fire8/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/1x1/BatchNorm/beta:01squeezenet/fire8/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire8/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire8/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire8/expand/3x3/weights:0*squeezenet/fire8/expand/3x3/weights/Assign*squeezenet/fire8/expand/3x3/weights/read:02@squeezenet/fire8/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire8/expand/3x3/BatchNorm/beta:01squeezenet/fire8/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire8/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire8/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire8/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire8/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire8/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire8/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire8/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire8/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
"squeezenet/fire9/squeeze/weights:0'squeezenet/fire9/squeeze/weights/Assign'squeezenet/fire9/squeeze/weights/read:02=squeezenet/fire9/squeeze/weights/Initializer/random_uniform:08
?
)squeezenet/fire9/squeeze/BatchNorm/beta:0.squeezenet/fire9/squeeze/BatchNorm/beta/Assign.squeezenet/fire9/squeeze/BatchNorm/beta/read:02;squeezenet/fire9/squeeze/BatchNorm/beta/Initializer/zeros:08
?
0squeezenet/fire9/squeeze/BatchNorm/moving_mean:05squeezenet/fire9/squeeze/BatchNorm/moving_mean/Assign5squeezenet/fire9/squeeze/BatchNorm/moving_mean/read:02Bsqueezenet/fire9/squeeze/BatchNorm/moving_mean/Initializer/zeros:0
?
4squeezenet/fire9/squeeze/BatchNorm/moving_variance:09squeezenet/fire9/squeeze/BatchNorm/moving_variance/Assign9squeezenet/fire9/squeeze/BatchNorm/moving_variance/read:02Esqueezenet/fire9/squeeze/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/1x1/weights:0*squeezenet/fire9/expand/1x1/weights/Assign*squeezenet/fire9/expand/1x1/weights/read:02@squeezenet/fire9/expand/1x1/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/1x1/BatchNorm/beta:01squeezenet/fire9/expand/1x1/BatchNorm/beta/Assign1squeezenet/fire9/expand/1x1/BatchNorm/beta/read:02>squeezenet/fire9/expand/1x1/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/1x1/BatchNorm/moving_mean:08squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/1x1/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/1x1/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/1x1/BatchNorm/moving_variance:0<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/1x1/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/1x1/BatchNorm/moving_variance/Initializer/ones:0
?
%squeezenet/fire9/expand/3x3/weights:0*squeezenet/fire9/expand/3x3/weights/Assign*squeezenet/fire9/expand/3x3/weights/read:02@squeezenet/fire9/expand/3x3/weights/Initializer/random_uniform:08
?
,squeezenet/fire9/expand/3x3/BatchNorm/beta:01squeezenet/fire9/expand/3x3/BatchNorm/beta/Assign1squeezenet/fire9/expand/3x3/BatchNorm/beta/read:02>squeezenet/fire9/expand/3x3/BatchNorm/beta/Initializer/zeros:08
?
3squeezenet/fire9/expand/3x3/BatchNorm/moving_mean:08squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/Assign8squeezenet/fire9/expand/3x3/BatchNorm/moving_mean/read:02Esqueezenet/fire9/expand/3x3/BatchNorm/moving_mean/Initializer/zeros:0
?
7squeezenet/fire9/expand/3x3/BatchNorm/moving_variance:0<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/Assign<squeezenet/fire9/expand/3x3/BatchNorm/moving_variance/read:02Hsqueezenet/fire9/expand/3x3/BatchNorm/moving_variance/Initializer/ones:0
?
squeezenet/conv10/weights:0 squeezenet/conv10/weights/Assign squeezenet/conv10/weights/read:026squeezenet/conv10/weights/Initializer/random_uniform:08
?
squeezenet/conv10/biases:0squeezenet/conv10/biases/Assignsqueezenet/conv10/biases/read:02,squeezenet/conv10/biases/Initializer/zeros:08
?
squeezenet/Bottleneck/weights:0$squeezenet/Bottleneck/weights/Assign$squeezenet/Bottleneck/weights/read:02:squeezenet/Bottleneck/weights/Initializer/random_uniform:08
?
&squeezenet/Bottleneck/BatchNorm/beta:0+squeezenet/Bottleneck/BatchNorm/beta/Assign+squeezenet/Bottleneck/BatchNorm/beta/read:028squeezenet/Bottleneck/BatchNorm/beta/Initializer/zeros:08
?
-squeezenet/Bottleneck/BatchNorm/moving_mean:02squeezenet/Bottleneck/BatchNorm/moving_mean/Assign2squeezenet/Bottleneck/BatchNorm/moving_mean/read:02?squeezenet/Bottleneck/BatchNorm/moving_mean/Initializer/zeros:0
?
1squeezenet/Bottleneck/BatchNorm/moving_variance:06squeezenet/Bottleneck/BatchNorm/moving_variance/Assign6squeezenet/Bottleneck/BatchNorm/moving_variance/read:02Bsqueezenet/Bottleneck/BatchNorm/moving_variance/Initializer/ones:0*?
serving_default}
4
image_batch%
image_batch:0??)

embeddings
embeddings:0	?tensorflow/serving/predict