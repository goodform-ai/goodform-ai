��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu6
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
t
v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namev/dense_2/bias
m
"v/dense_2/bias/Read/ReadVariableOpReadVariableOpv/dense_2/bias*
_output_shapes
:*
dtype0
t
m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namem/dense_2/bias
m
"m/dense_2/bias/Read/ReadVariableOpReadVariableOpm/dense_2/bias*
_output_shapes
:*
dtype0
|
v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namev/dense_2/kernel
u
$v/dense_2/kernel/Read/ReadVariableOpReadVariableOpv/dense_2/kernel*
_output_shapes

:@*
dtype0
|
m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namem/dense_2/kernel
u
$m/dense_2/kernel/Read/ReadVariableOpReadVariableOpm/dense_2/kernel*
_output_shapes

:@*
dtype0
t
v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namev/dense_1/bias
m
"v/dense_1/bias/Read/ReadVariableOpReadVariableOpv/dense_1/bias*
_output_shapes
:@*
dtype0
t
m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namem/dense_1/bias
m
"m/dense_1/bias/Read/ReadVariableOpReadVariableOpm/dense_1/bias*
_output_shapes
:@*
dtype0
}
v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*!
shared_namev/dense_1/kernel
v
$v/dense_1/kernel/Read/ReadVariableOpReadVariableOpv/dense_1/kernel*
_output_shapes
:	�@*
dtype0
}
m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*!
shared_namem/dense_1/kernel
v
$m/dense_1/kernel/Read/ReadVariableOpReadVariableOpm/dense_1/kernel*
_output_shapes
:	�@*
dtype0
q
v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namev/dense/bias
j
 v/dense/bias/Read/ReadVariableOpReadVariableOpv/dense/bias*
_output_shapes	
:�*
dtype0
q
m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namem/dense/bias
j
 m/dense/bias/Read/ReadVariableOpReadVariableOpm/dense/bias*
_output_shapes	
:�*
dtype0
y
v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namev/dense/kernel
r
"v/dense/kernel/Read/ReadVariableOpReadVariableOpv/dense/kernel*
_output_shapes
:	�*
dtype0
y
m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namem/dense/kernel
r
"m/dense/kernel/Read/ReadVariableOpReadVariableOpm/dense/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�@*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_123484

NoOpNoOp
�H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-0
&layer-37
'layer-38
(layer_with_weights-1
(layer-39
)layer-40
*layer_with_weights-2
*layer-41
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_default_save_signature
2	optimizer
3
signatures*
* 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 

:	keras_api* 

;	keras_api* 

<	keras_api* 

=	keras_api* 

>	keras_api* 

?	keras_api* 

@	keras_api* 

A	keras_api* 

B	keras_api* 

C	keras_api* 

D	keras_api* 

E	keras_api* 

F	keras_api* 

G	keras_api* 

H	keras_api* 

I	keras_api* 

J	keras_api* 

K	keras_api* 

L	keras_api* 

M	keras_api* 

N	keras_api* 

O	keras_api* 

P	keras_api* 

Q	keras_api* 

R	keras_api* 

S	keras_api* 

T	keras_api* 

U	keras_api* 

V	keras_api* 

W	keras_api* 

X	keras_api* 

Y	keras_api* 

Z	keras_api* 

[	keras_api* 
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator* 
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
0
h0
i1
w2
x3
�4
�5*
0
h0
i1
w2
x3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
1_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
o
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
YS
VARIABLE_VALUEm/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEv/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEm/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEv/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEm/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEv/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEv/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEm/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEv/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_ratem/dense/kernelv/dense/kernelm/dense/biasv/dense/biasm/dense_1/kernelv/dense_1/kernelm/dense_1/biasv/dense_1/biasm/dense_2/kernelv/dense_2/kernelm/dense_2/biasv/dense_2/biastotal_1count_1totalcountConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_124072
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	iterationlearning_ratem/dense/kernelv/dense/kernelm/dense/biasv/dense/biasm/dense_1/kernelv/dense_1/kernelm/dense_1/biasv/dense_1/biasm/dense_2/kernelv/dense_2/kernelm/dense_2/biasv/dense_2/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_124154��

��
�
!__inference__wrapped_model_122836
input_1=
*model_dense_matmul_readvariableop_resource:	�:
+model_dense_biasadd_readvariableop_resource:	�?
,model_dense_1_matmul_readvariableop_resource:	�@;
-model_dense_1_biasadd_readvariableop_resource:@>
,model_dense_2_matmul_readvariableop_resource:@;
-model_dense_2_biasadd_readvariableop_resource:
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOpX
model/reshape/ShapeShapeinput_1*
T0*
_output_shapes
::��k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :_
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/reshape/ReshapeReshapeinput_1$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:����������
2model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
4model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
4model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
,model/tf.__operators__.getitem/strided_sliceStridedSlicemodel/reshape/Reshape:output:0;model/tf.__operators__.getitem/strided_slice/stack:output:0=model/tf.__operators__.getitem/strided_slice/stack_1:output:0=model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskn
,model/tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$model/tf.compat.v1.gather_1/GatherV2GatherV25model/tf.__operators__.getitem/strided_slice:output:05model/tf.compat.v1.gather_1/GatherV2/indices:output:02model/tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������l
*model/tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :i
'model/tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
"model/tf.compat.v1.gather/GatherV2GatherV25model/tf.__operators__.getitem/strided_slice:output:03model/tf.compat.v1.gather/GatherV2/indices:output:00model/tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������a
model/tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply/MulMul+model/tf.compat.v1.gather/GatherV2:output:0%model/tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������c
model/tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply_1/MulMul-model/tf.compat.v1.gather_1/GatherV2:output:0'model/tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������|
model/tf.compat.v1.size/SizeSize5model/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
 model/tf.__operators__.add/AddV2AddV2model/tf.math.multiply/Mul:z:0 model/tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������e
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/tf.expand_dims/ExpandDims
ExpandDims$model/tf.__operators__.add/AddV2:z:0,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������i
'model/tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
%model/tf.compat.v1.floor_div/FloorDivFloorDiv%model/tf.compat.v1.size/Size:output:00model/tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: k
)model/tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
'model/tf.broadcast_to/BroadcastTo/shapePack)model/tf.compat.v1.floor_div/FloorDiv:z:02model/tf.broadcast_to/BroadcastTo/shape/1:output:02model/tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
!model/tf.broadcast_to/BroadcastToBroadcastTo(model/tf.expand_dims/ExpandDims:output:00model/tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
model/tf.math.subtract/SubSub5model/tf.__operators__.getitem/strided_slice:output:0*model/tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������n
,model/tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$model/tf.compat.v1.gather_5/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_5/GatherV2/indices:output:02model/tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������n
,model/tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$model/tf.compat.v1.gather_4/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_4/GatherV2/indices:output:02model/tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������c
model/tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply_4/MulMul-model/tf.compat.v1.gather_4/GatherV2:output:0'model/tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������c
model/tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply_5/MulMul-model/tf.compat.v1.gather_5/GatherV2:output:0'model/tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������g
model/tf.compat.v1.size_1/SizeSizemodel/tf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
"model/tf.__operators__.add_2/AddV2AddV2 model/tf.math.multiply_4/Mul:z:0 model/tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������g
%model/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
!model/tf.expand_dims_1/ExpandDims
ExpandDims&model/tf.__operators__.add_2/AddV2:z:0.model/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������k
)model/tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
'model/tf.compat.v1.floor_div_1/FloorDivFloorDiv'model/tf.compat.v1.size_1/Size:output:02model/tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: n
,model/tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :k
)model/tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$model/tf.compat.v1.gather_3/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_3/GatherV2/indices:output:02model/tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������n
,model/tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : k
)model/tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
$model/tf.compat.v1.gather_2/GatherV2GatherV2model/tf.math.subtract/Sub:z:05model/tf.compat.v1.gather_2/GatherV2/indices:output:02model/tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������m
+model/tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+model/tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
)model/tf.broadcast_to_1/BroadcastTo/shapePack+model/tf.compat.v1.floor_div_1/FloorDiv:z:04model/tf.broadcast_to_1/BroadcastTo/shape/1:output:04model/tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
#model/tf.broadcast_to_1/BroadcastToBroadcastTo*model/tf.expand_dims_1/ExpandDims:output:02model/tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������c
model/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply_2/MulMul-model/tf.compat.v1.gather_2/GatherV2:output:0'model/tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������c
model/tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.multiply_3/MulMul-model/tf.compat.v1.gather_3/GatherV2:output:0'model/tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
model/tf.math.subtract_2/SubSubmodel/tf.math.subtract/Sub:z:0,model/tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
"model/tf.__operators__.add_1/AddV2AddV2 model/tf.math.multiply_2/Mul:z:0 model/tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������n
,model/tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : k
)model/tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model/tf.compat.v1.gather_6/GatherV2GatherV2 model/tf.math.subtract_2/Sub:z:05model/tf.compat.v1.gather_6/GatherV2/indices:output:02model/tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
model/tf.math.subtract_1/SubSub&model/tf.__operators__.add_1/AddV2:z:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
"model/tf.compat.v1.norm_1/norm/mulMul-model/tf.compat.v1.gather_6/GatherV2:output:0-model/tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:~
4model/tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"model/tf.compat.v1.norm_1/norm/SumSum&model/tf.compat.v1.norm_1/norm/mul:z:0=model/tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
#model/tf.compat.v1.norm_1/norm/SqrtSqrt+model/tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
&model/tf.compat.v1.norm_1/norm/SqueezeSqueeze'model/tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
 model/tf.compat.v1.norm/norm/mulMul model/tf.math.subtract_1/Sub:z:0 model/tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������s
"model/tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
 model/tf.compat.v1.norm/norm/SumSum$model/tf.compat.v1.norm/norm/mul:z:0+model/tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(}
!model/tf.compat.v1.norm/norm/SqrtSqrt)model/tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:w
$model/tf.compat.v1.norm/norm/SqueezeSqueeze%model/tf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: c
model/tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/tf.math.multiply_6/MulMul-model/tf.compat.v1.norm/norm/Squeeze:output:0'model/tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: h
model/tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/tf.math.reduce_max/MaxMax/model/tf.compat.v1.norm_1/norm/Squeeze:output:0'model/tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
model/tf.math.maximum/MaximumMaximum model/tf.math.multiply_6/Mul:z:0%model/tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
model/tf.math.truediv/truedivRealDivmodel/tf.math.subtract/Sub:z:0!model/tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:���������d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape!model/tf.math.truediv/truediv:z:0model/flatten/Const:output:0*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
model/dense/Relu6Relu6model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������v
model/dropout/IdentityIdentitymodel/dense/Relu6:activations:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@n
model/dense_1/Relu6Relu6model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@y
model/dropout_1/IdentityIdentity!model/dense_1/Relu6:activations:0*
T0*'
_output_shapes
:���������@�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_2/SoftmaxSoftmaxmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������n
IdentityIdentitymodel/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
$__inference_signature_wrapper_123484
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_122836o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_123858

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@R
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_123816

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_122969p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_123111

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�r
�
A__inference_model_layer_call_and_return_conditional_losses_123350

inputs
dense_123332:	�
dense_123334:	�!
dense_1_123338:	�@
dense_1_123340:@ 
dense_2_123344:@
dense_2_123346:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_122853�
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlice reshape/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:����������
flatten/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_122938�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_123332dense_123334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_122951�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_123111�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_123338dense_1_123340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_122982�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123122�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_123344dense_2_123346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_123013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_123000

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

_
C__inference_reshape_layer_call_and_return_conditional_losses_122853

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_123811

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������S
Relu6Relu6BiasAdd:output:0*
T0*(
_output_shapes
:����������c
IdentityIdentityRelu6:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
ԍ
�
A__inference_model_layer_call_and_return_conditional_losses_123647

inputs7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOpQ
reshape/ShapeShapeinputs*
T0*
_output_shapes
::��e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:x
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:����������
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlicereshape/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapetf.math.truediv/truediv:z:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
dense/Relu6Relu6dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout/dropout/MulMuldense/Relu6:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������l
dropout/dropout/ShapeShapedense/Relu6:activations:0*
T0*
_output_shapes
::���
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_1/Relu6Relu6dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_1/dropout/MulMuldense_1/Relu6:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������@p
dropout_1/dropout/ShapeShapedense_1/Relu6:activations:0*
T0*
_output_shapes
::���
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_2/MatMulMatMul#dropout_1/dropout/SelectV2:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_1_layer_call_fn_123863

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123000o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_123791

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�u
�
A__inference_model_layer_call_and_return_conditional_losses_123233

inputs
dense_123215:	�
dense_123217:	�!
dense_1_123221:	�@
dense_1_123223:@ 
dense_2_123227:@
dense_2_123229:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_122853�
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlice reshape/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:����������
flatten/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_122938�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_123215dense_123217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_122951�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_122969�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_123221dense_1_123223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_122982�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123000�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_123227dense_2_123229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_123013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_123894

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_123013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_dropout_1_layer_call_fn_123868

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123122`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_123847

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_122982o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_123518

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_123350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_123501

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_123233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_122938

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_123880

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_123013

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_122982

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@R
Relu6Relu6BiasAdd:output:0*
T0*'
_output_shapes
:���������@b
IdentityIdentityRelu6:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_reshape_layer_call_fn_123767

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_122853d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_123122

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�~
�
A__inference_model_layer_call_and_return_conditional_losses_123762

inputs7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOpQ
reshape/ShapeShapeinputs*
T0*
_output_shapes
::��e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:x
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:����������
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlicereshape/Reshape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapetf.math.truediv/truediv:z:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
dense/Relu6Relu6dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������j
dropout/IdentityIdentitydense/Relu6:activations:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_1/Relu6Relu6dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@m
dropout_1/IdentityIdentitydense_1/Relu6:activations:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
"__inference__traced_restore_124154
file_prefix0
assignvariableop_dense_kernel:	�,
assignvariableop_1_dense_bias:	�4
!assignvariableop_2_dense_1_kernel:	�@-
assignvariableop_3_dense_1_bias:@3
!assignvariableop_4_dense_2_kernel:@-
assignvariableop_5_dense_2_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: 4
!assignvariableop_8_m_dense_kernel:	�4
!assignvariableop_9_v_dense_kernel:	�/
 assignvariableop_10_m_dense_bias:	�/
 assignvariableop_11_v_dense_bias:	�7
$assignvariableop_12_m_dense_1_kernel:	�@7
$assignvariableop_13_v_dense_1_kernel:	�@0
"assignvariableop_14_m_dense_1_bias:@0
"assignvariableop_15_v_dense_1_bias:@6
$assignvariableop_16_m_dense_2_kernel:@6
$assignvariableop_17_v_dense_2_kernel:@0
"assignvariableop_18_m_dense_2_bias:0
"assignvariableop_19_v_dense_2_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_m_dense_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_v_dense_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_m_dense_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_v_dense_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_m_dense_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_v_dense_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_m_dense_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_v_dense_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp$assignvariableop_16_m_dense_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_v_dense_2_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_m_dense_2_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_v_dense_2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

_
C__inference_reshape_layer_call_and_return_conditional_losses_123780

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�u
�
A__inference_model_layer_call_and_return_conditional_losses_123020
input_1
dense_122952:	�
dense_122954:	�!
dense_1_122983:	�@
dense_1_122985:@ 
dense_2_123014:@
dense_2_123016:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_122853�
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlice reshape/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:����������
flatten/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_122938�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_122952dense_122954*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_122951�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_122969�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_122983dense_1_122985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_122982�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123000�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_123014dense_2_123016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_123013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
&__inference_model_layer_call_fn_123248
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_123233o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
D
(__inference_dropout_layer_call_fn_123821

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_123111a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_123838

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ʯ
�
__inference__traced_save_124072
file_prefix6
#read_disablecopyonread_dense_kernel:	�2
#read_1_disablecopyonread_dense_bias:	�:
'read_2_disablecopyonread_dense_1_kernel:	�@3
%read_3_disablecopyonread_dense_1_bias:@9
'read_4_disablecopyonread_dense_2_kernel:@3
%read_5_disablecopyonread_dense_2_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: :
'read_8_disablecopyonread_m_dense_kernel:	�:
'read_9_disablecopyonread_v_dense_kernel:	�5
&read_10_disablecopyonread_m_dense_bias:	�5
&read_11_disablecopyonread_v_dense_bias:	�=
*read_12_disablecopyonread_m_dense_1_kernel:	�@=
*read_13_disablecopyonread_v_dense_1_kernel:	�@6
(read_14_disablecopyonread_m_dense_1_bias:@6
(read_15_disablecopyonread_v_dense_1_bias:@<
*read_16_disablecopyonread_m_dense_2_kernel:@<
*read_17_disablecopyonread_v_dense_2_kernel:@6
(read_18_disablecopyonread_m_dense_2_bias:6
(read_19_disablecopyonread_v_dense_2_bias:+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_m_dense_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_v_dense_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_10/DisableCopyOnReadDisableCopyOnRead&read_10_disablecopyonread_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp&read_10_disablecopyonread_m_dense_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_v_dense_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_m_dense_1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_v_dense_1_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_m_dense_1_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_v_dense_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_16/DisableCopyOnReadDisableCopyOnRead*read_16_disablecopyonread_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp*read_16_disablecopyonread_m_dense_2_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_17/DisableCopyOnReadDisableCopyOnRead*read_17_disablecopyonread_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp*read_17_disablecopyonread_v_dense_2_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:@}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_m_dense_2_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_19/DisableCopyOnReadDisableCopyOnRead(read_19_disablecopyonread_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp(read_19_disablecopyonread_v_dense_2_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
A__inference_dense_layer_call_and_return_conditional_losses_122951

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������S
Relu6Relu6BiasAdd:output:0*
T0*(
_output_shapes
:����������c
IdentityIdentityRelu6:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_123800

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_122951p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_123833

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_123785

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_122938`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_123885

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�r
�
A__inference_model_layer_call_and_return_conditional_losses_123130
input_1
dense_123102:	�
dense_123104:	�!
dense_1_123113:	�@
dense_1_123115:@ 
dense_2_123124:@
dense_2_123126:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_122853�
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&tf.__operators__.getitem/strided_sliceStridedSlice reshape/PartitionedCall:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:���������*

begin_mask*
end_maskh
&tf.compat.v1.gather_1/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_1/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0/tf.compat.v1.gather_1/GatherV2/indices:output:0,tf.compat.v1.gather_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������f
$tf.compat.v1.gather/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :c
!tf.compat.v1.gather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather/GatherV2GatherV2/tf.__operators__.getitem/strided_slice:output:0-tf.compat.v1.gather/GatherV2/indices:output:0*tf.compat.v1.gather/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������[
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply/MulMul%tf.compat.v1.gather/GatherV2:output:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_1/MulMul'tf.compat.v1.gather_1/GatherV2:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:���������p
tf.compat.v1.size/SizeSize/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes
: �
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0*
T0*'
_output_shapes
:���������_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims/ExpandDims
ExpandDimstf.__operators__.add/AddV2:z:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������c
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.floor_div/FloorDivFloorDivtf.compat.v1.size/Size:output:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*
_output_shapes
: e
#tf.broadcast_to/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#tf.broadcast_to/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
!tf.broadcast_to/BroadcastTo/shapePack#tf.compat.v1.floor_div/FloorDiv:z:0,tf.broadcast_to/BroadcastTo/shape/1:output:0,tf.broadcast_to/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to/BroadcastToBroadcastTo"tf.expand_dims/ExpandDims:output:0*tf.broadcast_to/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:����������
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0$tf.broadcast_to/BroadcastTo:output:0*
T0*+
_output_shapes
:���������h
&tf.compat.v1.gather_5/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_5/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_5/GatherV2/indices:output:0,tf.compat.v1.gather_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_4/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_4/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_4/GatherV2/indices:output:0,tf.compat.v1.gather_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������]
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_4/MulMul'tf.compat.v1.gather_4/GatherV2:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_5/MulMul'tf.compat.v1.gather_5/GatherV2:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*'
_output_shapes
:���������[
tf.compat.v1.size_1/SizeSizetf.math.subtract/Sub:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:���������a
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.expand_dims_1/ExpandDims
ExpandDims tf.__operators__.add_2/AddV2:z:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������e
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
!tf.compat.v1.floor_div_1/FloorDivFloorDiv!tf.compat.v1.size_1/Size:output:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*
_output_shapes
: h
&tf.compat.v1.gather_3/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :e
#tf.compat.v1.gather_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_3/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_3/GatherV2/indices:output:0,tf.compat.v1.gather_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������h
&tf.compat.v1.gather_2/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
tf.compat.v1.gather_2/GatherV2GatherV2tf.math.subtract/Sub:z:0/tf.compat.v1.gather_2/GatherV2/indices:output:0,tf.compat.v1.gather_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������g
%tf.broadcast_to_1/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%tf.broadcast_to_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
#tf.broadcast_to_1/BroadcastTo/shapePack%tf.compat.v1.floor_div_1/FloorDiv:z:0.tf.broadcast_to_1/BroadcastTo/shape/1:output:0.tf.broadcast_to_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
tf.broadcast_to_1/BroadcastToBroadcastTo$tf.expand_dims_1/ExpandDims:output:0,tf.broadcast_to_1/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:���������]
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_2/MulMul'tf.compat.v1.gather_2/GatherV2:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:���������]
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.multiply_3/MulMul'tf.compat.v1.gather_3/GatherV2:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:����������
tf.math.subtract_2/SubSubtf.math.subtract/Sub:z:0&tf.broadcast_to_1/BroadcastTo:output:0*
T0*+
_output_shapes
:����������
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_2/Mul:z:0tf.math.multiply_3/Mul:z:0*
T0*'
_output_shapes
:���������h
&tf.compat.v1.gather_6/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : e
#tf.compat.v1.gather_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
tf.compat.v1.gather_6/GatherV2GatherV2tf.math.subtract_2/Sub:z:0/tf.compat.v1.gather_6/GatherV2/indices:output:0,tf.compat.v1.gather_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes

:�
tf.math.subtract_1/SubSub tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:����������
tf.compat.v1.norm_1/norm/mulMul'tf.compat.v1.gather_6/GatherV2:output:0'tf.compat.v1.gather_6/GatherV2:output:0*
T0*
_output_shapes

:x
.tf.compat.v1.norm_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
tf.compat.v1.norm_1/norm/SumSum tf.compat.v1.norm_1/norm/mul:z:07tf.compat.v1.norm_1/norm/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
tf.compat.v1.norm_1/norm/SqrtSqrt%tf.compat.v1.norm_1/norm/Sum:output:0*
T0*
_output_shapes

:�
 tf.compat.v1.norm_1/norm/SqueezeSqueeze!tf.compat.v1.norm_1/norm/Sqrt:y:0*
T0*
_output_shapes
:*
squeeze_dims
 �
tf.compat.v1.norm/norm/mulMultf.math.subtract_1/Sub:z:0tf.math.subtract_1/Sub:z:0*
T0*'
_output_shapes
:���������m
tf.compat.v1.norm/norm/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
tf.compat.v1.norm/norm/SumSumtf.compat.v1.norm/norm/mul:z:0%tf.compat.v1.norm/norm/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(q
tf.compat.v1.norm/norm/SqrtSqrt#tf.compat.v1.norm/norm/Sum:output:0*
T0*
_output_shapes

:k
tf.compat.v1.norm/norm/SqueezeSqueezetf.compat.v1.norm/norm/Sqrt:y:0*
T0*
_output_shapes
: ]
tf.math.multiply_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
tf.math.multiply_6/MulMul'tf.compat.v1.norm/norm/Squeeze:output:0!tf.math.multiply_6/Mul/y:output:0*
T0*
_output_shapes
: b
tf.math.reduce_max/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tf.math.reduce_max/MaxMax)tf.compat.v1.norm_1/norm/Squeeze:output:0!tf.math.reduce_max/Const:output:0*
T0*
_output_shapes
: �
tf.math.maximum/MaximumMaximumtf.math.multiply_6/Mul:z:0tf.math.reduce_max/Max:output:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.subtract/Sub:z:0tf.math.maximum/Maximum:z:0*
T0*+
_output_shapes
:����������
flatten/PartitionedCallPartitionedCalltf.math.truediv/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_122938�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_123102dense_123104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_122951�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_123111�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_123113dense_1_123115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_122982�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_123122�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_123124dense_2_123126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_123013w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_122969

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_123905

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_123365
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_123350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-0
&layer-37
'layer-38
(layer_with_weights-1
(layer-39
)layer-40
*layer_with_weights-2
*layer-41
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_default_save_signature
2	optimizer
3
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
(
:	keras_api"
_tf_keras_layer
(
;	keras_api"
_tf_keras_layer
(
<	keras_api"
_tf_keras_layer
(
=	keras_api"
_tf_keras_layer
(
>	keras_api"
_tf_keras_layer
(
?	keras_api"
_tf_keras_layer
(
@	keras_api"
_tf_keras_layer
(
A	keras_api"
_tf_keras_layer
(
B	keras_api"
_tf_keras_layer
(
C	keras_api"
_tf_keras_layer
(
D	keras_api"
_tf_keras_layer
(
E	keras_api"
_tf_keras_layer
(
F	keras_api"
_tf_keras_layer
(
G	keras_api"
_tf_keras_layer
(
H	keras_api"
_tf_keras_layer
(
I	keras_api"
_tf_keras_layer
(
J	keras_api"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
(
L	keras_api"
_tf_keras_layer
(
M	keras_api"
_tf_keras_layer
(
N	keras_api"
_tf_keras_layer
(
O	keras_api"
_tf_keras_layer
(
P	keras_api"
_tf_keras_layer
(
Q	keras_api"
_tf_keras_layer
(
R	keras_api"
_tf_keras_layer
(
S	keras_api"
_tf_keras_layer
(
T	keras_api"
_tf_keras_layer
(
U	keras_api"
_tf_keras_layer
(
V	keras_api"
_tf_keras_layer
(
W	keras_api"
_tf_keras_layer
(
X	keras_api"
_tf_keras_layer
(
Y	keras_api"
_tf_keras_layer
(
Z	keras_api"
_tf_keras_layer
(
[	keras_api"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
L
h0
i1
w2
x3
�4
�5"
trackable_list_wrapper
L
h0
i1
w2
x3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
1_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
&__inference_model_layer_call_fn_123248
&__inference_model_layer_call_fn_123365
&__inference_model_layer_call_fn_123501
&__inference_model_layer_call_fn_123518�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
A__inference_model_layer_call_and_return_conditional_losses_123020
A__inference_model_layer_call_and_return_conditional_losses_123130
A__inference_model_layer_call_and_return_conditional_losses_123647
A__inference_model_layer_call_and_return_conditional_losses_123762�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_122836input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_reshape_layer_call_fn_123767�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_reshape_layer_call_and_return_conditional_losses_123780�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_123785�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_123791�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_123800�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_123811�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_123816
(__inference_dropout_layer_call_fn_123821�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_123833
C__inference_dropout_layer_call_and_return_conditional_losses_123838�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_123847�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_123858�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�@2dense_1/kernel
:@2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_123863
*__inference_dropout_1_layer_call_fn_123868�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_123880
E__inference_dropout_1_layer_call_and_return_conditional_losses_123885�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_123894�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_123905�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_123248input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_123365input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_123501inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_123518inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_123020input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_123130input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_123647inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_123762inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_123484input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_reshape_layer_call_fn_123767inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_reshape_layer_call_and_return_conditional_losses_123780inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_123785inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_123791inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_123800inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_123811inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_layer_call_fn_123816inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_123821inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_123833inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_123838inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_123847inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_123858inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_1_layer_call_fn_123863inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_1_layer_call_fn_123868inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_123880inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_123885inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_123894inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_123905inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
:	�2m/dense/kernel
:	�2v/dense/kernel
:�2m/dense/bias
:�2v/dense/bias
!:	�@2m/dense_1/kernel
!:	�@2v/dense_1/kernel
:@2m/dense_1/bias
:@2v/dense_1/bias
 :@2m/dense_2/kernel
 :@2v/dense_2/kernel
:2m/dense_2/bias
:2v/dense_2/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_122836ohiwx��0�-
&�#
!�
input_1���������
� "1�.
,
dense_2!�
dense_2����������
C__inference_dense_1_layer_call_and_return_conditional_losses_123858dwx0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_1_layer_call_fn_123847Ywx0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
C__inference_dense_2_layer_call_and_return_conditional_losses_123905e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
(__inference_dense_2_layer_call_fn_123894Z��/�,
%�"
 �
inputs���������@
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_123811dhi/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_123800Yhi/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_123880c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_123885c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
*__inference_dropout_1_layer_call_fn_123863X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
*__inference_dropout_1_layer_call_fn_123868X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
C__inference_dropout_layer_call_and_return_conditional_losses_123833e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_123838e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
(__inference_dropout_layer_call_fn_123816Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
(__inference_dropout_layer_call_fn_123821Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
C__inference_flatten_layer_call_and_return_conditional_losses_123791c3�0
)�&
$�!
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_flatten_layer_call_fn_123785X3�0
)�&
$�!
inputs���������
� "!�
unknown����������
A__inference_model_layer_call_and_return_conditional_losses_123020rhiwx��8�5
.�+
!�
input_1���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_123130rhiwx��8�5
.�+
!�
input_1���������
p 

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_123647qhiwx��7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_123762qhiwx��7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
&__inference_model_layer_call_fn_123248ghiwx��8�5
.�+
!�
input_1���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_123365ghiwx��8�5
.�+
!�
input_1���������
p 

 
� "!�
unknown����������
&__inference_model_layer_call_fn_123501fhiwx��7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_123518fhiwx��7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
C__inference_reshape_layer_call_and_return_conditional_losses_123780c/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������
� �
(__inference_reshape_layer_call_fn_123767X/�,
%�"
 �
inputs���������
� "%�"
unknown����������
$__inference_signature_wrapper_123484zhiwx��;�8
� 
1�.
,
input_1!�
input_1���������"1�.
,
dense_2!�
dense_2���������